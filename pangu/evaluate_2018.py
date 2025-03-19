import os
import numpy as np
import onnx
import onnxruntime as ort
from netCDF4 import Dataset
import xarray as xr
import h5py
import time
import fsspec
import zarr

def inference_single_sample(ort_session, input_upper, input_surface):
    
    # Run the inference session
    output_upper, output_surface = ort_session.run(None, {'input':input_upper, 'input_surface':input_surface})
    print(f"Shape of output_upper: {output_upper.shape}")
    print(f"Shape of output_surface: {output_surface.shape}")

    return output_upper, output_surface

def model_pipeline(model_type=6):
    model = onnx.load(f'/data/pangu/checkpoints/pangu_weather_{model_type}.onnx')

    # Set the behavier of onnxruntime
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena=False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    # Increase the number for faster inference and more memory consumption
    options.intra_op_num_threads = 1

    # Set the behavier of cuda provider
    cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}

    # Initialize onnxruntime session for Pangu-Weather Models
    ort_session = ort.InferenceSession(f'/data/pangu/checkpoints/pangu_weather_{model_type}.onnx', sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])

    return ort_session

def get_zarr_name(base_dir):
        """
        Convert variable to zarr name.
        """
        # create zarr path for variable
        zarr_path_surface = f"{base_dir}/surface.zarr"
        zarr_path_upper = f"{base_dir}/upper.zarr"
        return zarr_path_upper, zarr_path_surface

def update_and_save_data(fs, output_upper, output_surface, output_dir, vars_atmospheric, vars_surface, press_levels, latitude_values, longitude_values):

    # Get zarr path for upper and surface
    zarr_path_upper, zarr_path_surface = get_zarr_name(output_dir)
    print(f"Zarr path upper: {zarr_path_upper}")
    print(f"Zarr path surface: {zarr_path_surface}")

    # make the ouput data a dataset with the same coordinates as input data
    ds_upper = xr.DataArray(output_upper, dims=['valid_time', 'vars', 'pressure_level', 'latitude', 'longitude'], coords={'vars': vars_atmospheric, 'pressure_level': press_levels, 'latitude': latitude_values, 'longitude': longitude_values}).to_dataset(name='data')
    ds_surface = xr.DataArray(output_surface, dims=['valid_time', 'vars', 'latitude', 'longitude'], coords={'vars': vars_surface, 'latitude': latitude_values, 'longitude': longitude_values}).to_dataset(name='data')
    print(f"ds_upper: {ds_upper}")
    print(f"ds_surface: {ds_surface}")

    # Specify the chunking
    chunking_surface = {"valid_time": 1, "vars": 4, "latitude": 721, "longitude": 1440}
    chunking_upper = {"valid_time": 1, "vars": 5, "pressure_level": 13, "latitude": 721, "longitude": 1440}

    # Re-chunk the dataset
    ds_upper = ds_upper.chunk(chunking_upper)
    ds_surface = ds_surface.chunk(chunking_surface)
    print(f"Defined chunking.")

    # UPPER
    if fs.exists(zarr_path_upper):
        mode = "a"
        append_dim = "valid_time"
        create = False
    else:
        mode = "w"
        append_dim = None
        create = True
    # Upload the data to the Zarr dataset
    mapper = fs.get_mapper(zarr_path_upper, create=create)
    ds_upper.to_zarr(
        mapper, 
        mode=mode, 
        consolidated=True,
        append_dim=append_dim,
    )
    print(f"Saved upper data to {zarr_path_upper}.")
    
    # SURFACE
    if fs.exists(zarr_path_surface):
        mode = "a"
        append_dim = "valid_time"
        create = False
    else:
        mode = "w"
        append_dim = None
        create = True
    # Upload the data to the Zarr dataset
    mapper = fs.get_mapper(zarr_path_surface, create=create)
    ds_surface.to_zarr(
        mapper, 
        mode=mode, 
        consolidated=True,
        append_dim=append_dim,
    )
    print(f"Saved surface data to {zarr_path_surface}.\n")

    return zarr_path_upper, zarr_path_surface

def create_virtual_source(mode, data_dir, vars, vds_filename):

    if mode=='upper':
        start = time.time()
        ex_ds_filename = os.path.join(data_dir, "atmospheric/z_2018.nc")
        with h5py.File(ex_ds_filename, 'r') as f:
            sh = f["z"].shape
            print(f"Shape atmospheric data: {sh}")
        layout = h5py.VirtualLayout(shape=(5,) + sh, dtype=np.float32)
        # Iterate over variables
        for i, var in enumerate(vars):
            entry_key = var
            #print(f"*** Creating virtual source for var: {var} ***")
            filename = os.path.join(data_dir, f"atmospheric/{var}_2018.nc")
            vsource = h5py.VirtualSource(filename, entry_key, shape=sh)
            layout[i, :, :, :, :] = vsource
        # Add virtual dataset to output file
        vds_path = os.path.join(data_dir, vds_filename)
        with h5py.File(vds_path, 'w', libver='latest') as f:
            f.create_virtual_dataset('data', layout, fillvalue=-1)
        # Check data
        with h5py.File(vds_path, "r") as f:
            print("Virtual dataset ATMOSPHERIC")
            print(f"Shape: {f['data'].shape}")
            #print(f["data"][:, :2, 10:12, 200:202, 400:402])
        end = time.time()
        print(f"Time taken for setting up virtual dataset ATMOSPHERIC: {end-start} secs.")
    elif mode=='surface':
        start = time.time()
        ex_ds_filename = os.path.join(data_dir, "surface/msl_2018.nc")
        with h5py.File(ex_ds_filename, 'r') as f:
            sh = f["msl"].shape
            print(f"Shape surface data: {sh}")
            layout = h5py.VirtualLayout(shape=(4,) + sh, dtype=np.float32)
        for i, var in enumerate(vars):
            # entry key 
            #print(f"*** Creating virtual source for var: {var} ***")
            filename = os.path.join(data_dir, f"surface/{var}_2018.nc")
            if var == 'u10m':
                entry_key = 'u10'
            elif var == 'v10m':
                entry_key = 'v10'
            else:
                entry_key = var
            vsource = h5py.VirtualSource(filename, entry_key, shape=sh)
            layout[i, :, :, :] = vsource
        # Add virtual dataset to output file
        vds_path = os.path.join(data_dir, vds_filename)
        with h5py.File(vds_path, 'w', libver='latest') as f:
            f.create_virtual_dataset('data', layout, fillvalue=-1)
        # Check data
        with h5py.File(vds_path, "r") as f:
            print("Virtual dataset SURFACE")
            print(f"Shape: {f['data'].shape}")
            # get length of dataset
            num_samples = f["data"].shape[0]
            #print(f["data"][:, :2, 200:202, 400:402])
        end = time.time()
        print(f"Time taken for setup virtaul dataset SURFACE: {end-start} secs.\n")

    return vds_path

def convert_zarr_to_h5(zarr_path, h5_path):
    """
    Convert a Zarr dataset to an HDF5 file.

    Parameters:
    zarr_path (str): Path to the Zarr dataset.
    h5_path (str): Path to save the HDF5 file.
    """
    # Open the Zarr dataset
    start = time.time()
    with xr.open_zarr(zarr_path) as ds:
        print(f"Loaded Zarr dataset from {zarr_path}")
        end = time.time()
        print(f"Time taken to load Zarr dataset: {end - start:.2f} seconds.\n")

        # Save the dataset as an HDF5 file
        start = time.time()
        ds.to_netcdf(h5_path)
        print(f"Saved HDF5 file to {h5_path}")
        end = time.time()
        print(f"Time taken to save HDF5 file: {end - start:.2f} seconds.\n")

def inference_2018():
    total_start = time.time()

    # Get filehandler
    fs = fsspec.filesystem("file")
    print(F"FILESYSTEM: {fs}")
    
    # The directory of your input and output data
    data_dir = '/data/pangu/raw_data'
    output_dir = '/data/pangu/preds'
    vars_surface = ['msl', 'u10m', 'v10m', 't2m'] # MSLP, U10, V10, T2M in the exact order
    vars_atmospheric = ['z', 'q', 't', 'u', 'v'] # Z, Q, T, U and V in the exact order

    # VDS DATASET: ATMOSPHERIC & SURFACE
    path_vds_upper = create_virtual_source(mode='upper', data_dir=data_dir, vars=vars_atmospheric, vds_filename='VDS_atmospheric.h5')
    path_vds_surface = create_virtual_source(mode='surface', data_dir=data_dir, vars=vars_surface, vds_filename='VDS_surface.h5')
    
    # create ort session
    ort_session = model_pipeline(6)

    # Extract latitude and longitude values for later
    with xr.open_dataset(os.path.join(data_dir, "surface/msl_2018.nc")) as ds_example: # load as an example to extract lat and long attributes
        latitude_values = ds_example['latitude'].values
        longitude_values = ds_example['longitude'].values

    # Iterate over all samples in 2018
    with h5py.File(path_vds_surface, "r") as f_surface:
        with h5py.File(path_vds_upper, "r") as f_atmospheric:
            print(f"Size of surface data: {f_surface['data'].shape}")
            print(f"Size of atmosphericdata: {f_atmospheric['data'].shape}")
            #for id in range(num_samples):
            for id in range(3):
                start = time.time()
                # Load the upper-air numpy arrays
                input_upper = f_atmospheric["data"][:, id, :, :, :]
                input_surface = f_surface["data"][:, id, :, :]
                print(f"Shape of input_upper: {input_upper.shape}")
                print(f"Shape of input_surface: {input_surface.shape}")

                # check whether all values are equal to zero
                assert not np.all(input_upper == -1)
                assert not np.all(input_surface == -1)

                # Change dtype to float32
                if input_upper.dtype != np.float32:
                    input_upper = input_upper.astype(np.float32)
                if input_surface.dtype != np.float32:
                    input_surface = input_surface.astype(np.float32)
                assert(np.isnan(input_upper).any()==False)
                assert(np.isnan(input_surface).any()==False)
                end = time.time()
                print(f"Time taken for loading one sample: {end-start} secs. \n")

                # Run inference
                start = time.time()
                output_upper, output_surface = inference_single_sample(ort_session, input_upper, input_surface)
                output_upper = np.expand_dims(output_upper, axis=0)
                output_surface = np.expand_dims(output_surface, axis=0)
                print(f"Output upper shape: {output_upper.shape}")
                print(f"Output surface shape: {output_surface.shape}\n")
                end = time.time()
                print(f"Evaluated data for sample {id}.")
                print(f"Time taken for processing one sample: {end-start} secs. \n")

                # Save using zarr chunks
                print(f"Now saving ...")
                start = time.time()
                press_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
                zarr_path_upper, zarr_path_surface = update_and_save_data(fs, output_upper, output_surface, output_dir, vars_atmospheric, vars_surface, press_levels, latitude_values, longitude_values)
                end = time.time()
                print(f"Time taken for saving one sample: {end-start} secs. \n")

    print(f"Finished inference for 2018.")
    total_end = time.time()
    print(f"Total time taken: {total_end-total_start} secs.")

    # Convert to h5 file
    h5_path_upper = f"{output_dir}/upper.h5"
    h5_path_surface = f"{output_dir}/surface.h5"
    convert_zarr_to_h5(zarr_path_upper, h5_path_upper)
    convert_zarr_to_h5(zarr_path_surface, h5_path_surface)

    # Compare data
    print(f"*** COMPARE H5 AND ZARR FILE ***")
    # H5
    with h5py.File(h5_path_surface, 'r') as f:
        print(f"CHECK 1: H5 DATA")
        print(f"ZARR shape: {f['data'].shape}")
        print(f"ZARR data: {f['data'][:2, :2, 0, 0]}\n")
    # ZARR
    zarr_upper_data = xr.open_zarr(zarr_path_surface)
    print(f"CHECK 2: ZARR DATA")
    print(f"Zarr attributes: {zarr_upper_data.data.shape}")
    print(f"Zarr attributes: {zarr_upper_data.data.values[:2, :2, 0, 0]}")

# Start inference!
inference_2018()

