import os
import numpy as np
import onnx
import onnxruntime as ort
from netCDF4 import Dataset
import xarray as xr
import h5py

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

def update_and_save_data(output_upper, output_surface, output_dir):

    # Get zarr path for upper and surface
    zarr_path_upper, zarr_path_surface = get_zarr_name(output_dir)
    print(f"Zarr path upper: {zarr_path_upper}")
    print(f"Zarr path surface: {zarr_path_surface}")

    # make the ouput data a dataset with the same coordinates as input data
    ds_upper = xr.DataArray(output_upper, dims=['vars', 'level', 'lat', 'lon'], coords={'vars': vars_atmospheric, 'level': ds_atmospheric['level'], 'lat': ds_atmospheric['lat'], 'lon': ds_atmospheric['lon']}).to_dataset(name='data')
    ds_surface = xr.DataArray(output_surface, dims=['vars', 'lat', 'lon'], coords={'vars': vars_surface, 'lat': ds_surface['lat'], 'lon': ds_surface['lon']}).to_dataset(name='data')
    print(f"ds_upper: {ds_upper}")
    print(f"ds_surface: {ds_surface}")

    # Specify the chunking
    chunking = {"valid_time": 1, "latitude": 721, "longitude": 1440}
    # TODO: How to chunk vars? Want to have all dims included
    """if "level" in ds.dims:
        chunking["level"] = 1"""

    # Re-chunk the dataset
    ds_upper = ds_upper.chunk(chunking)
    ds_surface = ds_surface.chunk(chunking)

    # UPPER
    # Check if the Zarr dataset exists
    if self.fs.exists(zarr_path_upper):
        mode = "a"
        append_dim = "valid_time"
        create = False
    else:
        mode = "w"
        append_dim = None
        create = True
    # Upload the data to the Zarr dataset
    mapper = self.fs.get_mapper(zarr_path_upper, create=create)
    ds_upper.to_zarr(
        mapper, 
        mode=mode, 
        consolidated=True,
        append_dim=append_dim,
    )

    # SURFACE
    # Check if the Zarr dataset exists
    if self.fs.exists(zarr_path_surface):
        mode = "a"
        append_dim = "valid_time"
        create = False
    else:
        mode = "w"
        append_dim = None
        create = True
    # Upload the data to the Zarr dataset
    mapper = self.fs.get_mapper(zarr_path_surface, create=create)
    ds_surface.to_zarr(
        mapper, 
        mode=mode, 
        consolidated=True,
        append_dim=append_dim,
    )
    
def inference_2018():
    
    # The directory of your input and output data
    data_dir = '/data/pangu/raw_data'
    output_dir = '/data/pangu/preds'
    vars_surface = ['msl', 'u10m', 'v10m', 't2m'] # MSLP, U10, V10, T2M in the exact order
    vars_atmospheric = ['z', 'q', 't', 'u', 'v'] # Z, Q, T, U and V in the exact order

    # ATMOSPHERIC
    filename = os.path.join(data_dir, "atmospheric/z_2018.nc")
    with h5py.File(filename, 'r') as f:
        sh = f["z"].shape
        print(f"Shape atmospheric data: {sh}")
    layout = h5py.VirtualLayout(shape=(5,) + sh, dtype=np.float32)
    # Iterate over variables
    for i, var in enumerate(vars_atmospheric):
        entry_key = var
        print(f"*** Creating virtual source for var: {var} ***")
        filename = os.path.join(data_dir, f"atmospheric/{var}_2018.nc")
        vsource = h5py.VirtualSource(filename, entry_key, shape=sh)
        layout[i, :, :, :, :] = vsource
    # Add virtual dataset to output file
    with h5py.File(os.path.join(data_dir, "VDS_atmospheric.h5"), 'w', libver='latest') as f:
        f.create_virtual_dataset('data', layout, fillvalue=5)
    # Check data
    with h5py.File(os.path.join(data_dir, "VDS_atmospheric.h5"), "r") as f:
        print("Virtual dataset ATMOSPHERIC")
        print(f"Shape: {f['data'].shape}")
        print(f["data"][:, :2, 10:12, 200:202, 400:402])

    # SURFACE
    filename = os.path.join(data_dir, "surface/msl_2018.nc")
    with h5py.File(filename, 'r') as f:
        sh = f["msl"].shape
        print(f"Shape surface data: {sh}")
        layout = h5py.VirtualLayout(shape=(4,) + sh, dtype=np.float32)
    for i, var in enumerate(vars_surface):
        # entry key 
        entry_key = var
        print(f"*** Crating virtual source for var: {var} ***")
        filename = os.path.join(data_dir, f"surface/{var}_2018.nc")
        vsource = h5py.VirtualSource(filename, entry_key, shape=sh)
        layout[i, :, :, :] = vsource
    # Add virtual dataset to output file
    with h5py.File(os.path.join(data_dir, "VDS_surface.h5"), 'w', libver='latest') as f:
        f.create_virtual_dataset('data', layout, fillvalue=5)
    # Check data
    with h5py.File(os.path.join(data_dir, "VDS_surface.h5"), "r") as f:
        print("Virtual dataset SURFACE")
        print(f"Shape: {f['data'].shape}")
        # get length of dataset
        num_samples = f["data"].shape[0]
        print(f["data"][:, :2, 200:202, 400:402])

    # create ort session
    ort_session = model_pipeline(6)

    # Iterate over all samples in 2018
    #for id in range(num_samples):
    with h5py.File(os.path.join(data_dir, "VDS_surface.h5"), "r") as f_surface:
        with h5py.File(os.path.join(data_dir, "VDS_atmospheric.h5"), "r") as f_atmospheric:
            print(f"Size of surface data: {f_surface['data'].shape}")
            print(f"Size of atmosphericdata: {f_atmospheric['data'].shape}")
            for id in range(20):
                # Load the upper-air numpy arrays
                input_upper = f_atmospheric["data"][:, id, :, :, :]
                input_surface = f_surface["data"][:, id, :, :]
                print(f"Shape of input_upper: {input_upper.shape}")
                print(f"Shape of input_surface: {input_surface.shape}")
                print(f"Input data upper: {input_upper[:4, 6:8, 200:202, 400:402]}")
                print(f"Input data surface: {input_surface[:4, 200:202, 400:402]}\n")

                # check whether all values are equal to zero
                #assert not np.all(input_upper == 0)
                #assert not np.all(input_surface == 0)

                # Change dtype to float32
                if input_upper.dtype != np.float32:
                    input_upper = input_upper.astype(np.float32)
                if input_surface.dtype != np.float32:
                    input_surface = input_surface.astype(np.float32)
                assert(np.isnan(input_upper).any()==False)
                assert(np.isnan(input_surface).any()==False)

                output_upper, output_surface = inference_single_sample(ort_session, input_upper, input_surface)
                print(f"Output upper shape: {output_upper.shape}")
                print(f"Output surface shape: {output_surface.shape}\n")

                # delete input data and output data
                #del input_upper, input_surface, output_upper, output_surface

                # think about a method to save the data, maybe merge all in a h5 file that we append to 
                #update_and_save_data(output_upper, output_surface, output_dir)
    
    """# Convert the surface variables
    zarr_path_upper, zarr_path_surface = get_zarr_name(output_dir)
    # SURFACE
    zarr_upper_data = xr.open_zarr(zarr_path_surface) # 1640x4x721x1440
    print(f"Shape of data: {zarr_upper_data.data.shape}")
    print(f"Data: {zarr_upper_data.data.values[:5, :2, :2, :2, :2]}")
    # UPPER"""


        
    

inference_2018()

