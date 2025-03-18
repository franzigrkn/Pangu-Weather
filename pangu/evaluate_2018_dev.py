import os
import numpy as np
import onnx
import onnxruntime as ort
from netCDF4 import Dataset
import xarray as xr
import h5py
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

def update_and_save_data(fs, output_upper, output_surface, output_dir, vars_atmospheric, vars_surface, press_levels, ds_surface_era5):

    # Get zarr path for upper and surface
    zarr_path_upper, zarr_path_surface = get_zarr_name(output_dir)
    #print(f"Zarr path upper: {zarr_path_upper}")
    #print(f"Zarr path surface: {zarr_path_surface}")

    # make the ouput data a dataset with the same coordinates as input data
    ds_upper = xr.DataArray(output_upper, dims=['valid_time', 'vars', 'pressure_level', 'latitude', 'longitude'], coords={'vars': vars_atmospheric, 'pressure_level': press_levels, 'latitude': ds_surface_era5['latitude'], 'longitude': ds_surface_era5['longitude']}).to_dataset(name='data')
    ds_surface = xr.DataArray(output_surface, dims=['valid_time', 'vars', 'latitude', 'longitude'], coords={'vars': vars_surface, 'latitude': ds_surface_era5['latitude'], 'longitude': ds_surface_era5['longitude']}).to_dataset(name='data')
    #print(f"ds_upper: {ds_upper}")
    #print(f"ds_surface: {ds_surface}")

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
    
def inference_2018():

    # Get filehandler
    fs = fsspec.filesystem("file")
    print(F"FILESYSTEM: {fs}")

    # create ort session
    ort_session = model_pipeline(6)

    # load test data
    output_dir = "/data/pangu/output_data"
    surface_data = np.load('/data/pangu/input_data/input_surface.npy')
    upper_data = np.load('/data/pangu/input_data/input_upper.npy')

    # Iterate over all samples in 2018
    print(f"Size of surface data: {surface_data.shape}")
    print(f"Size of data: {upper_data.shape}")
    for id in range(7):
        output_upper, output_surface = inference_single_sample(ort_session,  upper_data, surface_data)
        output_upper = np.expand_dims(output_upper, axis=0)
        output_surface = np.expand_dims(output_surface, axis=0)
        print(f"Output upper shape: {output_upper.shape}")
        print(f"Output surface shape: {output_surface.shape}\n")

        # think about a method to save the data, maybe merge all in a h5 file that we append to 
        ds_surface = xr.open_dataset('/data/pangu/input_data/msl_2018_month_01.nc') # load as an example to extract lat and long attributes
        vars_atmospheric = ['z', 'q', 't', 'u', 'v']
        vars_surface = ['msl', 'u10m', 'v10m', 't2m']
        press_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
        update_and_save_data(fs, output_upper, output_surface, output_dir, vars_atmospheric, vars_surface, press_levels, ds_surface)
        
    print(f"Finished inference for 2018.")
    # Checkout zarr data
    zarr_path_upper, zarr_path_surface = get_zarr_name(output_dir)
    zarr_upper_data = xr.open_zarr(zarr_path_upper)
    print(f"Shape of data: {zarr_upper_data.data.shape}")
    print(f"Data: {zarr_upper_data.data.values[:5, :2, :2, :2, :2]}")

    # Save as h5 file - opening and transforming the zarr file to h5 - using xr or h5py? Probabl;y xr oom issue


inference_2018()

