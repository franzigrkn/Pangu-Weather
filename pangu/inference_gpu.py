import os
import numpy as np
import onnx
import onnxruntime as ort
import xarray as xr

vars=['msl', 't2m']

# The directory of your input and output data
input_data_dir = 'input_data'
output_data_dir = 'output_data'
model_24 = onnx.load('checkpoints/pangu_weather_24.onnx')

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
ort_session_24 = ort.InferenceSession('checkpoints/pangu_weather_24.onnx', sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])

# Load the upper-air numpy arrays
input_upper = np.load(os.path.join(input_data_dir, 'input_upper.npy')).astype(np.float32)
print(f"Shape of input_upper: {input_upper.shape}")
# Load the surface numpy arrays
#input_surface = np.load(os.path.join(input_data_dir, 'input_surface.npy')).astype(np.float32)
ds_surface = xr.open_mfdataset(
    paths=[input_data_dir + f"/{var}_2018_month_01.nc" for var in vars],
    concat_dim=['vars'],
    combine='nested'
)
print(f"DS new: {ds_surface}")
# get one datasample
input_surface = ds_surface.isel(valid_time=0).msl.values.squeeze()
print(f"Shape of input_surface: {input_surface.shape}")

# Run the inference session
output_upper, output_surface = ort_session_24.run(None, {'input':input_upper, 'input_surface':input_surface})
print(f"Shape of output_upper: {output_upper.shape}")
print(f"Shape of output_surface: {output_surface.shape}")
# Save the results
np.save(os.path.join(output_data_dir, 'output_upper_test'), output_upper)
np.save(os.path.join(output_data_dir, 'output_surface_test'), output_surface)