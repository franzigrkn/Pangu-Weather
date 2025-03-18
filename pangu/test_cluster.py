import numpy as np
import os
import onnx
import time
import onnxruntime as ort
import xarray as xr

data_dir = '/data/pangu/raw_data/input_data_test'

#print all files in data dir
print(os.listdir('/data/pangu'))

surface_data = np.load(os.path.join(data_dir, 'input_surface.npy')).astype(np.float32)
upper_data = np.load(os.path.join(data_dir, 'input_upper.npy')).astype(np.float32)

print(surface_data.shape)
print(upper_data.shape)

"""# The directory of your input and output data
data_dir_test = '/data/pangu/raw_data/msl_2018_month_01.nc'
ds = xr.open_dataset(data_dir_test)
print(f"ds: {ds}")"""

def inference(input_surface, input_upper, output_dir):
    start = time.time()
    # The directory of your input and output data
    print(f"Files in dir: {os.listdir('pangu')}")
    model = onnx.load('pangu/pangu_weather_6.onnx')

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
    ort_session = ort.InferenceSession('pangu/pangu_weather_6.onnx', sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])

    # Run the inference session
    output_upper, output_surface = ort_session.run(None, {'input':input_upper, 'input_surface':input_surface})
    print(f"Shape of output_upper: {output_upper.shape}")
    print(f"Shape of output_surface: {output_surface.shape}")
    # Save the results
    np.save(os.path.join(output_dir, 'output_upper'), output_upper)
    np.save(os.path.join(output_dir, 'output_surface'), output_surface)

    end = time.time()
    print(f"Time taken: {end-start}")

inference(surface_data, upper_data, data_dir)