import numpy as np
import os
import xarray as xr

surface_data = np.load('output_data/output_surface.npy')
upper_data = np.load('output_data/output_upper.npy')

print(surface_data.shape)
print(upper_data.shape)

# The directory of your input and output data
data_dir = 'input_data/z_test_data.nc'
ds = xr.open_dataset(data_dir)
print(f"ds: {ds}")
print(f"z test data (shape): {ds.z.shape}")
print(f"z test data: {ds.z.values[0, 0, :10, :10]}")
ds.close()