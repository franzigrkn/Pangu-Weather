import numpy as np
import os
import xarray as xr

surface_data = np.load('/data/Pangu/output_data/output_surface.npy')
upper_data = np.load('/data/Pangu/output_data/output_upper.npy')

print(surface_data.shape)
print(upper_data.shape)

# The directory of your input and output data
data_dir = '/data/Pangu/input_data/z_test_data.nc'
ds = xr.open_dataset(data_dir)
print(f"ds: {ds}")
print(f"z test data (shape): {ds.latitude.shape}")
print(f"z test data: {ds.latitude.values[:10]}")
ds.close()

# Checkout h5 file
import h5py
data_dir = "/data/Pangu/output_data/surface.h5"
with h5py.File(data_dir, 'r') as f:
    #print(f"f ds: {f}")
    print(f"Keys of ds: {f.keys()}")
    print(f"ZARR shape: {f['data'].shape}")
    print(f"ZARR data: {f['data'][:2, :2, 0, 0]}\n")

# Compare values zarr
zarr_dir = "/data/Pangu/output_data/surface.zarr"
zarr_upper_data = xr.open_zarr(zarr_dir)
#print(f"Zarr loaded ds: {zarr_upper_data}")
#print(f"Zarr attributes: {zarr_upper_data._ARRAY_DIMENSIONS}")
print(f"Zarr attributes: {zarr_upper_data.data.shape}")
print(f"Zarr attributes: {zarr_upper_data.data.values[:2, :2, 0, 0]}")
