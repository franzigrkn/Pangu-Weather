import os
import numpy as np
import onnx
import onnxruntime as ort
from netCDF4 import Dataset
import xarray as xr
import h5py

vars=['msl', 't2m']

# Checkout datasets
""" # The directory of your input and output data
data_dir = 'input_data'
input_msl = xr.open_dataset(os.path.join(data_dir, 'msl_2018_month_01.h5'))
input_t2m = xr.open_dataset(os.path.join(data_dir, 't2m_2018_month_01.h5'))

print(f"*** MSL dataset ***")
print(input_msl)
print(f"*** T2M dataset ***")
print(input_t2m)

input_msl.close()
input_t2m.close()"""

# Merge
"""ds_surface = xr.open_mfdataset(
    paths=[data_dir + f"/{var}_2018_month_01.h5" for var in vars],
    concat_dim=['vars'],
    combine='nested',
    data_vars='all',
)
print(f"DS new: {ds_surface}")
print(f"ds surface coords: {ds_surface.dims}\n")
print(f"ds surface coords: {ds_surface.coords}\n")
ds_surface.coords['vars'] = ["msl", "t2m"]
print(f"ds surface coords: {ds_surface.coords}\n")"""

# Layout
data_dir = '/data/Pangu/input_data'
filename = os.path.join(data_dir, "msl_2018_month_01.h5")
sh = h5py.File(filename, 'r')["msl"].shape
print(f"Shape: {sh}")
layout = h5py.VirtualLayout(shape=(2,) + sh, dtype=np.float32)

for i, var in enumerate(vars):
    # entry key 
    entry_key = f"{var}"
    print(f"*** VAR: {var} ***")
    filename = os.path.join(data_dir, f"{var}_2018_month_01.h5")
    sh = h5py.File(filename, 'r')[f"{entry_key}"].shape
    vsource = h5py.VirtualSource(filename, entry_key, shape=sh)
    layout[i, :, :, :] = vsource

# Add virtual dataset to output file
with h5py.File(os.path.join(data_dir, "VDS.h5"), 'w', libver='latest') as f:
    f.create_virtual_dataset('data', layout, fillvalue=0)

with h5py.File(os.path.join(data_dir, "VDS.h5"), "r") as f:
    print("Virtual dataset:")
    print(f"Virtual dataset shape: {f['data'].shape}")
    print(f"Virtual dataset: {f}")
    print(f["data"][:, :2, :2, :2])


"""# get one datasample
input_surface = ds_surface.isel(valid_time=0).msl.values.squeeze()
print(f"Shape of input_surface: {input_surface.shape}")"""

