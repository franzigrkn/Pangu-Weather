import os
import numpy as np
import onnx
import onnxruntime as ort
from netCDF4 import Dataset
import xarray as xr
import h5py


vars=['msl', 't2m']
# get only the first 50 samples

# Layout
data_dir = '/data/Pangu/input_data'

# Split datasets
for i, var in enumerate(vars):
    print(f"*** VAR: {var} ***")
    entry_key = f"{var}"
    filename = os.path.join(data_dir, f"{var}_2018_month_01.h5")
    with xr.open_dataset(filename) as ds:
        print(f"Opened xr dataset: {ds}")
        # Select only the first 50
        ds = ds.isel(valid_time=slice(0, 50))
        print(f"Selected first 50 samples: {ds}")
        # Save to new file
        new_filename = os.path.join(data_dir, f"{var}_2018_month_01_first50.h5")
        ds.to_netcdf(new_filename)


"""# Layout
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
    print(f["data"][:, :2, :2, :2])"""