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

###############################
mode = 'surface' # surface
#parts = [1,2,3,4,5,6,7,8,9]
parts = [10,11,12,13,14,15,16,17]
###############################

# The directory of your input and output data
data_dir = '/data/pangu/raw_data'
vars_surface = ['msl', 'u10m', 'v10m', 't2m'] # MSL, U10, V10, T2M in the exact order
vars_atmospheric = ['z', 'q', 't', 'u', 'v'] # Z, Q, T, U and V in the exact order

# ATMOSPHERE
if mode == 'atmospheric':
    for part in parts:
        print(f"*** PART: {part} ***")
        start_part = time.time()
        start_sample = (part - 1) * 100
        if part*100 > 1640:
            end_sample = 1640
        else:
            end_sample = part * 100
        for var in vars_atmospheric:
            print(f"*** VAR: {var} ***")
            start_var = time.time()
            entry_key = f"{var}"
            filename = os.path.join(data_dir, f"atmospheric/{var}_2018.nc")
            with xr.open_dataset(filename) as ds:
                with ds.isel(valid_time=slice(start_sample, end_sample)) as ds_cut:
                    print(f"Selected samples from {start_sample} to {end_sample}: {ds_cut}")
                    # Save to new file
                    new_filename = os.path.join(data_dir, f"atmospheric/{var}_2018_PART{part}.h5")
                    ds_cut.to_netcdf(new_filename)
            # Time
            end_var = time.time()
            print(f"VAR {var} - Time taken to process: {end_var - start_var}")
        end_part = time.time()
        print(f"Time taken to process part {part}: {end_part - start_part}\n")

# SURFACE
elif mode == 'surface':
    for part in parts:
        print(f"*** PART: {part} ***")
        start_part = time.time()
        start_sample = (part - 1) * 100
        end_sample = part * 100
        for var in vars_surface:
            print(f"*** VAR: {var} ***")
            start_var = time.time()
            entry_key = f"{var}"
            filename = os.path.join(data_dir, f"surface/{var}_2018.nc")
            with xr.open_dataset(filename) as ds:
                with ds.isel(valid_time=slice(start_sample, end_sample)) as ds_cut:
                    print(f"Selected samples from {start_sample} to {end_sample}: {ds_cut}")
                    # Save to new file
                    new_filename = os.path.join(data_dir, f"surface/{var}_2018_PART{part}.h5")
                    ds_cut.to_netcdf(new_filename)
            # Time
            end_var = time.time()
            print(f"VAR {var} - Time taken to process: {end_var - start_var}")
        end_part = time.time()
        print(f"Time taken to process part {part}: {end_part - start_part}\n")

print(f"Finished the extraction of the data for {mode} variables.")
final_end = time.time()
print(f"Total time taken: {final_end-start_var}")
