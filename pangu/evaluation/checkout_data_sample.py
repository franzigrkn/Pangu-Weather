import h5py
import numpy as np
import os
import time

data_dir = '/data/pangu'
path_vds_gt_surface = os.path.join(data_dir, 'raw_data/surface/msl_2018_PART1.h5')
path_vds_gt_surface_full = os.path.join(data_dir, 'raw_data/surface/msl_2018.nc')
print(f"gt surface: {path_vds_gt_surface}")
print(f"gt upper: {path_vds_gt_surface_full}")


with h5py.File(path_vds_gt_surface, "r") as f_gt_surface:
    print(f"dataset gt surface: {f_gt_surface}")
    print(f"Shape of gt surface : {f_gt_surface['msl'].shape}")

with h5py.File(path_vds_gt_surface_full, "r") as f_gt_full:
    print(f"dataset gt upper: {f_gt_full}")
    print(f"Shape of gt upper : {f_gt_full['msl'].shape}")