import h5py
import numpy as np
import os
import time

import tables
tables.file._open_files.close_all()

data_dir = '/data/pangu'
path_vds_gt_surface = os.path.join(data_dir, 'virtual_ds/evaluation/VDS_GT_surface.h5')
path_vds_gt_upper = os.path.join(data_dir, 'virtual_ds/evaluation/VDS_GT_atmospheric.h5')


with h5py.File(path_vds_gt_surface, "r") as f_gt_surface:
    print(f"dataset gt surface: {f_gt_surface}")

with h5py.File(path_vds_gt_upper, "r") as f_gt_upper:
    print(f"dataset gt upper: {f_gt_upper}")