import numpy as np
import os
import xarray as xr
from loss import WeatherBenchLoss_plain

# LOSS
loss_fn = WeatherBenchLoss_plain()

surface_data = np.load('/data/Pangu/output_data/output_surface.npy')
upper_data = np.load('/data/Pangu/output_data/output_upper.npy')

#print(surface_data.shape)
#print(upper_data.shape)

"""# The directory of your input and output data
data_dir = '/data/Pangu/test_inference/loss_upper.npy'
data_upper = np.load(data_dir)
print(f"loss_surface: {data_upper.shape}")"""



"""# Checkout surface preds for first 100 samples
print(f"*** PANGU PREDS ***")
data_dir = '/data/Pangu/test_inference/surface_PART1.h5'
ds_pred = xr.open_dataset(data_dir)
print(f"data shape: {ds_pred.data.shape}")
#print(f"(msl) data shape sample 0 : {ds_pred.data[0,0].shape}")
print(f"[MSL] data sample 0 : {np.mean(ds_pred.data.values[0,0])}")
print(f"[MSL] data sample 1 : {np.mean(ds_pred.data.values[1,0])}\n")

# Checkout surface GT for first 100 samples - MSL
print(f"*** [MSL] GROUND TRUTH ***")
data_dir = '/data/Pangu/test_inference/msl_2018_PART1.h5'
ds_gt = xr.open_dataset(data_dir)
#print(f"data shape: {ds_gt.msl.shape}")
print(f"[MSL] data shape: {ds_gt.msl.shape}")
print(f"[MSL] data sample 0 : {np.mean(ds_gt.msl.values[1])}")
print(f"[MSL] data sample 1 : {np.mean(ds_gt.msl.values[2])}")

# Checkout surface GT for first 100 samples - MSL
print(f"*** [MSL] GROUND TRUTH ***")
data_dir = '/data/Pangu/test_inference/t2m_2018_PART1.h5'
ds_t2m_gt = xr.open_dataset(data_dir)
#print(f"data shape: {ds_gt.msl.shape}")
print(f"[MSL] data shape: {ds_t2m_gt.t2m.shape}")
print(f"[MSL] data sample 0 : {np.mean(ds_t2m_gt.t2m.values[1])}")
print(f"[MSL] data sample 1 : {np.mean(ds_t2m_gt.t2m.values[2])}\n")

# Loss MSL
final_loss = []
for i in range(99):
    target = np.expand_dims(ds_gt.msl[i+1], axis=0)
    pred = np.expand_dims(ds_pred.data[i,0], axis=0)
    #print(f"Shape of target: {target.shape}")
    #print(f"Shape of pred: {pred.shape}")
    loss = loss_fn.compute_loss(target, pred)
    #print(f"Loss: {loss}")
    final_loss.append(loss)
final_loss = np.mean(final_loss)
print(f"[MSL] Final loss: {final_loss}\n")

# Loss T2M
final_loss = []
for i in range(99):
    target = np.expand_dims(ds_t2m_gt.t2m[i+1], axis=0)
    pred = np.expand_dims(ds_pred.data[i,3], axis=0)
    #print(f"Shape of target: {target.shape}")
    #print(f"Shape of pred: {pred.shape}")
    loss = loss_fn.compute_loss(target, pred)
    #print(f"Loss: {loss}")
    final_loss.append(loss)
final_loss = np.mean(final_loss)
print(f"[T2M] Final loss: {final_loss}")"""


# Checkout preds
data_dir = "/data/Pangu/test_inference/loss_upper.npy"
data_upper = np.load(data_dir)
print(f"loss shape: {data_upper.shape}")

# z 500, 700, 850
z850, z700, z500 = data_upper[0, [2,3,5]]
print(f"z850: {z850}")
print(f"z700: {z700}")
print(f"z500: {z500}\n")

# t 500, 700, 850
t850, t700, t500 = data_upper[2, [2,3,5]]
print(f"t850: {t850}")
print(f"t700: {t700}")
print(f"t500: {t500}\n")

# u 500, 700, 850
u850, u700, u500 = data_upper[3, [2,3,5]]
print(f"u850: {u850}")
print(f"u700: {u700}")
print(f"u500: {u500}\n")

# v 500, 700, 850
v850, v700, v500 = data_upper[4, [2,3,5]]
print(f"v850: {v850}")
print(f"v700: {v700}")
print(f"v500: {v500}")



