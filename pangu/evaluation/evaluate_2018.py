
import h5py
import numpy as np
import os
import time

from loss import WeatherBenchLoss_plain

#####################
mode = 'atmosphere' # surface or atmosphere
every_2nd = True
#####################

# data dirs
preds_dir = "/data/preds"
data_dir = '/data/pangu' #/raw_data'
save_dir = os.path.join(data_dir, 'evaluation')
vars_surface = ['msl', 'u10m', 'v10m', 't2m'] # MSL, U10, V10, T2M in the exact order
vars_atmospheric = ['z', 'q', 't', 'u', 'v'] # Z, Q, T, U and V in the exact order

# Virutal DATASET
def create_virtual_source(mode, data_dir, vars, vds_filename):
    # get parentidr from data dir
    vds_dir = os.path.join(data_dir, 'virtual_ds/evaluation')

    # INPUT SURFACE
    if mode=='surface_input':
        start = time.time()
        ex_ds_filename = os.path.join(data_dir, f"raw_data/surface/msl_2018.nc")
        with h5py.File(ex_ds_filename, 'r') as f:
            sh = f["msl"].shape
            print(f"Shape surface data: {sh}")
            layout = h5py.VirtualLayout(shape=(4,) + sh, dtype=np.float32)
        for i, var in enumerate(vars):
            # entry key 
            filename = os.path.join(data_dir, f"raw_data/surface/{var}_2018.nc")
            if var == 'u10m':
                entry_key = 'u10'
            elif var == 'v10m':
                entry_key = 'v10'
            else:
                entry_key = var
            vsource = h5py.VirtualSource(filename, entry_key, shape=sh)
            layout[i, :, :, :] = vsource
        # Add virtual dataset to output file
        vds_path = os.path.join(vds_dir, vds_filename)
        with h5py.File(vds_path, 'w', libver='latest') as f:
            f.create_virtual_dataset('data', layout, fillvalue=-1)
        # Check data
        with h5py.File(vds_path, "r") as f:
            print("Virtual dataset SURFACE GT")
            print(f"Shape: {f['data'].shape}")
            # get length of dataset
            num_samples = f["data"].shape[0]
        end = time.time()
        print(f"Time taken for setup virtual dataset SURFACE GT: {end-start} secs.\n")

    # INPUT ATMOSPHERIC
    elif mode=='upper_input':
        start = time.time()
        ex_ds_filename = os.path.join(data_dir, f"raw_data/atmospheric/z_2018.nc")
        with h5py.File(ex_ds_filename, 'r') as f:
            sh = f["z"].shape
            print(f"Shape atmospheric data: {sh}")
            layout = h5py.VirtualLayout(shape=(5,) + sh, dtype=np.float32)
        for i, var in enumerate(vars):
            # entry key 
            filename = os.path.join(data_dir, f"raw_data/surface/{var}_2018.nc")
            if var == 'u10m':
                entry_key = 'u10'
            elif var == 'v10m':
                entry_key = 'v10'
            else:
                entry_key = var
            vsource = h5py.VirtualSource(filename, entry_key, shape=sh)
            layout[i, :, :, :] = vsource
        # Add virtual dataset to output file
        vds_path = os.path.join(vds_dir, vds_filename)
        with h5py.File(vds_path, 'w', libver='latest') as f:
            f.create_virtual_dataset('data', layout, fillvalue=-1)
        # Check data
        with h5py.File(vds_path, "r") as f:
            print("Virtual dataset ATMPOSPHERIC GT")
            print(f"Shape: {f['data'].shape}")
            # get length of dataset
            num_samples = f["data"].shape[0]
        end = time.time()
        print(f"Time taken for setup virtual dataset ATMOSPHERIC GT: {end-start} secs.\n")
    
    # PREDICTIONS SURFACE
    elif mode=='surface_preds':
        start = time.time()
        ex_ds_filename = os.path.join(data_dir, f"preds/surface_PART1.h5")
        with h5py.File(ex_ds_filename, 'r') as f:
            sh = f["data"].shape
            print(f"Shape of loaded surface preds: {sh}")
            layout = h5py.VirtualLayout(shape=(1460,) + sh[1:], dtype=np.float32)
        for part in range(1,16):
            if part==15:
                start = (part-1)*100
                end = 1460
            else:
                start = (part-1)*100
                end = part*100
            # entry key 
            filename = os.path.join(data_dir, f"preds/surface_PART{part}.h5")
            entry_key = 'data'
            if part == 15:
                sh = (60,) + sh[1:]
            else:
                sh = (100,) + sh[1:]
            vsource = h5py.VirtualSource(filename, entry_key, shape=sh)
            layout[start:end, :, :, :] = vsource
        # Add virtual dataset to output file
        vds_path = os.path.join(vds_dir, vds_filename)
        with h5py.File(vds_path, 'w', libver='latest') as f:
            f.create_virtual_dataset('data', layout, fillvalue=-1)
        # Check data
        with h5py.File(vds_path, "r") as f:
            print("Virtual dataset SURFACE PREDS")
            print(f"Shape: {f['data'].shape}")
            # get length of dataset
            num_samples = f["data"].shape[0]
        end = time.time()
        print(f"Time taken for setup virtual dataset SURFACE PREDS: {end-start} secs.\n")

    # PREDICTIONS ATMOSPHERIC
    elif mode=='upper_preds':
        start = time.time()
        ex_ds_filename = os.path.join(data_dir, f"preds/upper_PART1.h5")
        with h5py.File(ex_ds_filename, 'r') as f:
            sh = f["data"].shape
            print(f"Shape of loaded surface preds: {sh}")
            layout = h5py.VirtualLayout(shape=(1460,) + sh[1:], dtype=np.float32)
        for part in range(1,16):
            if part==15:
                start = (part-1)*100
                end = 1460
            else:
                start = (part-1)*100
                end = part*100
            # entry key 
            filename = os.path.join(data_dir, f"preds/upper_PART{part}.h5")
            entry_key = 'data'
            if part == 15:
                sh = (60,) + sh[1:]
            else:
                sh = (100,) + sh[1:]
            vsource = h5py.VirtualSource(filename, entry_key, shape=sh)
            layout[start:end, :, :, :, :] = vsource
        # Add virtual dataset to output file
        vds_path = os.path.join(vds_dir, vds_filename)
        with h5py.File(vds_path, 'w', libver='latest') as f:
            f.create_virtual_dataset('data', layout, fillvalue=-1)
        # Check data
        with h5py.File(vds_path, "r") as f:
            print("Virtual dataset ATMOSPHERIC PREDS")
            print(f"Shape: {f['data'].shape}")
            # get length of dataset
            num_samples = f["data"].shape[0]
        end = time.time()
        print(f"Time taken for setup virtual dataset ATMOSPHERIC PREDS: {end-start} secs.\n")

    return vds_path

def evaluation(mode, every_2nd, save_dir):
    complete_start = time.time()

    # Data samples
    if every_2nd:
        eval_range = range(0, 1460, 2)
        #eval_range = range(0, 8, 2)
    else:
        eval_range = range(1460)
        #eval_range = range(3)

    # LOSS
    loss_fn = WeatherBenchLoss_plain()

    # Evaluate SURFACE
    if mode == 'surface':
        # VDS: PREDICTIONS AND GROUND TRUTH
        path_vds_gt_surface = create_virtual_source(mode='surface_input', data_dir=data_dir, vars=vars_surface, vds_filename=f"VDS_GT_surface.h5")
        path_vds_preds_surface = create_virtual_source(mode='surface_preds', data_dir=data_dir, vars=vars_surface, vds_filename=f"VDS_PREDS_surface.h5")
        print(f"*** EVALUATING SURFACE VARIABLES ***")
        # create np array of shape 4 x 1460
        loss_surface = np.zeros((4, 1))
        start_total = time.time()
        with h5py.File(path_vds_gt_surface, "r") as f_gt_surface:
            with h5py.File(path_vds_preds_surface, "r") as f_preds_surface:
                # do it for all variables together
                # we need to shift the ground truth by one, iterating from 0 (representing first) to 1458 (representing last)
                for i in eval_range:
                    print(f"*** ITERATION {i} ***")
                    # Loads preds and ground truth
                    start = time.time()
                    gt = f_gt_surface['data'][:, i, :, :]
                    preds = f_preds_surface['data'][i-1, :, :, :]
                    print(f"    Shape of gt: {gt.shape}")
                    print(f"    Shape of preds: {preds.shape}")
                    end = time.time()
                    print(f"    Time taken for loading one sample: {end-start} secs.")
                    # Evaluate
                    start = time.time()
                    loss = loss_fn.compute_loss(gt, preds)
                    print(f"    Shape of loss: {loss.shape}")
                    if i==0:
                        loss_surface = loss
                    else:
                        loss_surface = np.concatenate((loss_surface, loss), axis=-1)
                    print(f"    Loss surface shape: {loss_surface.shape}")
                    print(f"    Loss: {loss}")
                    end = time.time()
                    print(f"    Time taken for evaluating one datapoint: {end-start} secs.\n")
                # Average over all time steps
                final_loss_surface = np.mean(loss_surface, axis=-1)
                print(f"Final shape of loss: {final_loss_surface.shape}")
            end_total = time.time()
            print(f"Total time taken for evaluating all datapoints: {end_total-start_total} secs.")
            print(f"*** END OF EVALUATION SURFACE VARIABLES ***\n")
        
        # Save SURFACE LOSS
        np.save(os.path.join(save_dir, 'loss_surface.npy'), final_loss_surface)
        del loss_surface

    elif mode == 'atmosphere':
        # VDS: PREDICTIONS AND GROUND TRUTH
        path_vds_gt_upper = create_virtual_source(mode='upper_input', data_dir=data_dir, vars=vars_atmospheric, vds_filename=f"VDS_GT_atmospheric.h5")
        path_vds_preds_upper = create_virtual_source(mode='upper_preds', data_dir=data_dir, vars=vars_surface, vds_filename=f"VDS_PREDS_atmospheric.h5")
        # Evaluate SURFACE
        print(f"*** EVALUATING ATMOSPHERIC VARIABLES ***")
        # create np array of shape 5 x 13 x 1460
        loss_upper = np.zeros((5, 13, 1460))
        start_total = time.time()
        with h5py.File(path_vds_gt_upper, "r") as f_gt_upper:
            with h5py.File(path_vds_preds_upper, "r") as f_preds_upper:
                for i in eval_range:
                    print(f"*** ITERATION {i} ***")
                    # Loads preds and ground truth
                    start = time.time()
                    gt = f_gt_upper['data'][:, i, :, :, :]
                    preds = f_preds_upper['data'][i-1, :, :, :, :]
                    print(f"    Shape of gt: {gt.shape}")
                    print(f"    Shape of preds: {preds.shape}")
                    end = time.time()
                    print(f"    Time taken for loading one sample: {end-start} secs.")

                    # Evaluate
                    start = time.time()
                    loss = loss_fn.compute_loss(gt, preds)
                    print(f"    Shape of loss: {loss.shape}")
                    loss_upper[:, :, i] = loss
                    print(f"    Loss (first 4): {loss[:4, 0]}")
                    end = time.time()
                    print(f"    Time taken for evaluating one datapoint: {end-start} secs.")
                # Average over all time steps
                final_loss_upper = np.mean(loss_upper, axis=-1)
                print(f"Final shape of loss: {final_loss_upper.shape}")
            end_total = time.time()
            print(f"Total time taken for evaluating all datapoints: {end_total-start_total} secs.\n")
            print(f"*** END OF EVALUATION ATMOSPHERIC VARIABLES ***\n")

        # Save ATMOSPHERIC LOSS
        np.save(os.path.join(save_dir, 'loss_upper.npy'), final_loss_upper)
        del loss_upper

    complete_end = time.time()
    print(f"Complete evaluation time: {complete_end-complete_start} secs.")

if __name__ == "__main__":
    evaluation(mode, every_2nd, save_dir)
    print("Evaluation done.")
    