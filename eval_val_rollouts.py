# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:57:33 2023

@author: locro
"""

import argparse
import pickle
import platform
import torch
from torch.utils.data import DataLoader

from analysis_utils import load_trained_model
from models import DreamerV2, MultistepPredictor, MultistepDelta

if platform.system() == 'Windows':
    # We are running on Windows
    analysis_dir = '/Users/locro/Documents/Stanford/SocialWorldModeling/'
    data_dir = '/Users/locro/Documents/Stanford/analysis/data/'
    checkpoint_dir = analysis_dir
elif platform.system() == 'Linux':
    # We are running on Linux
    analysis_dir = '/home/locross/SocialWorldModeling'
    data_dir = '/home/locross/analysis/data/'
    checkpoint_dir = '/mnt/fs2/locross/analysis/'
    

if __name__ == "__main__":
    # train test splits
    data_file = data_dir+'train_test_splits_3D_dataset.pkl'
    with open(data_file, 'rb') as f:
        loaded_dataset = pickle.load(f)
    train_dataset, test_dataset = loaded_dataset
    
    # burn in first 50 frames, predict next 50 frames
    burn_in_length = 50
    rollout_length = 50
    
    batch_size = 64
    save_plot = False
    # include training data or just validation
    #eval_sets = ['Train', 'Validation']
    eval_sets = ['Validation']
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_dict= {
        'rssm_discrete': [DreamerV2, 'config', 'rssm_h1024_l2_mlp1024', 'rssm_h1024_l2_mlp1024', 'RSSM Discrete'],
        'multistep_pred': [MultistepPredictor, 'config', 'multistep_pred_h1024_l2_mlp1024', 'multistep_pred_h1024_l2_mlp1024_epoch8000', 'Multistep Predictor'],
        'multistep_delta': [MultistepDelta, 'config', 'multistep_delta_h1024_l2_mlp1024_l2', 'multistep_delta_h1024_l2_mlp1024_l2', 'Multistep Delta']
        }
    keys2analyze = ['rssm_discrete']
    results = []
    for key in keys2analyze:
        print(key)
        model_info = model_dict[key]
        model_name = model_info[4]
        model = load_trained_model(model_info, key, 300, DEVICE)
        
        for train_or_val in eval_sets:
            if train_or_val == 'Train':
                input_data = train_dataset.dataset.tensors[0][train_dataset.indices,:,:]
                # sub select 1000 to ease computation
                torch.manual_seed(100) # set random seed 
                indices = torch.randperm(input_data.size(0)) # get random indices 
                input_data = input_data[indices[:1000],:,:] # select 1000 rows
            elif train_or_val == 'Validation':
                input_data = test_dataset.dataset.tensors[0][test_dataset.indices,:,:]
                
            batch_trajs = []
            real_trajs = []
            imag_trajs = []
            loader = DataLoader(input_data, batch_size=batch_size, shuffle=True, pin_memory=True)
            for batch_x in loader:
                batch_x = batch_x.to(DEVICE)
                rollout_x = model.forward_rollout(batch_x, burn_in_length, rollout_length)
                
            
            
            
            for i in range(input_data.shape[0]):
                if i % 100 == 0:
                    print(i)
                end = burn_in_length + rollout_length #temp
                traj_sample = input_data[i,:end,:]
                batch_trajs.append(traj_sample)
                if i > 0 and i % batch_size == 0:
                    batch_x = torch.stack(batch_trajs, dim=0)
                    batch_x = torch.stack(batch_trajs, dim=0)
                    batch_x = batch_x.to(DEVICE)
                    if key[:9] == 'multistep':
                        x_hat_imag = model.forward_rollout(batch_x, burn_in_length, rollout_length)
                    elif key[:4] == 'rssm':
                        batch_x_burnin = batch_x[:,:burn_in_length,:]
                        output = model.encoder(batch_x_burnin)
                        prior = output[0]
                        x_hat_imag = get_imagined_states(prior, model, rollout_length)
                    x_hat_imag = x_hat_imag.to("cpu")
                    for j in range(x_hat_imag.size(0)):
                        traj = x_hat_imag[j, :, :]
                        imag_trajs.append(traj)
                    for real_traj in batch_trajs:
                        real_trajs.append(real_traj[burn_in_length:,:])
                    batch_trajs = []
                    torch.cuda.empty_cache()
            # compute last batch that is less than batch_size
            batch_x = torch.stack(batch_trajs, dim=0)
            batch_x = batch_x.to(DEVICE)
            if key[:9] == 'multistep':
                x_hat_imag = model.forward_rollout(batch_x, burn_in_length, rollout_length)
            elif key[:4] == 'rssm':
                batch_x_burnin = batch_x[:,:burn_in_length,:]
                output = model.encoder(batch_x_burnin)
                prior = output[0]
                x_hat_imag = get_imagined_states(prior, model, rollout_length)
            x_hat_imag = x_hat_imag.to("cpu")
            for j in range(x_hat_imag.size(0)):
                traj = x_hat_imag[j, :, :]
                imag_trajs.append(traj)
            for real_traj in batch_trajs:
                real_trajs.append(real_traj[burn_in_length:,:])
            torch.cuda.empty_cache()
                
            assert len(real_trajs) == len(imag_trajs)
            
            x_supervise = torch.stack(real_trajs)
            x_hat = torch.stack(imag_trajs).to('cpu')
            mse = ((x_supervise - x_hat)**2).mean().item()
            mse_by_time = ((x_supervise - x_hat)**2).mean(0)
            mse_by_time = mse_by_time.mean(1).detach().numpy()
            
            for i, disp in enumerate(mse_by_time):
                results_by_time.append({'Timepoint': i, 'MSE': disp, 'Model': model_name, 'Eval Set': train_or_val})
            
            results.append({
                        'Model': model_name,
                        'Task': 'Reconstruction error',
                        'Metric': 'MSE',
                        'Score': mse,
                        'Eval Set': train_or_val
                    })
            
            recon_features = eval_recon_features(x_supervise, x_hat, model_name)
            for item in recon_features.items():
                results.append({
                        'Model': model_name,
                        'Task': 'Reconstruct Features',
                        'Metric': item[0],
                        'Score': item[1],
                        'Eval Set': train_or_val
                    })