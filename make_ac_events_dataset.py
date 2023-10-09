# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 15:41:12 2023

@author: locro
"""

import argparse
import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd
import pickle
import platform
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from scipy.signal import find_peaks

from annotate_pickup_timepoints import annotate_pickup_timepoints
from annotate_goal_timepoints import annotate_goal_timepoints
from analysis_utils import load_trained_model, get_data_columns

from constants_lc import MODEL_DICT_VAL, DEFAULT_VALUES, DATASET_NUMS

def normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    return (arr - min_val) / (max_val - min_val)


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config_dir', type=str, action='store',
                        default=DEFAULT_VALUES['model_config_dir'],
                        help='Model config directory')
    parser.add_argument('--analysis_dir', type=str, action='store',
                        default=DEFAULT_VALUES['analysis_dir'], 
                        help='Analysis directory')
    parser.add_argument('--data_dir', type=str,
                         default=DEFAULT_VALUES['data_dir'], 
                         help='Data directory')
    parser.add_argument('--dataset', type=str,
                         default=DEFAULT_VALUES['dataset'], 
                         help='Dataset name')
    parser.add_argument('--checkpoint_dir', type=str, 
                        default=DEFAULT_VALUES['checkpoint_dir'], 
                        help='Checkpoint directory')
    parser.add_argument('--events_filename', type=str, 
                        default=None, 
                        help='Save filename')
    parser.add_argument('--model_key', type=str, required=True, help='Model to use for event labeling')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--batch_size', type=int, action='store',
                        default=DEFAULT_VALUES['batch_size'],
                        help='Batch size')
    return parser.parse_args()

if __name__ == '__main__':
    args = load_args()
    torch.manual_seed(DEFAULT_VALUES['eval_seed'])
    print("Using seed: {}".format(DEFAULT_VALUES['eval_seed']))
    model_info = MODEL_DICT_VAL[args.model_key]
    model_name = model_info['model_label']
    model = load_trained_model(model_info, args)
    model.eval()
    
    rollout_length = 30
    min_burnin = 5
    batch_size = args.batch_size

    # train test splits
    dataset_file = os.path.join(args.data_dir, args.dataset)
    loaded_dataset = pickle.load(open(dataset_file, 'rb'))
    train_dataset, test_dataset = loaded_dataset

    event_inds = {}
    for train_or_val in ['train', 'val']:
        print("Processing {} set".format(train_or_val))
        if train_or_val == 'train':
            input_data = train_dataset.dataset.tensors[0][train_dataset.indices,:,:]
        else:
            input_data = test_dataset.dataset.tensors[0][test_dataset.indices,:,:]
        num_timesteps = input_data.size(1)
        loader = DataLoader(input_data, batch_size=batch_size, shuffle=False, pin_memory=True)
        data_indices = list(range(len(input_data)))
        
        # load dataset info
        exp_info_file = dataset_file[:-4]+'_exp_info.pkl'
        if os.path.isfile(exp_info_file):
            with open(exp_info_file, 'rb') as f:
                exp_info_dict = pickle.load(f)
        else:
            print('DS info dict not found')
        
        wm_loss_by_time = np.zeros([input_data.size(0), input_data.size(1)])
        
        for batch_idx, batch_x in enumerate(loader):
            print('Batch',batch_idx)
            batch_x = batch_x.to(args.device)
            # Get the indices of the current batch
            current_indices = data_indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            for t in range(min_burnin, (num_timesteps-rollout_length)):
                if t%50 == 0:
                    print(t)
                burn_in_length = t
                x_hat = model.forward_rollout(batch_x, burn_in_length, rollout_length)
                x_supervise = batch_x[:,t:(t+rollout_length),:]
                wm_loss_temp = ((x_supervise - x_hat)**2).sum(dim=2)
                wm_loss = wm_loss_temp.mean(dim=1)
                wm_loss = wm_loss.detach().cpu().numpy()
                wm_loss_by_time[current_indices,t] = wm_loss
                
        # find peaks for each trial
        # Initialize a list to store the indices of the peaks for each trial
        peaks_indices = []

        for trial_num in range(input_data.size(0)):
            # Normalize the world model loss for the current trial
            normalized_loss = normalize(wm_loss_by_time[trial_num, :])
            
            # Find the peaks
            peaks, _ = find_peaks(normalized_loss, height=0.2, distance=10, prominence=0.2)
            
            # Append the indices of the peaks to our list
            peak_list = peaks.tolist()
            # always include last timestep as an event such that it is predicted if nothing else (or nothing sooner)
            last_timestep = input_data.size(1) - 1
            peak_list.append(last_timestep)
            peaks_indices.append(peak_list)

        event_inds[train_or_val] = peaks_indices
            
    if args.events_filename is None:
        events_filename = 'event_inds_'+args.model_key+'.pkl'
    else:
        events_filename = args.events_filename
    save_file = os.path.join(args.data_dir,'events_data','event_inds', events_filename)
    with open(save_file, 'wb') as f:
        pickle.dump(event_inds, f)