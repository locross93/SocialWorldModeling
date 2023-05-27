# -*- coding: utf-8 -*-
"""
Created on Thu May 25 17:53:57 2023

@author: locro
"""

import argparse
import numpy as np
import os
import pandas as pd
import pickle
import random
import torch
from torch.utils.data import TensorDataset, Subset

from constants_lc import DEFAULT_VALUES

def get_trialtype_inds(names_dict):
    # loop through the list of strings
    for i, name in enumerate(names_dict['exp_names']):
        # get the first part of the name before the underscore
        prefix = name.split("_")[0]+'_'+name.split("_")[1]
        # if the prefix is already in the dictionary, append the index to the list
        if prefix in names_dict:
            names_dict[prefix].append(i)
        # otherwise, create a new list with the index
        else:
            names_dict[prefix] = [i]
    
    return names_dict

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tdw_data_dir', type=str,
                         default='/mnt/fs2/locross/AttentionAgent/img_out/', 
                         help='TDW Data directory')
    parser.add_argument('--data_dir', type=str,
                         default=DEFAULT_VALUES['data_dir'], 
                         help='Save data directory')
    parser.add_argument('--dataset_name', type=str,
                         default='dataset_5_25_23.pkl', 
                         help='Dataset name')
    
    return parser.parse_args()


if __name__ == '__main__': 
    args = load_args()
    dir_folders = os.listdir(args.tdw_data_dir)
    down_sam = 5
    all_trial_data = []
    goal_labels = []
    all_exp_names = []
    event_logs = []
    
    for exp_name in dir_folders:
        trial_obs_file = args.tdw_data_dir+exp_name+'/trial_obs.csv'
        if os.path.exists(trial_obs_file):
            if os.stat(trial_obs_file).st_size == 0:
                # if an empty file
                continue
            else:
                trial_data = pd.read_csv(trial_obs_file)
        else:
            continue
        
        if trial_data.shape == (1500, 49):
            cols2remove = ['Unnamed: 0', 'obj0_name', 'obj1_name', 'obj2_name', 
                       'agent0_name', 'agent1_name', 'agent2_name', 'agent2_x', 'agent2_y', 'agent2_z',
                       'agent2_rot_x', 'agent2_rot_y', 'agent2_rot_z', 'agent2_rot_w']
        else:
            continue
        
        trial_data = trial_data.drop(cols2remove, axis=1)
        trial_array = trial_data.to_numpy()
        
        assert trial_array.shape[1] == 35
        
        #check for outlier values and skip if found
        max_trial_vals = np.max(np.abs(trial_array), axis=0)
        if len(np.where(max_trial_vals > 7)[0]) > 0:
            print('Outlier values found in',exp_name)
            continue
        
        event_log_file = args.tdw_data_dir+exp_name+'/event_log.csv'
        if os.path.exists(event_log_file):
            if os.stat(event_log_file).st_size == 0:
                event_logs.append('No events')
            else:
                event_log = pd.read_csv(event_log_file)
                event_logs.append(event_log)
        else:
            event_logs.append(False)
        
        # downsample and flatten
        trial_array = trial_array[::down_sam , :]
        #trial_array = trial_array.reshape(1, -1)
        all_trial_data.append(trial_array)
        #goal_labels.append(goal_flags)
        all_exp_names.append(exp_name)
        print(exp_name)
        
    input_tensor = np.stack(all_trial_data, axis=0)
    dataset = TensorDataset(torch.tensor(input_tensor).float())
    
    # make training and test splits
    # only take trials with logs for validation data
    logged_indices = [i for i, x in enumerate(event_logs) if x is not False]
    validation_split = 0.2 # the fraction of data for testing
    num_val_trials = np.array(input_tensor.shape[0] * validation_split).astype(int)
    val_idx = random.choices(logged_indices, k=num_val_trials)
    train_idx = [i for i in range(len(dataset)) if i not in val_idx]
    random.shuffle(train_idx)
    train_dataset = Subset(dataset, train_idx) # a subset of the dataset with train indices
    test_dataset = Subset(dataset, val_idx) # a subset of the dataset with test indices
    
    save_file = os.path.join(args.data_dir, 'dataset_5_25_23.pkl')
    full_dataset = [train_dataset, test_dataset]
    with open(save_file, 'wb') as handle:
        pickle.dump(full_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    exp_info_dict = {}
    exp_info_dict['train'] = {}
    exp_info_dict['val'] = {}
    exp_info_dict['train']['indices'] = train_idx
    exp_info_dict['val']['indices'] = val_idx
    exp_info_dict['train']['exp_names'] = [all_exp_names[i] for i in train_idx]
    exp_info_dict['val']['exp_names'] = [all_exp_names[i] for i in val_idx]
    exp_info_dict['train'] = get_trialtype_inds(exp_info_dict['train'])
    exp_info_dict['val'] = get_trialtype_inds(exp_info_dict['val'])
    exp_info_dict['train']['event_logs'] = [event_logs[i] for i in train_idx]
    exp_info_dict['val']['event_logs'] = [event_logs[i] for i in val_idx]
    
    exp_info_file = os.path.join(args.data_dir, 'dataset_5_25_23_exp_info.pkl')
    with open(exp_info_file, 'wb') as handle:
        pickle.dump(exp_info_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

