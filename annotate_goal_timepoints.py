# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 17:57:38 2023

@author: locro
"""

import os
import pickle
import numpy as np
import platform
from scipy.spatial import distance

if platform.system() == 'Windows':
    # We are running on Windows
    analysis_dir = '/Users/locro/Documents/Stanford/analysis/'
    checkpoint_dir = analysis_dir
elif platform.system() == 'Linux':
    # We are running on Linux
    analysis_dir = '/home/locross/analysis/'
    checkpoint_dir = '/mnt/fs2/locross/analysis/'
    
os.chdir(analysis_dir)
from eval_goal_events import eval_recon_goals

def annotate_goal_timepoints(train_or_val='val'):
    # load data
    data_file = analysis_dir+'data/train_test_splits_3D_dataset.pkl'
    with open(data_file, 'rb') as f:
        loaded_dataset = pickle.load(f)
    train_dataset, test_dataset = loaded_dataset
    
    if train_or_val == 'train':
        input_data = train_dataset.dataset.tensors[0][train_dataset.indices,:,:].numpy()
    elif train_or_val == 'val':
        input_data = test_dataset.dataset.tensors[0][test_dataset.indices,:,:].numpy()
    
    input_matrices = input_data.reshape(-1, 300, 23)
    scores, y_val, y_recon = eval_recon_goals(input_matrices, input_matrices)
    
    data_columns = ['obj0_x', 'obj0_y', 'obj0_z', 'obj1_x', 'obj1_y', 'obj1_z', 'obj2_x',
           'obj2_y', 'obj2_z', 'agent0_x', 'agent0_y', 'agent0_z', 'agent0_rot_x',
           'agent0_rot_y', 'agent0_rot_z', 'agent0_rot_w', 'agent1_x', 'agent1_y',
           'agent1_z', 'agent1_rot_x', 'agent1_rot_y', 'agent1_rot_z',
           'agent1_rot_w']    
    dims = ['x', 'y', 'z']
    fp1_pos = np.array([0.0, 0.5, -6.0])
    
    # for every goal find the timepoint where object was delivered to goal location
    goal_timepoints = np.ones(y_val.shape)*-1
    num_trials = input_matrices.shape[0]
    num_timepoints = input_matrices.shape[1]
    num_goals = 3
    for i in range(num_trials):
        trial_x = input_matrices[i,:,:]
        for j in range(num_goals):
            if y_val[i,j]:
                # if a goal occurred, find when
                pos_inds = [data_columns.index('obj'+str(j)+'_'+dim) for dim in dims]
                trial_obj_pos = trial_x[:,pos_inds]
                target_dist = distance.euclidean(fp1_pos, trial_obj_pos[-1,:])
                assert target_dist < 2.0
                for t in range(num_timepoints):
                    if distance.euclidean(trial_obj_pos[t,:], trial_obj_pos[-1,:]) < 0.25:
                        goal_timepoints[i,j] = t
                        break
                    
    return goal_timepoints