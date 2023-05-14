# -*- coding: utf-8 -*-
"""
Created on Fri May 12 17:13:33 2023

@author: locro
"""

import matplotlib.pylab as plt
import os
import pickle
import numpy as np
import platform
from scipy.spatial import distance
from sklearn.metrics import accuracy_score, precision_score, recall_score

if platform.system() == 'Windows':
    # We are running on Windows
    analysis_dir = '/Users/locro/Documents/Stanford/SocialWorldModeling/'
    data_dir = '/Users/locro/Documents/Stanford/analysis/data/'
    checkpoint_dir = analysis_dir
elif platform.system() == 'Linux':
    # We are running on Linux
    analysis_dir = '/home/locross/SocialWorldModeling/'
    data_dir = '/home/locross/analysis/data/'
    checkpoint_dir = '/mnt/fs2/locross/analysis/'
    
os.chdir(analysis_dir)

def eval_recon_goals(input_matrices, recon_matrices, model_name='', final_location=True, plot=True):
    if input_matrices.shape[1:] != (300, 23):
        input_matrices = input_matrices.reshape(-1, 300, 23)
        
    if recon_matrices.shape[1:] != (300, 23):
        recon_matrices = recon_matrices.reshape(-1, 300, 23)
        
    if hasattr(recon_matrices, 'requires_grad') and recon_matrices.requires_grad:
        recon_matrices = recon_matrices.detach().numpy()
        
    scores = {}
    
    # relabel y based on validation input tensor
    num_trials = input_matrices.shape[0]
    num_goals = 3
    y_labels = -1*np.ones([num_trials, num_goals])
    
    data_columns = ['obj0_x', 'obj0_y', 'obj0_z', 'obj1_x', 'obj1_y', 'obj1_z', 'obj2_x',
           'obj2_y', 'obj2_z', 'agent0_x', 'agent0_y', 'agent0_z', 'agent0_rot_x',
           'agent0_rot_y', 'agent0_rot_z', 'agent0_rot_w', 'agent1_x', 'agent1_y',
           'agent1_z', 'agent1_rot_x', 'agent1_rot_y', 'agent1_rot_z',
           'agent1_rot_w']    
    dims = ['x', 'y', 'z']
    fp1_pos = np.array([0.0, 0.5, -6.0])
    
    for i in range(num_trials):
        trial_x = input_matrices[i,:,:]
        for j in range(num_goals):
            event_conditions = []
            pos_inds = [data_columns.index('obj'+str(j)+'_'+dim) for dim in dims]
            trial_obj_pos = trial_x[:,pos_inds]
            # check that final location is different than initial location
            start_end_dist = distance.euclidean(trial_obj_pos[0,:], trial_obj_pos[-1,:])
            if start_end_dist > 1.0:
                event_conditions.append(True)
            else:
                event_conditions.append(False)
            if final_location:
                # check that final location is close to target
                target_dist = distance.euclidean(fp1_pos, trial_obj_pos[-1,:])
            else:
                # check if any location at any step is close to target
                dist2target = np.array([distance.euclidean(fp1_pos, step_pos) for step_pos in trial_obj_pos])
                target_dist = np.min(dist2target)
            if target_dist < 2.0:
                event_conditions.append(True)
            else:
                event_conditions.append(False)
            if all(event_conditions):
                y_labels[i,j] = 1
            else:
                y_labels[i,j] = 0
                
    # evaluate whether the same events are labeled correctly in the reconstructed data 
    y_recon = -1*np.ones([num_trials, num_goals])
    
    for i in range(num_trials):
        trial_x = recon_matrices[i,:,:]
        for j in range(num_goals):
            event_conditions = []
            pos_inds = [data_columns.index('obj'+str(j)+'_'+dim) for dim in dims]
            trial_obj_pos = trial_x[:,pos_inds]
            # check that final location is different than initial location
            start_end_dist = distance.euclidean(trial_obj_pos[0,:], trial_obj_pos[-1,:])
            if start_end_dist > 1.0:
                event_conditions.append(True)
            else:
                event_conditions.append(False)
            if final_location:
                # check that final location is close to target
                target_dist = distance.euclidean(fp1_pos, trial_obj_pos[-1,:])
            else:
                # check if any location at any step is close to target
                dist2target = np.array([distance.euclidean(fp1_pos, step_pos) for step_pos in trial_obj_pos])
                target_dist = np.min(dist2target)
            if target_dist < 2.0:
                event_conditions.append(True)
            else:
                event_conditions.append(False)
            if all(event_conditions):
                y_recon[i,j] = 1
            else:
                y_recon[i,j] = 0
                
    scores['Accuracy'] = accuracy_score(y_labels.reshape(-1), y_recon.reshape(-1))
    scores['Precision'] = precision_score(y_labels.reshape(-1), y_recon.reshape(-1))
    scores['Recall'] = recall_score(y_labels.reshape(-1), y_recon.reshape(-1))
    
    if plot:
        # Get the keys and values from the dictionary
        keys = list(scores.keys())
        values = list(scores.values())
    
        # Create a bar plot using the keys and values
        plt.bar(keys, values)
        # Label the axes and show the plot
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title('Reconstructing Goal Events '+model_name)
        plt.show()
    
    return scores, y_labels, y_recon

def annotate_goal_timepoints(train_or_val='val'):
    # load data
    data_file = data_dir+'train_test_splits_3D_dataset.pkl'
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