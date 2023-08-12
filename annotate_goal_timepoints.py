# -*- coding: utf-8 -*-
"""
Created on Fri May 12 17:13:33 2023

@author: locro
"""

import matplotlib.pylab as plt
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import accuracy_score, precision_score, recall_score
from analysis_utils import get_data_columns


def eval_recon_goals(input_matrices, recon_matrices, model_name='', final_location=False, plot=True, ds_num=2, obj_dist_thr=2.0, agent_dist_thr=1.0):
    if hasattr(recon_matrices, 'requires_grad') and recon_matrices.requires_grad:
        recon_matrices = recon_matrices.detach().numpy()
        
    scores = {}
    
    # relabel y based on validation input tensor
    num_trials = input_matrices.shape[0]
    num_goals = 3
    y_labels = -1*np.ones([num_trials, num_goals])
    
    data_columns = get_data_columns(ds_num)   
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
                min_timepoint = np.argmin(dist2target)
            if target_dist < obj_dist_thr:
                event_conditions.append(True)
                if agent_dist_thr is not None:
                    # is either agent close to the object when it is delivered?
                    agent_close = False
                    for k in range(2):
                        agent_pos_inds = [data_columns.index('agent'+str(k)+'_'+dim) for dim in dims]
                        trial_agent_pos = trial_x[:,agent_pos_inds]
                        agent_dist = distance.euclidean(trial_agent_pos[min_timepoint,:], trial_obj_pos[min_timepoint,:])
                        if agent_dist < agent_dist_thr:
                            agent_close = True
                            break
                    if agent_close:
                        event_conditions.append(True)
                    else:
                        event_conditions.append(False)
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
            # check for and replace NaNs
            trial_obj_pos = np.nan_to_num(trial_obj_pos, nan=0.0, posinf=0.0, neginf=0.0)
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
                min_timepoint = np.argmin(dist2target)
            if target_dist < obj_dist_thr:
                event_conditions.append(True)
                if agent_dist_thr is not None:
                    # is either agent close to the object when it is delivered?
                    agent_close = False
                    for k in range(2):
                        agent_pos_inds = [data_columns.index('agent'+str(k)+'_'+dim) for dim in dims]
                        trial_agent_pos = trial_x[:,agent_pos_inds]
                        agent_dist = distance.euclidean(trial_agent_pos[min_timepoint,:], trial_obj_pos[min_timepoint,:])
                        if agent_dist < agent_dist_thr:
                            agent_close = True
                            break
                    if agent_close:
                        event_conditions.append(True)
                    else:
                        event_conditions.append(False)
            else:
                event_conditions.append(False)
            if all(event_conditions):
                y_recon[i,j] = 1
                #print('goal '+str(j)+' is a success')
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


def annotate_goal_timepoints(loaded_dataset, train_or_val='val', ds_num=2):
    # load train and val dataset
    train_dataset, test_dataset = loaded_dataset
    
    if train_or_val == 'train':
        input_matrices = train_dataset.dataset.tensors[0][train_dataset.indices,:,:].numpy()
    elif train_or_val == 'val':
        input_matrices = test_dataset.dataset.tensors[0][test_dataset.indices,:,:].numpy()
    
    scores, y_val, y_recon = eval_recon_goals(input_matrices, input_matrices, ds_num=ds_num)
    
    data_columns = get_data_columns(ds_num)    
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