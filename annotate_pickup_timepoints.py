# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 20:54:38 2023

@author: locro
"""

import matplotlib.pylab as plt
import os
import pickle
import numpy as np
import platform
from scipy.spatial import distance
from sklearn.metrics import accuracy_score, precision_score, recall_score
from functools import reduce

if platform.system() == 'Windows':
    # We are running on Windows
    analysis_dir = '/Users/locro/Documents/Stanford/analysis/'
    checkpoint_dir = analysis_dir
elif platform.system() == 'Linux':
    # We are running on Linux
    analysis_dir = '/home/locross/analysis/'
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


def annotate_pickup_timepoints(train_or_val='val', pickup_or_move='move'):
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

    # for every goal find the timepoint where object was delivered to goal location
    pickup_timepoints = np.ones(y_val.shape)*-1
    num_trials = input_matrices.shape[0]
    num_goals = 3
    for i in range(num_trials):
        trial_x = input_matrices[i,:,:]
        for j in range(num_goals):
            if y_val[i,j]:
                # if a goal occurred, find when
                pos_inds = [data_columns.index('obj'+str(j)+'_'+dim) for dim in dims]
                trial_obj_pos = trial_x[:,pos_inds]
                for k in range(2):
                    pos_inds2 = [data_columns.index('agent'+str(k)+'_'+dim) for dim in dims]
                    trial_agent_pos = trial_x[:,pos_inds2]
                    if pickup_or_move == 'pickup':
                        pick_up_bool, pick_up_ind = detect_obj_pick_up_timepoint(trial_agent_pos, trial_obj_pos)
                    elif pickup_or_move == 'move':
                        # use first timepoint where object moved instead of less reliable pick up point
                        pick_up_bool, pick_up_ind = detect_object_move_timepoint(trial_obj_pos)
                    if pick_up_bool:
                        pickup_timepoints[i,j] = pick_up_ind
                        break
                    
    return pickup_timepoints


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) > stepsize)[0]+1)


def detect_obj_pick_up_timepoint(trial_agent_pos, trial_obj_pos):
    picked_up = False
    dropped = False
    # calculate how close agent is to object
    agent_obj_dist = np.array([distance.euclidean(trial_agent_pos[t,[0,2]], trial_obj_pos[t,[0,2]]) for t in range(len(trial_agent_pos))])
    # find inds where obj meet criteria
    # 1. obj is above y_thr
    # 2. obj is moving
    # 3. obj is close to agent
    # 4. largest y delta should be beginning or end of the sequence
    #y_thr = pick_up_y_thr[temp_obj_name]
    y_thr = 1e-3
    pick_up_inds = np.where((trial_obj_pos[:,1] > y_thr) & (trial_obj_pos[:,1] < 0.6))[0]
    #pick_up_event_inds = consecutive(pick_up_inds)
    obj_pos_delta = np.zeros(trial_obj_pos.shape)
    obj_pos_delta[1:,:] = np.abs(trial_obj_pos[1:,:] - trial_obj_pos[:-1,:])
    obj_pos_delta_sum = obj_pos_delta.sum(axis=1)
    obj_moving_inds = np.where(obj_pos_delta_sum > 1e-5)[0]
    agent_close_inds = np.where(agent_obj_dist < 0.8)[0]
    pick_up_move_close = reduce(np.intersect1d,(pick_up_inds, obj_moving_inds, agent_close_inds))
    pick_up_event_inds = consecutive(pick_up_move_close)
    for pick_up_event in pick_up_event_inds:
        if len(pick_up_event) > 5:
            # largest y delta should be beginning or end of the sequence
            obj_delta_event = obj_pos_delta[pick_up_event,:]
            amax_delta = np.argmax(obj_delta_event[:,1])
            if amax_delta == 0 or amax_delta == (len(pick_up_event)-1):
                picked_up = True
                dropped = True
                pick_up_ind = pick_up_event[0]
            
    if picked_up and dropped:
        return True, pick_up_ind
    else:
        return False, []
    
    
def detect_object_move_timepoint(trial_obj_pos, move_thr=0.1):
    obj_pos_delta = np.abs(trial_obj_pos[1:,:] - trial_obj_pos[:-1,:])
    obj_pos_delta_sum = obj_pos_delta.sum(axis=1)
    
    # initialize a counter and an index
    count = 0
    index = -1
    # loop through the array elements, ignoring first 5 steps where movement sometimes occurs
    for i in range(5, len(obj_pos_delta_sum)):
    	# check if the element is greater than 1e-5
    	if obj_pos_delta_sum[i] > (move_thr / trial_obj_pos.shape[0]):
    		# increment the counter
    		count += 1
    		# check if the counter is equal to 2
    		if count == 2:
    			# store the index of the first element of the consecutive values
    			index = i - 1
    		# check if the counter is greater than 2
    		elif count > 2:
    			# return the index of the first element of the consecutive values
    			break
    	else:
    		# reset the counter to zero
    		count = 0
            
    total_movement = obj_pos_delta_sum.sum()
    
    if total_movement > move_thr:
        return 1, index
    else:
        return 0, index