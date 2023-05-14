# -*- coding: utf-8 -*-
"""
Created on Fri May 12 16:11:36 2023

@author: locro
"""

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pickle
import platform
from scipy.spatial import distance
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch

from analysis_utils import load_trained_model
from models import DreamerV2, MultistepPredictor, MultistepDelta, TransformerMSPredictor
from annotate_pickup_timepoints import annotate_pickup_timepoints
from annotate_goal_timepoints import annotate_goal_timepoints

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
    
def detect_object_move(trial_obj_pos, move_thr):
    obj_pos_delta = np.abs(trial_obj_pos[1:,:] - trial_obj_pos[:-1,:])
    obj_pos_delta_sum = obj_pos_delta.sum(axis=1)
    total_movement = obj_pos_delta_sum.sum()
    
    if total_movement > move_thr:
        return 1
    else:
        return 0
    
    
def eval_move_events(input_matrices, recon_matrices, move_thr=0.1, plot=True, model_name=''):
    if input_matrices.shape[1:] != (300, 23):
        input_matrices = input_matrices.reshape(-1, 300, 23)
        
    if recon_matrices.shape[1:] != (300, 23):
        recon_matrices = recon_matrices.reshape(-1, 300, 23)
        
    if hasattr(recon_matrices, 'requires_grad') and recon_matrices.requires_grad:
        recon_matrices = recon_matrices.detach().numpy()
        
    scores = {}
    
    data_columns = ['obj0_x', 'obj0_y', 'obj0_z', 'obj1_x', 'obj1_y', 'obj1_z', 'obj2_x',
           'obj2_y', 'obj2_z', 'agent0_x', 'agent0_y', 'agent0_z', 'agent0_rot_x',
           'agent0_rot_y', 'agent0_rot_z', 'agent0_rot_w', 'agent1_x', 'agent1_y',
           'agent1_z', 'agent1_rot_x', 'agent1_rot_y', 'agent1_rot_z',
           'agent1_rot_w'] 
    dims = ['x', 'y', 'z']

    num_trials = input_matrices.shape[0]
    num_objs = 3
    obj_moved_flag = np.zeros([num_trials, 3])
    move_thr_true = 0.1
    for i in range(num_trials):
        trial_x = input_matrices[i,:,:]
        for j in range(num_objs):
            pos_inds = [data_columns.index('obj'+str(j)+'_'+dim) for dim in dims]
            trial_obj_pos = trial_x[:,pos_inds]
            obj_moved_flag[i,j] = detect_object_move(trial_obj_pos, move_thr_true)
            
    recon_moved_flag = np.zeros([num_trials, 3])
    for i in range(num_trials):
        trial_x = recon_matrices[i,:,:]
        for j in range(num_objs):
            pos_inds = [data_columns.index('obj'+str(j)+'_'+dim) for dim in dims]
            trial_obj_pos = trial_x[:,pos_inds]
            recon_moved_flag[i,j] = detect_object_move(trial_obj_pos, move_thr)
            
    scores['Accuracy'] = accuracy_score(obj_moved_flag.reshape(-1), recon_moved_flag.reshape(-1))
    scores['Precision'] = precision_score(obj_moved_flag.reshape(-1), recon_moved_flag.reshape(-1))
    scores['Recall'] = recall_score(obj_moved_flag.reshape(-1), recon_moved_flag.reshape(-1))
    
    # Get the keys and values from the dictionary
    if plot:
        keys = list(scores.keys())
        values = list(scores.values())
    
        # Create a bar plot using the keys and values
        plt.bar(keys, values)
        # Label the axes and show the plot
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title('Reconstructing Move Events '+model_name)
        plt.ylim([0, 1])
        plt.show()
    
    return scores, obj_moved_flag, recon_moved_flag

def eval_move_events_in_rollouts(model, input_data, ds='First'):
    if ds == 'First':
        # first dataset
        pickup_timepoints = annotate_pickup_timepoints(train_or_val, pickup_or_move='move')
        goal_timepoints = annotate_goal_timepoints(train_or_val)
        single_goal_trajs = np.where((np.sum(pickup_timepoints > -1, axis=1) == 1))[0]
        multi_goal_trajs = np.where((np.sum(pickup_timepoints > -1, axis=1) == 3))[0]
    else:
        # use event log for new datasets - TO DO
        pickup_timepoints = []
        
    imagined_trajs = np.zeros(input_data.shape)
    for i in range(input_data.shape[0]):
        if i%50 == 0:
            print(i)
        x = input_data[i,:,:].unsqueeze(0)
        if i in single_goal_trajs:
            # burn in to a few frames past the goal, so it is clear it is a single goal trial - TO DO THIS WILL BE DIFF FOR DS2
            # get the only goal point in the trajectory
            burn_in_length = np.max(goal_timepoints[i,:]).astype(int) + 10
        elif i in multi_goal_trajs:
            # burn in to the pick up point of the 2nd object that is picked up, so its unambiguous that all objects will be delivered
            burn_in_length = np.sort(pickup_timepoints[i,:])[1].astype(int)
        else:
            burn_in_length = non_goal_burn_in
        # store the steps of burn in with real frames in imagined_trajs
        imagined_trajs[i,:burn_in_length,:] = x[:,:burn_in_length,:].cpu()
        # rollout model for the rest of the trajectory
        rollout_length = num_timepoints - burn_in_length
        rollout_x = model.forward_rollout(x, burn_in_length, rollout_length).cpu().detach()
        # store the steps after pick up with predicted frames in imagined_trajs
        imagined_trajs[i,burn_in_length:,:] = rollout_x
        
    input_data = input_data.to('cpu')
    scores, obj_moved_flag, recon_moved_flag = eval_move_events(input_data, imagined_trajs, move_thr, plot=True, model_name=model_name)
        
    return scores
    
    
if __name__ == "__main__":
    # train test splits
    data_file = data_dir+'train_test_splits_3D_dataset.pkl'
    with open(data_file, 'rb') as f:
        loaded_dataset = pickle.load(f)
    train_dataset, test_dataset = loaded_dataset
    
    save_plot = False
    save_file = 'eval_rollouts_move_events'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_or_val = 'val'
    # for non goal trials, how many steps to show before rollout - should large enough to disambiguate behavior
    non_goal_burn_in = 50
    move_thr = 4.0
    
    if train_or_val == 'train':
        input_data = train_dataset.dataset.tensors[0][train_dataset.indices,:,:]
    elif train_or_val == 'val':
        input_data = test_dataset.dataset.tensors[0][test_dataset.indices,:,:]
    num_timepoints = input_data.size(1)
    
    model_dict = {
        'rssm': {'class': DreamerV2, 'config': 'rssm_disc_default_config.json', 
                      'model_dir': 'rssm_h1024_l2_mlp1024', 'model_label': 'RSSM Discrete'},
        'transformer': {'class': TransformerMSPredictor, 'config': 'transformer_default_config.json', 
                      'model_dir': 'transformer_mp', 'epoch': '500', 'model_label': 'Transformer MP'},
        }
    #keys2analyze = ['rssm_discrete', 'multistep_predictor']
    keys2analyze = ['transformer']
    results = []
    for key in keys2analyze:
        print(key)
        model_info = model_dict[key]
        model_name = model_info['model_label']
        model = load_trained_model(model_info, DEVICE)
        
        # put on gpu
        input_data = input_data.to(DEVICE)
        
        scores = eval_move_events_in_rollouts(model, input_data)
        
        results.append({
                    'Model': model_name,
                    'Metric': 'Accuracy',
                    'Score': scores['Accuracy']
                })
        results.append({
                    'Model': model_name,
                    'Metric': 'Precision',
                    'Score': scores['Precision']
                })
        results.append({
                    'Model': model_name,
                    'Metric': 'Recall',
                    'Score': scores['Recall']
                })
        
    df_plot = pd.DataFrame(results)
    
    # Plot the results using seaborn
    fig = plt.figure()
    sns.barplot(x='Metric', y='Score', hue='Model', data=df_plot) 
    plt.title('Evaluate Move Events From Forward Rollouts', fontsize=14)
    plt.xlabel('Metric', fontsize=16) 
    plt.ylabel('Score', fontsize=16)
    plt.ylim([0, 1])
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    if save_plot:
        df_plot.to_csv(analysis_dir+'results/figures/'+save_file+'.csv')
        plt.savefig(analysis_dir+'results/figures/'+save_file, dpi=300, bbox_inches='tight')
    plt.show()