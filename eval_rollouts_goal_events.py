# -*- coding: utf-8 -*-
"""
Created on Thu May 11 13:43:20 2023

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
from models import DreamerV2, MultistepPredictor, MultistepDelta
from annotate_pickup_timepoints import annotate_pickup_timepoints

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
    
    
def eval_goal_events_in_rollouts(model, input_data, ds='First'):
    if ds == 'First':
        # first dataset
        pickup_timepoints = annotate_pickup_timepoints(train_or_val, pickup_or_move='move')
        single_goal_trajs = np.where((np.sum(pickup_timepoints > -1, axis=1) == 1))[0]
    else:
        # use event log for new datasets - TO DO
        pickup_timepoints = []
        
    num_single_goal_trajs = len(single_goal_trajs)
    imagined_trajs = np.zeros([num_single_goal_trajs, input_data.shape[1], input_data.shape[2]])
    real_trajs = []
    imag_trajs = []
    for i,row in enumerate(single_goal_trajs):
        if i%50 == 0:
            print(i)
        steps2pickup = np.max(pickup_timepoints[row,:]).astype(int)
        rollout_length = num_timepoints - steps2pickup
        x = input_data[row,:,:].unsqueeze(0)
        rollout_x = model.forward_rollout(x, steps2pickup, rollout_length).cpu().detach()
        real_traj = x[:,steps2pickup:,:].to("cpu")
        assert rollout_x.size() == real_traj.size()
        real_trajs.append(real_traj)
        imag_trajs.append(rollout_x)
        imagined_trajs[i,steps2pickup:,:] = rollout_x
    x_true = torch.cat(real_trajs, dim=1)
    x_hat = torch.cat(imag_trajs, dim=1)
    mse = ((x_true - x_hat)**2).mean().item()
    
    full_trajs = input_data[single_goal_trajs,:,:].cpu()
    scores, y_labels, y_recon = eval_recon_goals(full_trajs, imagined_trajs, final_location=True, plot=False)
    # evaluate whether only appropriate goals (after object picked up) are reconstructed
    pickup_subset = pickup_timepoints[single_goal_trajs,:]
    indices = np.argwhere(pickup_subset > -1)
    accuracy = np.mean(y_recon[indices[:,0],indices[:,1]])
    
    return accuracy, mse
    

if __name__ == "__main__":
    # train test splits
    data_file = data_dir+'train_test_splits_3D_dataset.pkl'
    with open(data_file, 'rb') as f:
        loaded_dataset = pickle.load(f)
    train_dataset, test_dataset = loaded_dataset
    
    save_plot = True
    save_file = 'eval_rollouts_goal_events'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_or_val = 'val'
    
    if train_or_val == 'train':
        input_data = train_dataset.dataset.tensors[0][train_dataset.indices,:,:]
    elif train_or_val == 'val':
        input_data = test_dataset.dataset.tensors[0][test_dataset.indices,:,:]
    num_timepoints = input_data.size(1)
    
    model_dict= {
        'rssm_discrete': [DreamerV2, 'config', 'rssm_h1024_l2_mlp1024', 'rssm_h1024_l2_mlp1024', 'RSSM Discrete'],
        'multistep_predictor': [MultistepPredictor, 'config', 'mp_input_embed_h1024_l2_mlp1024_l2', 'mp_input_embed_h1024_l2_mlp1024_l2', 'Multistep Predictor'],
        'multistep_delta': [MultistepDelta, 'config', 'multistep_delta_h1024_l2_mlp1024_l2', 'multistep_delta_h1024_l2_mlp1024_l2', 'Multistep Delta']
        }
    keys2analyze = ['rssm_discrete', 'multistep_predictor']
    #keys2analyze = ['rssm_discrete']
    results = []
    for key in keys2analyze:
        print(key)
        model_info = model_dict[key]
        model_name = model_info[4]
        model = load_trained_model(model_info, key, num_timepoints, DEVICE)
        
        # put on gpu
        input_data = input_data.to(DEVICE)
        
        accuracy, mse = eval_goal_events_in_rollouts(model, input_data)
        
        results.append({
                    'Model': model_name,
                    'Score': accuracy,
                    'MSE': mse
                })
        
    df_plot = pd.DataFrame(results)
    
    # Plot the results using seaborn
    fig = plt.figure()
    sns.barplot(x='Model', y='Score', data=df_plot) 
    plt.title('Reconstruct Goal Events From Imagination', fontsize=18)
    plt.xlabel('Model Name', fontsize=16) 
    plt.ylabel('Accuracy', fontsize=16)
    plt.ylim([0, 1])
    if save_plot:
        df_plot.to_csv(analysis_dir+'results/figures/'+save_file+'.csv')
        plt.savefig(analysis_dir+'results/figures/'+save_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    fig = sns.barplot(x='Model', y='MSE', data=df_plot) 
    plt.title('Reconstruct Error of Goal Events in Imagination', fontsize=18)
    plt.xlabel('Model Name', fontsize=16) 
    plt.ylabel('MSE', fontsize=16)
    if save_plot:
        plt.savefig(analysis_dir+'results/figures/'+save_file+'_recon_error', dpi=300, bbox_inches='tight')
    plt.show()
            
        
        