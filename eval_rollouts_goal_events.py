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
from annotate_pickup_timepoints import annotate_pickup_timepoints, eval_recon_goals

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
            
        
        