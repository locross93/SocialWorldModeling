# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:57:33 2023

@author: locro
"""

import argparse
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pickle
import platform
import seaborn as sns
from scipy.stats import pearsonr
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
    analysis_dir = '/home/locross/SocialWorldModeling/'
    data_dir = '/home/locross/analysis/data/'
    checkpoint_dir = '/mnt/fs2/locross/analysis/'
    
    
def eval_recon_features(input_matrices, recon_matrices, model_name='', remove_outliers=True):
    assert len(input_matrices.shape) == 3
    assert len(recon_matrices.shape) == 3
        
    if hasattr(recon_matrices, 'requires_grad') and recon_matrices.requires_grad:
        recon_matrices = recon_matrices.detach().numpy()
    
    scores = {}
        
    data_columns = ['obj0_x', 'obj0_y', 'obj0_z', 'obj1_x', 'obj1_y', 'obj1_z', 'obj2_x',
           'obj2_y', 'obj2_z', 'agent0_x', 'agent0_y', 'agent0_z', 'agent0_rot_x',
           'agent0_rot_y', 'agent0_rot_z', 'agent0_rot_w', 'agent1_x', 'agent1_y',
           'agent1_z', 'agent1_rot_x', 'agent1_rot_y', 'agent1_rot_z',
           'agent1_rot_w']
    
    # 3 x 3 subplot for obj by x, y, z
    # put correlation in corner of each plot
    dims = ['x', 'y', 'z']
    fig, axes =  plt.subplots(3,3)
    # Adjust the spacing between the subplots and the margins of the figure
    fig.subplots_adjust(top=0.85, hspace=1.0, wspace=0.5)
    for row in range(3):
        for col,dim in enumerate(dims):
            var_ind = (row*3)+col
            var_name = data_columns[var_ind]
            x_true = input_matrices[:,:,var_ind].reshape(-1)
            x_pred = recon_matrices[:,:,var_ind].reshape(-1)
            # remove extreme values
            if remove_outliers:
                x_pred_abs = np.abs(x_pred)
                normal_inds = np.where(x_pred_abs < 15)[0]
                corr = round(pearsonr(x_true[normal_inds], x_pred[normal_inds])[0], 3)
            else:
                corr = round(pearsonr(x_true, x_pred)[0], 3)
            scores[var_name] = corr
            axes[row][col].scatter(x_true[normal_inds], x_pred[normal_inds])
            axes[row][col].set_title(var_name)
            axes[row][col].text(0.25, 0.9, 'r='+str(corr), horizontalalignment='center',
                                verticalalignment='center', transform=axes[row][col].transAxes)
    fig.supxlabel('True Values')
    fig.supylabel('Reconstructed Values')
    fig.suptitle('Reconstructing Object Features '+model_name)
    plt.show()
    
    fig, axes =  plt.subplots(2,3)
    fig.subplots_adjust(top=0.85, hspace=0.4, wspace=0.4)
    for row in range(2):
        for col,dim in enumerate(dims):
            var_ind = 9 + (row*7) + col
            var_name = data_columns[var_ind]
            x_true = input_matrices[:,:,var_ind].reshape(-1)
            x_pred = recon_matrices[:,:,var_ind].reshape(-1)
            # remove extreme values
            if remove_outliers:
                x_pred_abs = np.abs(x_pred)
                normal_inds = np.where(x_pred_abs < 15)[0]
                corr = round(pearsonr(x_true[normal_inds], x_pred[normal_inds])[0], 3)
            else:
                corr = round(pearsonr(x_true, x_pred)[0], 3)
            scores[var_name] = corr
            axes[row][col].scatter(x_true[normal_inds], x_pred[normal_inds])
            axes[row][col].set_title(var_name)
            axes[row][col].text(0.25, 0.9, 'r='+str(corr), horizontalalignment='center',
                                verticalalignment='center', transform=axes[row][col].transAxes)
    fig.supxlabel('True Values')
    fig.supylabel('Reconstructed Values')
    fig.suptitle('Reconstructing Agent Features '+model_name)
    plt.show()
    
    return scores


def eval_forward_rollout_mse(model, input_data, batch_size, burn_in_length, rollout_length):
    real_trajs = []
    imag_trajs = []
    loader = DataLoader(input_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    for batch_x in loader:
        batch_x = batch_x.to(DEVICE)
        rollout_x = model.forward_rollout(batch_x, burn_in_length, rollout_length)
        rollout_x = rollout_x.to("cpu")
        real_traj = batch_x[:,burn_in_length:(burn_in_length+rollout_length),:].to("cpu")
        assert rollout_x.size() == real_traj.size()
        imag_trajs.append(rollout_x)
        real_trajs.append(real_traj)
        
    x_supervise = torch.cat(real_trajs, dim=0)
    x_hat = torch.cat(imag_trajs, dim=0)
    mse = ((x_supervise - x_hat)**2).mean().item()
    
    recon_features = eval_recon_features(x_supervise, x_hat, model_name)
        
    return mse, recon_features
    

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
    save_plot = True
    save_file = 'eval_val_rollouts'
    # include training data or just validation
    eval_sets = ['Train', 'Validation']
    #eval_sets = ['Validation']
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_dict= {
        'rssm_discrete': [DreamerV2, 'config', 'rssm_h1024_l2_mlp1024', 'rssm_h1024_l2_mlp1024', 'RSSM Discrete'],
        'multistep_predictor': [MultistepPredictor, 'config', 'mp_input_embed_h1024_l2_mlp1024_l2', 'mp_input_embed_h1024_l2_mlp1024_l2', 'Multistep Predictor'],
        'multistep_delta': [MultistepDelta, 'config', 'multistep_delta_h1024_l2_mlp1024_l2', 'multistep_delta_h1024_l2_mlp1024_l2', 'Multistep Delta']
        }
    keys2analyze = ['rssm_discrete', 'multistep_predictor']
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
                
            mse, recon_features = eval_forward_rollout_mse(model, input_data, batch_size, burn_in_length, rollout_length)
                
            results.append({
                        'Model': model_name,
                        'Task': 'Reconstruction error',
                        'Metric': 'MSE',
                        'Score': mse,
                        'Eval Set': train_or_val
                    })
            
            for item in recon_features.items():
                results.append({
                        'Model': model_name,
                        'Task': 'Reconstruct Features',
                        'Metric': item[0],
                        'Score': item[1],
                        'Eval Set': train_or_val
                    })
            
            
    df_plot = pd.DataFrame(results)
    task_names = df_plot.Task.unique()
    
    # Plot the results using seaborn
    fig = plt.figure()
    task = task_names[0]
    sns.barplot(x='Model', y='Score', hue='Eval Set', data=df_plot[df_plot['Task']==task]) 
    plt.title('Model Error', fontsize=18)
    plt.xlabel('Model Name', fontsize=16) 
    plt.ylabel('Reconstruction Error (MSE)', fontsize=14) 
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    if save_plot:
        df_plot.to_csv(analysis_dir+'results/figures/'+save_file+'.csv')
        plt.savefig(analysis_dir+'results/figures/'+save_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot reconstructed feature correlations
    # Take all objects and agents as a whole
    task = 'Reconstruct Features'
    recon_feats_df = df_plot.loc[(df_plot['Task']==task) & (df_plot['Eval Set']=='Validation')]
    recon_feats_df = recon_feats_df.copy() # to avoid copy warnings
    metric_names = list(recon_feats_df['Metric'])
    new_metric_names = [string.replace('0', '').replace('1', '').replace('2', '') for string in metric_names]
    recon_feats_df.loc[:,'Feature'] = new_metric_names

    fig = plt.figure()
    sns.barplot(x='Feature', y='Score', hue='Model', data=recon_feats_df, errorbar=None, palette="Set2")
    plt.title('Evaluate Reconstructed Inputs - Validation Set', fontsize=18)
    plt.ylabel('Correlation (r)', fontsize=14)
    plt.xlabel('Feature', fontsize=16)
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    if save_plot:
        df_plot.to_csv(analysis_dir+'results/figures/'+save_file+'_feature_corr.csv')
        plt.savefig(analysis_dir+'results/figures/'+save_file+'_feature_corr', dpi=300, bbox_inches='tight')
    plt.show()