# -*- coding: utf-8 -*-
"""
Created on Thu May 11 16:17:37 2023

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
from annotate_goal_timepoints import eval_recon_goals

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
    
def eval_multigoal_events_in_rollouts(model, input_data, ds='First'):
    if ds == 'First':
        # first dataset
        pickup_timepoints = annotate_pickup_timepoints(train_or_val, pickup_or_move='move')
        multi_goal_trajs = np.where((np.sum(pickup_timepoints > -1, axis=1) == 3))[0]
    else:
        # use event log for new datasets - TO DO
        pickup_timepoints = []
        
    num_multi_goal_trajs = len(multi_goal_trajs)
    imagined_trajs = np.zeros([num_multi_goal_trajs, input_data.shape[1], input_data.shape[2]])
    num_timepoints = input_data.size(1)
    real_trajs = []
    imag_trajs = []
    for i,row in enumerate(multi_goal_trajs):
        if i%50 == 0:
            print(i)
        x = input_data[row,:,:].unsqueeze(0)
        # burn in to the pick up point of the 2nd object that is picked up, so its unambiguous that all objects will be delivered
        steps2pickup = np.sort(pickup_timepoints[row,:])[1].astype(int)
        # store the steps before pick up with real frames in imagined_trajs
        imagined_trajs[i,:steps2pickup,:] = x[:,:steps2pickup,:].cpu()
        # rollout model for the rest of the trajectory
        rollout_length = num_timepoints - steps2pickup
        rollout_x = model.forward_rollout(x, steps2pickup, rollout_length).cpu().detach()
        # get end portion of true trajectory to compare to rollout
        real_traj = x[:,steps2pickup:,:].to("cpu")
        assert rollout_x.size() == real_traj.size()
        real_trajs.append(real_traj)
        imag_trajs.append(rollout_x)
        imagined_trajs[i,steps2pickup:,:] = rollout_x
    x_true = torch.cat(real_trajs, dim=1)
    x_hat = torch.cat(imag_trajs, dim=1)
    mse = ((x_true - x_hat)**2).mean().item()
    
    full_trajs = input_data[multi_goal_trajs,:,:].cpu()
    scores, y_labels, y_recon = eval_recon_goals(full_trajs, imagined_trajs, final_location=False, plot=False)
    # 100% accuracy is all goal labels are == 1
    assert np.mean(y_labels) == 1.0
    # get accuracies separately for 2nd object and 3rd object (3rd object the hardest to imagine properly)
    goals_obj1 = []
    goals_obj2 = []
    goals_obj3 = []
    for i,row in enumerate(multi_goal_trajs):
        pickup_seq = np.argsort(pickup_timepoints[row,:])
        goals_obj1.append(y_recon[i,pickup_seq[0]])
        goals_obj2.append(y_recon[i,pickup_seq[1]])
        goals_obj3.append(y_recon[i,pickup_seq[2]])

    acc_g2 = np.mean(goals_obj2)
    acc_g3 = np.mean(goals_obj3)
    
    return acc_g2, acc_g3, mse
    

if __name__ == "__main__":
    # train test splits
    data_file = data_dir+'train_test_splits_3D_dataset.pkl'
    with open(data_file, 'rb') as f:
        loaded_dataset = pickle.load(f)
    train_dataset, test_dataset = loaded_dataset
    
    save_plot = False
    save_file = 'eval_rollouts_multigoal_events'
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
        
        acc_g2, acc_g3, mse = eval_multigoal_events_in_rollouts(model, input_data)
        
        results.append({
                    'Model': model_name,
                    'Accuracy': acc_g2,
                    'Goal Num': '2nd Goal',
                    'MSE': mse
                })
        
        results.append({
                    'Model': model_name,
                    'Accuracy': acc_g3,
                    'Goal Num': '3rd Goal',
                    'MSE': mse
                })
        
    df_plot = pd.DataFrame(results)
    
    # Plot the results using seaborn
    fig = plt.figure()
    sns.barplot(x='Model', y='Accuracy', hue='Goal Num', data=df_plot) 
    plt.title('Evaluate Multistep Goal Events From Forward Rollouts', fontsize=14)
    plt.xlabel('Model Name', fontsize=16) 
    plt.ylabel('Accuracy', fontsize=16)
    plt.ylim([0, 1])
    if save_plot:
        df_plot.to_csv(analysis_dir+'results/figures/'+save_file+'.csv')
        plt.savefig(analysis_dir+'results/figures/'+save_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    fig = sns.barplot(x='Model', y='MSE', data=df_plot, ci=None) 
    plt.title('Prediction Error of Goal Events in Imagination', fontsize=18)
    plt.xlabel('Model Name', fontsize=16) 
    plt.ylabel('MSE', fontsize=16)
    if save_plot:
        plt.savefig(analysis_dir+'results/figures/'+save_file+'_recon_error', dpi=300, bbox_inches='tight')
    plt.show()
            
        
        