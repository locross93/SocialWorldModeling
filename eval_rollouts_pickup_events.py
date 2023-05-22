# -*- coding: utf-8 -*-
"""
Created on Sun May 21 20:16:23 2023

@author: locro
"""

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pickle
import platform
from functools import reduce
from scipy.spatial import distance
import seaborn as sns
import torch

from analysis_utils import load_trained_model
from models import DreamerV2, MultistepPredictor, MultistepDelta, TransformerMSPredictor, TransformerWorldModel
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
    
def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) > stepsize)[0]+1)
    
def detect_obj_pick_up(trial_agent_pos, trial_obj_pos):
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
            index_percentage = amax_delta / len(obj_delta_event) * 100
            if index_percentage < 0.1 or index_percentage > 0.9:
                picked_up = True
                dropped = True
            
    if picked_up and dropped:
        return True
    else:
        return False
    
def eval_pickup_events(input_matrices, recon_matrices, model_name=''):
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
    obj_pick_up_flag = np.zeros([num_trials, 3])
    for i in range(num_trials):
        trial_x = input_matrices[i,:,:]
        for j in range(num_objs):
            pos_inds = [data_columns.index('obj'+str(j)+'_'+dim) for dim in dims]
            trial_obj_pos = trial_x[:,pos_inds]
            agent_moved = []
            for k in range(2):
                pos_inds2 = [data_columns.index('agent'+str(k)+'_'+dim) for dim in dims]
                trial_agent_pos = trial_x[:,pos_inds2]
                move_bool = detect_obj_pick_up(trial_agent_pos, trial_obj_pos)
                agent_moved.append(move_bool)
            if any(agent_moved):
                obj_pick_up_flag[i,j] = 1
                
    recon_pick_up_flag = np.zeros([num_trials, 3])
    for i in range(num_trials):
        trial_x = recon_matrices[i,:,:]
        for j in range(num_objs):
            pos_inds = [data_columns.index('obj'+str(j)+'_'+dim) for dim in dims]
            trial_obj_pos = trial_x[:,pos_inds]
            agent_moved = []
            for k in range(2):
                pos_inds2 = [data_columns.index('agent'+str(k)+'_'+dim) for dim in dims]
                trial_agent_pos = trial_x[:,pos_inds2]
                move_bool = detect_obj_pick_up(trial_agent_pos, trial_obj_pos)
                agent_moved.append(move_bool)
            if any(agent_moved):
                recon_pick_up_flag[i,j] = 1
                
    pickup_idxs = np.where(obj_pick_up_flag)
    accuracy = np.mean(recon_pick_up_flag[pickup_idxs])
    
    return accuracy, obj_pick_up_flag, recon_pick_up_flag
    
def eval_pickup_events_in_rollouts(model, input_data, ds='First'):
    if ds == 'First':
        # first dataset
        pickup_timepoints = annotate_pickup_timepoints(train_or_val, pickup_or_move='move')
        goal_timepoints = annotate_goal_timepoints(train_or_val)
        single_goal_trajs = np.where((np.sum(pickup_timepoints > -1, axis=1) == 1))[0]
        multi_goal_trajs = np.where((np.sum(pickup_timepoints > -1, axis=1) == 3))[0]
    else:
        # use event log for new datasets - TO DO
        pickup_timepoints = []
        
    # TO DO, ANALYZE EVERY PICKUP EVENT SEPARATELY, INCLUDING MULTI GOAL TRAJS
        
    num_single_goal_trajs = len(single_goal_trajs)
    imagined_trajs = np.zeros([num_single_goal_trajs, input_data.shape[1], input_data.shape[2]])
    num_timepoints = input_data.size(1)
    real_trajs = []
    imag_trajs = []
    for i,row in enumerate(single_goal_trajs):
        if i%50 == 0:
            print(i)
        x = input_data[row,:,:].unsqueeze(0)
        # get the only pick up point in the trajectory
        steps2pickup = np.max(pickup_timepoints[row,:]).astype(int)
        # burn in until right before the pick up point
        burn_in_length = steps2pickup - 10
        # store the steps before pick up with real frames in imagined_trajs
        imagined_trajs[i,:burn_in_length,:] = x[:,:burn_in_length,:].cpu()
        # rollout model for the rest of the trajectory
        rollout_length = num_timepoints - burn_in_length
        rollout_x = model.forward_rollout(x, burn_in_length, rollout_length).cpu().detach()
        # get end portion of true trajectory to compare to rollout
        real_traj = x[:,burn_in_length:,:].to("cpu")
        assert rollout_x.size() == real_traj.size()
        real_trajs.append(real_traj)
        imag_trajs.append(rollout_x)
        # store the steps after pick up with predicted frames in imagined_trajs
        imagined_trajs[i,burn_in_length:,:] = rollout_x
        
    full_trajs = input_data[single_goal_trajs,:,:].cpu()
    scores, y_labels, y_recon = eval_pickup_events(full_trajs, imagined_trajs)
    # evaluate whether only appropriate goals (after object picked up) are reconstructed
    pickup_subset = pickup_timepoints[single_goal_trajs,:]
    indices = np.argwhere(pickup_subset > -1)
    accuracy = np.mean(y_recon[indices[:,0],indices[:,1]])
    
    return accuracy

if __name__ == "__main__":
    # train test splits
    data_file = data_dir+'train_test_splits_3D_dataset.pkl'
    with open(data_file, 'rb') as f:
        loaded_dataset = pickle.load(f)
    train_dataset, test_dataset = loaded_dataset
    
    save_plot = True
    save_file = 'eval_rollouts_pickup_events_rssm_mp'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_or_val = 'val'
    
    if train_or_val == 'train':
        input_data = train_dataset.dataset.tensors[0][train_dataset.indices,:,:]
    elif train_or_val == 'val':
        input_data = test_dataset.dataset.tensors[0][test_dataset.indices,:,:]
    num_timepoints = input_data.size(1)
    
    model_dict = {
        'rssm_disc': {'class': DreamerV2, 'config': 'rssm_disc_default_config.json', 
                      'model_dir': 'rssm_disc', 'model_label': 'RSSM Discrete'},
        'multistep_pred': {'class': MultistepPredictor, 'config': 'multistep_predictor_default_config.json', 
                      'model_dir': 'multistep_predictor', 'model_label': 'Multistep Predictor'},
        'transformer': {'class': TransformerMSPredictor, 'config': 'transformer_default_config.json', 
                      'model_dir': 'transformer_mp', 'model_label': 'Transformer MP'},
        'transformer_wm': {'class': TransformerWorldModel, 'config': 'transformer_wm_default_config.json', 
                      'model_dir': 'transformer_wm', 'model_label': 'Transformer'},
        }
    
    keys2analyze = ['rssm_disc', 'multistep_pred']
    results = []
    for key in keys2analyze:
        print(key)
        model_info = model_dict[key]
        model_name = model_info['model_label']
        model = load_trained_model(model_info, DEVICE)
        
        # put on gpu
        input_data = input_data.to(DEVICE)
        
        accuracy = eval_pickup_events_in_rollouts(model, input_data)
        
        results.append({
                    'Model': model_name,
                    'Accuracy': accuracy,
                })
        
    df_plot = pd.DataFrame(results)
    
    # Plot the results using seaborn
    fig = plt.figure()
    sns.barplot(x='Model', y='Accuracy', data=df_plot) 
    plt.title('Evaluate Pick Up Events From Forward Rollouts', fontsize=14)
    plt.xlabel('Model Name', fontsize=16) 
    plt.ylabel('Accuracy', fontsize=16)
    plt.ylim([0, 1])
    if save_plot:
        df_plot.to_csv(analysis_dir+'results/figures/'+save_file+'.csv')
        plt.savefig(analysis_dir+'results/figures/'+save_file, dpi=300, bbox_inches='tight')
    plt.show()