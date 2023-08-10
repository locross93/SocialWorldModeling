# -*- coding: utf-8 -*-
"""
Created on Tue May 16 15:23:30 2023

@author: locro
"""
# example usage: python visualize_model_rollouts.py --model_key transformer_wm --trial_type single_goal --trial_num 0

import os
import argparse
import json
import pandas as pd
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage

from analysis_utils import load_trained_model, get_data_columns, inverse_normalize
from annotate_pickup_timepoints import annotate_pickup_timepoints
from annotate_goal_timepoints import annotate_goal_timepoints

from constants_lc import MODEL_DICT_VAL, DEFAULT_VALUES, DATASET_NUMS
    
    
def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config_dir', type=str, action='store',
                        default=DEFAULT_VALUES['model_config_dir'],
                        help='Model config directory')
    parser.add_argument('--analysis_dir', type=str, action='store',
                        default=DEFAULT_VALUES['analysis_dir'], 
                        help='Analysis directory')
    parser.add_argument('--data_dir', type=str,
                         default=DEFAULT_VALUES['data_dir'], 
                         help='Data directory')
    parser.add_argument('--dataset', type=str,
                         default=DEFAULT_VALUES['dataset'], 
                         help='Dataset name')
    parser.add_argument('--checkpoint_dir', type=str, 
                        default=DEFAULT_VALUES['checkpoint_dir'], 
                        help='Checkpoint directory')
    parser.add_argument('--model_key', type=str, required=True, help='Model to use for visualization')
    parser.add_argument('--train_or_val', type=str, default='val', help='Training or Validation Set')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    # which trial to visualize
    parser.add_argument('--trial_type', type=str, default='single_goal', help='Trial Type') # single_goal, multi_goal, all
    parser.add_argument('--trial_num', type=int, default=0, help='Trial Type')
    parser.add_argument('--goal_num', type=int, default=1, help='Goal number to burn in to for multigoal trials')
    parser.add_argument('--burn_in_length', type=int, default=50, help='Burn in length when trial type == all')
    # which visualizations
    parser.add_argument('--plot_traj_subplots', type=bool, default=True, help='Make subplot of true vs predicted trajectory')
    parser.add_argument('--make_video', type=int, default=1, help='Make video - compare real and reconstructed traj side by side')
    parser.add_argument('--eval_event_model', type=bool, default=False, help='Visualize Predicted Next Event')
    return parser.parse_args()

def load_config(file):
    with open(file) as f:
        config = json.load(f)
    return config

def inverse_transform_horizon(event_horizon_normalized, episode_length=300, train_burn_in_length=50):
    event_horizon = (event_horizon_normalized * ((episode_length - train_burn_in_length) - 1)) + 1
    return event_horizon

def make_traj_subplots(x_true, x_pred, subplot_dims, steps, save_file, data_columns, burn_in_length):
    # ie subplot_dims = (3,3)
    fig, ax = plt.subplots(subplot_dims[0], subplot_dims[1], figsize=(15, 15))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    assert len(x_true.shape) == 2
    assert len(x_pred.shape) == 2
    assert x_true.shape == x_pred.shape
    
    for i,step in enumerate(steps):
        row = int(i / 3)
        col = i % 3
        
        ax[row, col].set_xlim(-7, 7)
        ax[row, col].set_ylim(-7, 7)
        if step <= burn_in_length:
            ax[row, col].set_title('Burn-in: Frame: '+str(step))
        else:
            ax[row, col].set_title('Rollout: Frame: '+str(step))
        
        # plot real trajectory
        for obj_num in range(3):
            color = colors[obj_num]
            x_ind = data_columns.index('obj'+str(obj_num)+'_x')
            z_ind = data_columns.index('obj'+str(obj_num)+'_z')
            ax[row, col].plot(x_true[step,x_ind], x_true[step,z_ind], marker='o', markerfacecolor=color, markeredgecolor='black')
        for agent_num in range(2):
            color = colors[agent_num+3]
            x_ind = data_columns.index('agent'+str(agent_num)+'_x')
            z_ind = data_columns.index('agent'+str(agent_num)+'_z')
            ax[row, col].plot(x_true[step,x_ind], x_true[step,z_ind], marker='^', markersize=16, markerfacecolor=color, markeredgecolor='black')
        # observer
        ax[row, col].plot(0.0, -6.0, marker='s', markersize=12)
        
        # plot predicted trajectory
        for obj_num in range(3):
            color = colors[obj_num]
            x_ind = data_columns.index('obj'+str(obj_num)+'_x')
            z_ind = data_columns.index('obj'+str(obj_num)+'_z')
            ax[row, col].plot(x_pred[step,x_ind], x_pred[step,z_ind], marker='o', markerfacecolor=color, markeredgecolor=color, alpha=0.5)
        for agent_num in range(2):
            color = colors[agent_num+3]
            x_ind = data_columns.index('agent'+str(agent_num)+'_x')
            z_ind = data_columns.index('agent'+str(agent_num)+'_z')
            ax[row, col].plot(x_pred[step,x_ind], x_pred[step,z_ind], marker='^', markersize=16, markerfacecolor=color, markeredgecolor=color, alpha=0.5)
            
        # remove x and y tick labels
        ax[row, col].set_xticks([])
        ax[row, col].set_yticks([])

        if step <= burn_in_length:
            for spine in ax[row, col].spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(2.5)
    # save
    plt.savefig(save_file, dpi=300)
    
    return fig, ax

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def make_frame_compare(t):
    t_ind = find_nearest(timepoints, t)
    
    # clear
    ax[0].clear()
    ax[1].clear()
    
    ax[0].set_xlim(-7, 7)
    ax[0].set_ylim(-7, 7)
    for obj_num in range(3):
        x_ind = data_columns.index('obj'+str(obj_num)+'_x')
        z_ind = data_columns.index('obj'+str(obj_num)+'_z')
        ax[0].plot(x_true[t_ind,x_ind], x_true[t_ind,z_ind], marker='o')
    for agent_num in range(2):
        x_ind = data_columns.index('agent'+str(agent_num)+'_x')
        z_ind = data_columns.index('agent'+str(agent_num)+'_z')
        ax[0].plot(x_true[t_ind,x_ind], x_true[t_ind,z_ind], marker='^', markersize=16)
    # observer
    ax[0].plot(0.0, -6.0, marker='s', markersize=12)
    ax[0].set_title('Real Trajectory')

    ax[1].set_xlim(-7, 7)
    ax[1].set_ylim(-7, 7)
    for obj_num in range(3):
        x_ind = data_columns.index('obj'+str(obj_num)+'_x')
        z_ind = data_columns.index('obj'+str(obj_num)+'_z')
        ax[1].plot(x_pred[t_ind,x_ind], x_pred[t_ind,z_ind], marker='o')
    for agent_num in range(2):
        x_ind = data_columns.index('agent'+str(agent_num)+'_x')
        z_ind = data_columns.index('agent'+str(agent_num)+'_z')
        ax[1].plot(x_pred[t_ind,x_ind], x_pred[t_ind,z_ind], marker='^', markersize=16)
    ax[1].plot(0.0, -6.0, marker='s', markersize=12)
    ax[1].set_title('Reconstructed Trajectory')
    
    # if the current time index is less than the burn-in time, add the red border and 'BURN IN' text
    if t_ind < burn_in_length:
        for axis in ax:
            for spine in axis.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(2.5)  # adjust as needed for your desired thickness

        # add 'BURN IN' text and keep its handle
        burn_in_text = fig.suptitle('BURN IN', fontsize=20, color='red')
        #fig.text(0.5, 0.9, 'BURN IN', ha='center', va='center', fontsize=20, color='red')
    else:
        for axis in ax:
            for spine in axis.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.0)  # set back to default values

        no_text = fig.suptitle('', fontsize=20)
    
    # returning numpy image
    return mplfig_to_npimage(fig)

def plot_one_frame_compare(x_burnin, x_true, x_pred, data_columns, event_horizon=None, pred_horizon=None):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # clear
    ax[0].clear()
    ax[1].clear()
    ax[2].clear()
    
    ax[0].set_xlim(-7, 7)
    ax[0].set_ylim(-7, 7)
    for obj_num in range(3):
        x_ind = data_columns.index('obj'+str(obj_num)+'_x')
        z_ind = data_columns.index('obj'+str(obj_num)+'_z')
        ax[0].plot(x_burnin[0,x_ind], x_burnin[0,z_ind], marker='o')
    for agent_num in range(2):
        x_ind = data_columns.index('agent'+str(agent_num)+'_x')
        z_ind = data_columns.index('agent'+str(agent_num)+'_z')
        ax[0].plot(x_burnin[0,x_ind], x_burnin[0,z_ind], marker='^', markersize=16)
    # observer
    ax[0].plot(0.0, -6.0, marker='s', markersize=12)
    ax[0].set_title('Last Burn-in State')

    ax[1].set_xlim(-7, 7)
    ax[1].set_ylim(-7, 7)
    for obj_num in range(3):
        x_ind = data_columns.index('obj'+str(obj_num)+'_x')
        z_ind = data_columns.index('obj'+str(obj_num)+'_z')
        ax[1].plot(x_true[0,x_ind], x_true[0,z_ind], marker='o')
    for agent_num in range(2):
        x_ind = data_columns.index('agent'+str(agent_num)+'_x')
        z_ind = data_columns.index('agent'+str(agent_num)+'_z')
        ax[1].plot(x_true[0,x_ind], x_true[0,z_ind], marker='^', markersize=16)
    # observer
    ax[1].plot(0.0, -6.0, marker='s', markersize=12)
    ax[1].set_title('Next AC Signal Event State')
    if event_horizon is not None:
        ax[1].text(3.6, 6, f'Horizon {event_horizon}', color='red')

    ax[2].set_xlim(-7, 7)
    ax[2].set_ylim(-7, 7)
    for obj_num in range(3):
        x_ind = data_columns.index('obj'+str(obj_num)+'_x')
        z_ind = data_columns.index('obj'+str(obj_num)+'_z')
        ax[2].plot(x_pred[0,x_ind], x_pred[0,z_ind], marker='o')
    for agent_num in range(2):
        x_ind = data_columns.index('agent'+str(agent_num)+'_x')
        z_ind = data_columns.index('agent'+str(agent_num)+'_z')
        ax[2].plot(x_pred[0,x_ind], x_pred[0,z_ind], marker='^', markersize=16)
    ax[2].plot(0.0, -6.0, marker='s', markersize=12)
    ax[2].set_title('Predicted Next Event State')
    if pred_horizon is not None:
        ax[2].text(3.6, 6, f'Horizon {pred_horizon}', color='red')
    
    return fig, ax

if __name__ == '__main__':
    args = load_args()
    torch.manual_seed(DEFAULT_VALUES['eval_seed'])
    print("Using seed: {}".format(DEFAULT_VALUES['eval_seed']))
    model_info = MODEL_DICT_VAL[args.model_key]
    model_name = model_info['model_label']
    model = load_trained_model(model_info, args)
    model.eval()
    
    # load data
    dataset_file = os.path.join(args.data_dir, args.dataset)
    loaded_dataset = pickle.load(open(dataset_file, 'rb'))
    if dataset_file == os.path.join(args.data_dir, 'data_norm_velocity_val_only.pkl'):
        input_data = loaded_dataset
        # reset filename for exp_info_file
        dataset_file = os.path.join(args.data_dir, 'data_norm_velocity.pkl')
    else:
        train_dataset, test_dataset = loaded_dataset
        if args.train_or_val == 'train':
            input_data = train_dataset.dataset.tensors[0][train_dataset.indices,:,:]
        else:
            input_data = test_dataset.dataset.tensors[0][test_dataset.indices,:,:]
        
    if args.dataset in DATASET_NUMS and DATASET_NUMS[args.dataset] == 1:
        # first dataset use states to define events
        # select trajectory where goal occurred
        pickup_timepoints = annotate_pickup_timepoints(loaded_dataset, args.train_or_val, pickup_or_move='move', ds_num=DATASET_NUMS[args.dataset])
        single_goal_trajs = np.where((np.sum(pickup_timepoints > -1, axis=1) == 1))[0]
        multi_goal_trajs = np.where((np.sum(pickup_timepoints > -1, axis=1) == 3))[0]
        data_columns = get_data_columns(DATASET_NUMS[args.dataset])
    else:
        # load dataset info
        exp_info_file = dataset_file[:-4]+'_exp_info.pkl'
        if os.path.isfile(exp_info_file):
            with open(exp_info_file, 'rb') as f:
                exp_info_dict = pickle.load(f)
        else:
            print('DS info dict not found')
        # second dataset use event logger to define events
        pickup_timepoints = exp_info_dict[args.train_or_val]['pickup_timepoints']
        single_goal_trajs = exp_info_dict[args.train_or_val]['single_goal_trajs']
        multi_goal_trajs = exp_info_dict[args.train_or_val]['multi_goal_trajs']
        data_columns = exp_info_dict['data_columns']
        if 'min_max_values' in exp_info_dict:
            # data is normalize, project it back into regular input space
            min_values, max_values = exp_info_dict['min_max_values']
            velocity = any('vel' in s for s in data_columns)
            input_data = inverse_normalize(input_data, max_values.astype(np.float32), min_values.astype(np.float32), velocity)
            
    if args.trial_type == 'single_goal':
        traj_ind = single_goal_trajs[args.trial_num]
        steps2pickup = np.max(pickup_timepoints[traj_ind,:]).astype(int)
        burn_in_length = steps2pickup
    elif args.trial_type == 'multi_goal':
        traj_ind = multi_goal_trajs[args.trial_num]
        goal_num = args.goal_num
        steps2pickup = np.sort(pickup_timepoints[traj_ind,:])[goal_num].astype(int)
        burn_in_length = steps2pickup
    elif args.trial_type == 'all':
        # all trial types
        traj_ind = args.trial_num
        burn_in_length = args.burn_in_length
    print('Burn in length: {}'.format(burn_in_length))
    
    rollout_length = input_data.size(1) - burn_in_length
    x = input_data[traj_ind,:,:].unsqueeze(0)
    if x.dtype == torch.float64:
        x = x.float()
    x_true = x[0,:burn_in_length+rollout_length,:].numpy()
    x_pred = x_true.copy()
    # if args.model_key == 'transformer_wm':
    #     rollout_x = model.variable_length_rollout(x, steps2pickup, rollout_length).cpu().detach()
    # else:
    with torch.no_grad():
        rollout_x = model.forward_rollout(x, burn_in_length, rollout_length).cpu().numpy()
    x_pred[burn_in_length:,:] = rollout_x 
    
    if args.eval_event_model:
        pred_event_state = model.event_hat.numpy()
        if pred_event_state.shape[1] == 36:
            horizon_output = True
            pred_horizon = pred_event_state[:,-1]
            pred_horizon = int(inverse_transform_horizon(pred_horizon))
            pred_event_state = pred_event_state[:,:-1]
        else:
            horizon_output = False
        # load events data
        events_ds_file = os.path.join(args.analysis_dir,'results/event_inds/event_inds_mp_ds2.pkl')
        events_dataset = pickle.load(open(events_ds_file, 'rb'))
        event_inds = events_dataset[args.train_or_val]
        # get closest future event state in that trajectory from burn in state
        traj_event_inds = np.array(event_inds[traj_ind])
        burn_in_ind = burn_in_length - 1
        last_burnin_state = input_data[traj_ind,burn_in_ind,:].unsqueeze(0)
        closest_event_ind = np.min(traj_event_inds[np.where(traj_event_inds > burn_in_ind)[0]])
        # get event horizon from end state
        event_horizon = closest_event_ind - burn_in_ind
        event_state = input_data[traj_ind,closest_event_ind,:].unsqueeze(0)
        assert torch.equal(event_state, x[:,closest_event_ind,:])
        #event_state = event_state.numpy()
        if horizon_output:
            fig, ax = plot_one_frame_compare(last_burnin_state, event_state, pred_event_state, data_columns, event_horizon, pred_horizon)
        else:
            fig, ax = plot_one_frame_compare(last_burnin_state, event_state, pred_event_state, data_columns)
       
        # TEMP ADD TRUE NEXT EVENT
        # condition ms predictor on predicted next event
        # concatenate event_hat and event_horizon_hat
        # first normalize event_horizon
        #event_horizon = float((event_horizon - 1) / ((300 - 50) - 1))
        #event_state = torch.cat([event_state, torch.tensor(event_horizon).unsqueeze(0).unsqueeze(0)], dim=-1)
        #rollout_x  = model.mp_model.forward_rollout(x, event_state, burn_in_length, rollout_length)
        # Call MSPredictorEventContext's forward_rollout with event_hat
        with torch.no_grad():
            rollout_x  = model.mp_model.forward_rollout(x, pred_event_state, burn_in_length, rollout_length)
        x_pred[burn_in_length:,:] = rollout_x 
    
    viz_dir = os.path.join(args.analysis_dir, 'results/viz_trajs',args.model_key)
    if not os.path.exists(viz_dir):        
        os.makedirs(viz_dir)
        
    if args.trial_type == 'single_goal':
        save_file = os.path.join(viz_dir, 'traj_sg_trial'+str(args.trial_num))
    elif args.trial_type == 'multi_goal':
        save_file = os.path.join(viz_dir, 'traj_mg_trial'+str(args.trial_num))
    elif args.trial_type == 'all':
        save_file = os.path.join(viz_dir, 'traj_trial'+str(args.trial_num))
    # if a training set trial, add suffix
    if args.train_or_val == 'train':
        save_file = save_file+'_train'
        
    # make subplot of true vs predicted trajectory
    if args.plot_traj_subplots:
        subplot_dims = (3,3)
        num_frames = x_true.shape[0]
        num_steps = subplot_dims[0] * subplot_dims[1]
        steps = np.linspace(0, num_frames-1, num_steps, endpoint=True).astype(int)
        #steps = np.linspace(0, 56, num_steps, endpoint=True).astype(int)
        fig, ax = make_traj_subplots(x_true, x_pred, subplot_dims, steps, save_file, data_columns, burn_in_length)
    
    # make video - compare real and reconstructed traj side by side
    if args.make_video:
        timepoints = np.linspace(0, 10, x_true.shape[0])
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        compare_clip = mpy.VideoClip(make_frame_compare, duration=10)
        compare_clip.write_videofile(save_file+'.mp4', fps=30)
