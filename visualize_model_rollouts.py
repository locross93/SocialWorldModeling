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
import platform
import numpy as np
import matplotlib.pyplot as plt
import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage

from analysis_utils import load_trained_model, data_columns
from annotate_pickup_timepoints import annotate_pickup_timepoints
from annotate_goal_timepoints import annotate_goal_timepoints

from constants import MODEL_DICT_VAL

if platform.system() == 'Windows':
    # We are running on Windows
    analysis_dir = '/Users/locro/Documents/Stanford/SocialWorldModeling/'
    data_dir = '/Users/locro/Documents/Stanford/analysis/data/'
    checkpoint_dir = analysis_dir
elif platform.system() == 'Linux':
    # We are running on Linux
    analysis_dir = '/home/locross/SocialWorldModeling/'
    data_dir = '/mnt/fs2/locross/analysis/data/'
    checkpoint_dir = '/mnt/fs2/locross/analysis/'
    
    
def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_key', type=str, required=True, help='Model to use for visualization')
    parser.add_argument('--train_or_val', type=str, default='val', help='Training or Validation Set')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    # which trial to visualize
    parser.add_argument('--trial_type', type=str, default='single_goal', help='Trial Type') # single_goal, multi_goal, all
    parser.add_argument('--trial_num', type=int, default=0, help='Trial Type')
    parser.add_argument('--goal_num', type=int, default=0, help='Goal number to burn in to for multigoal trials')
    parser.add_argument('--burn_in_length', type=int, default=50, help='Burn in length when trial type == all')
    # which visualizations
    parser.add_argument('--plot_traj_subplots', type=bool, default=True, help='Make subplot of true vs predicted trajectory')
    parser.add_argument('--make_video', type=bool, default=True, help='Make video - compare real and reconstructed traj side by side')
    return parser.parse_args()

def load_config(file):
    with open(file) as f:
        config = json.load(f)
    return config

def make_traj_subplots(x_true, x_pred, subplot_dims, steps, save_file):
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
        ax[row, col].set_title('Frame: '+str(step))
        
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
    # save
    plt.savefig(save_file, dpi=300)
    
    return fig, ax

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def make_frame_compare(t):
    t_ind = find_nearest(timepoints, t)
    
    # TO DO - MAKE IT CLEAR WHEN BURN IN ENDS
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
    
    # returning numpy image
    return mplfig_to_npimage(fig)

if __name__ == '__main__':
    args = load_args()
    
    model_info = MODEL_DICT_VAL[args.model_key]
    model_name = model_info['model_label']
    model = load_trained_model(model_info, args.device)
    
    # load data
    data_file = os.path.join(data_dir, 'train_test_splits_3D_dataset.pkl')
    loaded_dataset = pickle.load(open(data_file, 'rb'))
    train_dataset, test_dataset = loaded_dataset
    if args.train_or_val == 'train':
        input_data = train_dataset.dataset.tensors[0][train_dataset.indices,:,:]
    else:
        input_data = test_dataset.dataset.tensors[0][test_dataset.indices,:,:]
    
    # select trajectory where goal occurred
    pickup_timepoints = annotate_pickup_timepoints(args.train_or_val, pickup_or_move='move')
    single_goal_trajs = np.where((np.sum(pickup_timepoints > -1, axis=1) == 1))[0]
    multi_goal_trajs = np.where((np.sum(pickup_timepoints > -1, axis=1) == 3))[0]
    
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
        
    rollout_length = input_data.size(1) - burn_in_length
    x = input_data[traj_ind,:,:].unsqueeze(0)
    x_true = x[0,:burn_in_length+rollout_length,:].numpy()
    x_pred = x_true.copy()
    #rollout_x = model.forward_rollout(x, burn_in_length, rollout_length).cpu().detach().numpy()
    rollout_x = model.variable_length_rollout(x, steps2pickup, rollout_length).cpu().detach()
    x_pred[burn_in_length:,:] = rollout_x 
    
    viz_dir = os.path.join(analysis_dir, 'results/viz_trajs',args.model_key)
    if not os.path.exists(viz_dir):
        os.mkdir(viz_dir)
        
    if args.trial_type == 'single_goal':
        save_file = os.path.join(viz_dir, 'traj_subplot'+'_sg_trial'+str(args.trial_num))
    elif args.trial_type == 'multi_goal':
        save_file = os.path.join(viz_dir, 'traj_subplot'+'_mg_trial'+str(args.trial_num))
    elif args.trial_type == 'all':
        save_file = os.path.join(viz_dir, 'traj_subplot'+'_trial'+str(args.trial_num))
        
    # make subplot of true vs predicted trajectory
    if args.plot_traj_subplots:
        subplot_dims = (3,3)
        num_frames = x_true.shape[0]
        num_steps = subplot_dims[0] * subplot_dims[1]
        steps = np.linspace(0, num_frames-1, num_steps, endpoint=True).astype(int)
        fig, ax = make_traj_subplots(x_true, x_pred, subplot_dims, steps, save_file)
    
    # make video - compare real and reconstructed traj side by side
    if args.make_video:
        timepoints = np.linspace(0, 10, x_true.shape[0])
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        compare_clip = mpy.VideoClip(make_frame_compare, duration=10)
        compare_clip.write_videofile(save_file+'.mp4', fps=30)