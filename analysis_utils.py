# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 12:00:58 2023

@author: locro
"""

import argparse
import json
import os
import torch
import pickle
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import torch

from constants_lc import DEFAULT_VALUES, MODEL_DICT_VAL

def get_data_columns(ds_num):
    if ds_num == 1:
        data_columns = ['obj0_x', 'obj0_y', 'obj0_z', 'obj1_x', 'obj1_y', 'obj1_z', 'obj2_x',
               'obj2_y', 'obj2_z', 'agent0_x', 'agent0_y', 'agent0_z', 'agent0_rot_x',
               'agent0_rot_y', 'agent0_rot_z', 'agent0_rot_w', 'agent1_x', 'agent1_y',
               'agent1_z', 'agent1_rot_x', 'agent1_rot_y', 'agent1_rot_z',
               'agent1_rot_w'] 
    elif ds_num == 2:
        data_columns = ['obj0_x', 'obj0_y', 'obj0_z', 'obj0_rot_x', 'obj0_rot_y', 'obj0_rot_z',
               'obj0_rot_w', 'obj1_x', 'obj1_y', 'obj1_z', 'obj1_rot_x', 'obj1_rot_y',
               'obj1_rot_z', 'obj1_rot_w', 'obj2_x', 'obj2_y', 'obj2_z', 'obj2_rot_x',
               'obj2_rot_y', 'obj2_rot_z', 'obj2_rot_w', 'agent0_x', 'agent0_y',
               'agent0_z', 'agent0_rot_x', 'agent0_rot_y', 'agent0_rot_z',
               'agent0_rot_w', 'agent1_x', 'agent1_y', 'agent1_z', 'agent1_rot_x',
               'agent1_rot_y', 'agent1_rot_z', 'agent1_rot_w']
        
    return data_columns

def load_config(file):
    with open(file) as f:
        config = json.load(f)
    return config


def load_trained_model(model_info, device='cpu', gnn_model=False):
    model_class = model_info['class']
    # load config and initialize model class
    analysis_dir = DEFAULT_VALUES['analysis_dir']
    checkpoint_dir = DEFAULT_VALUES['checkpoint_dir']
    config_file = os.path.join(analysis_dir, 'model_configs/',model_info['config'])
    config = load_config(config_file)
    if gnn_model:
        args = argparse.Namespace()
        for key in config.keys():
            setattr(args, key, config[key])
        # set default values
        setattr(args, 'env', 'tdw')
        setattr(args, 'gt', False)
        setattr(args, 'device', device)
        model = model_class(args)
    else:
        model = model_class(config)
    # load checkpoint weights
    # checkpoints are in folder named after model
    model_dir = os.path.join(checkpoint_dir, 'models', model_info['model_dir'])
    if 'epoch' in model_info:
        model_file_name = os.path.join(model_dir, model_info['model_dir']+'_epoch'+model_info['epoch'])
        model.load_state_dict(torch.load(model_file_name))
        print('Loading model',model_file_name)
    else:
        latest_checkpoint, _ =  get_highest_numbered_file(model_info['model_dir'], model_dir)
        print('Loading from last checkpoint',latest_checkpoint)
        model.load_state_dict(torch.load(latest_checkpoint))

    model.eval()
    model.device = device
    model.to(device)
        
    return model


def get_highest_numbered_file(model_filename, directory):
    highest_number = -1
    highest_numbered_file = None
    for filename in os.listdir(directory):
        if model_filename in filename:
            prefix, number = filename.rsplit('_', 1) if '_epoch' not in filename else filename.rsplit('_epoch', 1)
            if prefix == model_filename and int(number) > highest_number:
                highest_number = int(number)
                highest_numbered_file = os.path.join(directory, filename)
    assert highest_number > -1,'No checkpoints found'
    return highest_numbered_file, highest_number



def load_trained_model(model_info, config_dir, checkpoint_dir, device='cpu'):
    model_class = model_info['class']
    # load config and initialize model class
    config_file = os.path.join(config_dir, model_info['config'])
    config = load_config(config_file)
    model = model_class(config)
    # load checkpoint weights
    # checkpoints are in folder named after model
    model_dir = os.path.join(checkpoint_dir, 'models', model_info['model_dir'])
    if 'epoch' in model_info:
        model_file_name = os.path.join(model_dir, model_info['model_dir']+'_epoch'+model_info['epoch'])
        model.load_state_dict(torch.load(model_file_name))
        print('Loading model',model_file_name)
    else:
        latest_checkpoint, _ =  get_highest_numbered_file(model_info['model_dir'], model_dir)
        print('Loading from last checkpoint',latest_checkpoint)
        model.load_state_dict(torch.load(latest_checkpoint))

    model.eval()
    model.device = device
    model.to(device)
        
    return model

# @TODO plot functions here
# @TODO will need more work depending on the evaluation type
def plot_results(result_file):
    df_plot = pd.read_csv(result_file)
    # Plot the results using seaborn
    fig = plt.figure()
    sns.barplot(x='Model', y='Score', data=df_plot) 
    plt.title('Evaluate Single Goal Events From Forward Rollouts', fontsize=14)
    plt.xlabel('Model Name', fontsize=16) 
    plt.ylabel('Accuracy', fontsize=16)
    plt.ylim([0, 1])
    save_name = os.path.basename(result_file).split('.')[0]
    plt.savefig(f'{save_name}_acc.png', dpi=300, bbox_inches='tight')
    plt.show()

    fig = sns.barplot(x='Model', y='MSE', data=df_plot) 
    plt.title('Prediction Error of Goal Events in Imagination', fontsize=18)
    plt.xlabel('Model Name', fontsize=16) 
    plt.ylabel('MSE', fontsize=16)
    plt.savefig(f'{save_name}_mse.png', dpi=300, bbox_inches='tight')
