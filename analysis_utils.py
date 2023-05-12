# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 12:00:58 2023

@author: locro
"""

import json
import os
import pickle
import platform
import torch

if platform.system() == 'Windows':
    # We are running on Windows
    analysis_dir = '/Users/locro/Documents/Stanford/SocialWorldModeling/'
    checkpoint_dir = analysis_dir
elif platform.system() == 'Linux':
    # We are running on Linux
    analysis_dir = '/home/locross/SocialWorldModeling/'
    checkpoint_dir = '/mnt/fs2/locross/analysis/'
    
data_columns = ['obj0_x', 'obj0_y', 'obj0_z', 'obj1_x', 'obj1_y', 'obj1_z', 'obj2_x',
       'obj2_y', 'obj2_z', 'agent0_x', 'agent0_y', 'agent0_z', 'agent0_rot_x',
       'agent0_rot_y', 'agent0_rot_z', 'agent0_rot_w', 'agent1_x', 'agent1_y',
       'agent1_z', 'agent1_rot_x', 'agent1_rot_y', 'agent1_rot_z',
       'agent1_rot_w'] 

def load_config(file):
    with open(file) as f:
        config = json.load(f)
    return config


def load_trained_model(model_info, device='cpu'):
    model_class = model_info['class']
    # load config and initialize model class
    config_file = os.path.join(analysis_dir, 'models/configs/',model_info['config'])
    config = load_config(config_file)
    model = model_class(config)
    # load checkpoint weights
    # checkpoints are in folder named after model
    model_dir = os.path.join(checkpoint_dir, 'models', model_info['model_dir'])
    model_file_name = os.path.join(model_dir, model_info['model_dir']+'_epoch'+model_info['epoch'])
    if os.path.exists(model_file_name):
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
                highest_numbered_file = directory+filename
    assert highest_number > -1,'No checkpoints found'
    return highest_numbered_file, highest_number