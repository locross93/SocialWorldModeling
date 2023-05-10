# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 12:00:58 2023

@author: locro
"""

import os
import pickle
import platform
import torch

if platform.system() == 'Windows':
    # We are running on Windows
    analysis_dir = '/Users/locro/Documents/Stanford/analysis/'
    checkpoint_dir = analysis_dir
elif platform.system() == 'Linux':
    # We are running on Linux
    analysis_dir = '/home/locross/analysis/'
    checkpoint_dir = '/mnt/fs2/locross/analysis/'
    
data_columns = ['obj0_x', 'obj0_y', 'obj0_z', 'obj1_x', 'obj1_y', 'obj1_z', 'obj2_x',
       'obj2_y', 'obj2_z', 'agent0_x', 'agent0_y', 'agent0_z', 'agent0_rot_x',
       'agent0_rot_y', 'agent0_rot_z', 'agent0_rot_w', 'agent1_x', 'agent1_y',
       'agent1_z', 'agent1_rot_x', 'agent1_rot_y', 'agent1_rot_z',
       'agent1_rot_w'] 

def load_trained_model(model_info, key, num_steps, device='cpu'):
    # model parameters
    model_class = model_info[0]
    if type(model_info[1]) == dict:
        kwargs = model_info[1]
        model = model_class(**kwargs)
    elif model_info[1] == 'config':
        # load config
        config_file = analysis_dir+'models/configs/'+model_info[-3]
        with open(config_file, 'rb') as f:
            config = pickle.load(f)
        model = model_class(config)
    else:
        input_size = model_info[1]
        hidden_size = model_info[2]
        latent_size = 50
        model = model_class(input_size, hidden_size, latent_size)
    
    model_file_name = 'models/'+model_info[3]
    if os.path.exists(model_file_name) and not os.path.isdir(analysis_dir+'models/'+model_info[2]):
        model.load_state_dict(torch.load(model_file_name))
    elif os.path.isdir(checkpoint_dir+'models/'+model_info[2]):
        # checkpoints are in folder named after model
        model_dir = checkpoint_dir+'models/'+model_info[2]+'/'
        model_file_name = model_dir+model_info[3]
        if os.path.exists(model_file_name):
            model.load_state_dict(torch.load(model_file_name))
        else:
            latest_checkpoint, _ =  get_highest_numbered_file(model_info[3], [model_dir])
            print('Loading from last checkpoint',latest_checkpoint)
            model.load_state_dict(torch.load(latest_checkpoint))
    else:
        latest_checkpoint, _ =  get_highest_numbered_file(model_info[3], [checkpoint_dir+'models/'])
        print('Loading from last checkpoint',latest_checkpoint)
        model.load_state_dict(torch.load(latest_checkpoint))
    model.eval()
    model.device = device
    model.to(device)
    if key[:4] == 'lstm':
        model.batch_size = 1
        model.num_steps = num_steps
        
    return model


def get_highest_numbered_file(model_filename, directories):
    highest_number = -1
    highest_numbered_file = None
    for directory in directories:
        for filename in os.listdir(directory):
            if model_filename in filename:
                prefix, number = filename.rsplit('_', 1) if '_epoch' not in filename else filename.rsplit('_epoch', 1)
                if prefix == model_filename and int(number) > highest_number:
                    highest_number = int(number)
                    highest_numbered_file = directory+filename
    assert highest_number > -1,'No checkpoints found'
    return highest_numbered_file, highest_number