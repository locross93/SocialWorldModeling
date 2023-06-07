# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 12:00:58 2023

@author: locro
"""
import os
import json
import torch

from constants import MODEL_DICT_TRAIN

"""Global variables"""
model_dict = MODEL_DICT_TRAIN

def get_data_columns(ds_num):
    if ds_num == 1:
        data_columns = [
            'obj0_x', 'obj0_y', 'obj0_z', 'obj1_x', 'obj1_y', 'obj1_z', 
            'obj2_x', 'obj2_y', 'obj2_z', 'agent0_x', 'agent0_y', 'agent0_z', 'agent0_rot_x',
            'agent0_rot_y', 'agent0_rot_z', 'agent0_rot_w', 'agent1_x', 'agent1_y',
            'agent1_z', 'agent1_rot_x', 'agent1_rot_y', 'agent1_rot_z', 'agent1_rot_w'] 
    elif ds_num == 2:
        data_columns = [
            'obj0_x', 'obj0_y', 'obj0_z', 'obj0_rot_x', 'obj0_rot_y', 'obj0_rot_z', 'obj0_rot_w', 
            'obj1_x', 'obj1_y', 'obj1_z', 'obj1_rot_x', 'obj1_rot_y', 'obj1_rot_z', 'obj1_rot_w', 
            'obj2_x', 'obj2_y', 'obj2_z', 'obj2_rot_x', 'obj2_rot_y', 'obj2_rot_z', 'obj2_rot_w', 
            'agent0_x', 'agent0_y', 'agent0_z', 'agent0_rot_x', 'agent0_rot_y', 'agent0_rot_z', 'agent0_rot_w', 
            'agent1_x', 'agent1_y', 'agent1_z', 'agent1_rot_x', 'agent1_rot_y', 'agent1_rot_z', 'agent1_rot_w']
        
    return data_columns

def load_config(file):
    with open(file) as f:
        config = json.load(f)
    return config


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

def init_model_class(config, args=None):
    gnn_models = ['imma', 'gat', 'rfm']
    model_type = config['model_type']
    model_class = model_dict[model_type]
    if model_type in gnn_models:
        # set config params in args
        for key in config.keys():
            setattr(args, key, config[key])
        model = model_class(args)
    else:
        # init class with config
        model = model_class(config)
    return model

def load_trained_model(model_info, args):
    gnn_models = ['imma', 'gat', 'rfm']
    # load config and initialize model class
    config_file = os.path.join(args.model_config_dir, model_info['config'])
    config = load_config(config_file)
    model_type = config['model_type']
    model_class = model_dict[model_type]
    if model_type in gnn_models:
        # set config params in args
        for key in config.keys():
            setattr(args, key, config[key])
        model = model_class(args)
    else:
        # init class with config
        model = model_class(config)
    # load checkpoint weights
    # checkpoints are in folder named after model
    model_dir = os.path.join(args.checkpoint_dir, 'models', model_info['model_dir'])
    if 'epoch' in model_info:
        model_file_name = os.path.join(model_dir, model_info['model_dir']+'_epoch'+model_info['epoch'])
        model.load_state_dict(torch.load(model_file_name))
        print('Loading model', model_file_name)
    else:
        latest_checkpoint, _ =  get_highest_numbered_file(model_info['model_dir'], model_dir)
        print('Loading from last checkpoint', latest_checkpoint)
        model.load_state_dict(torch.load(latest_checkpoint))

    model.eval()
    model.device = args.device
    model.to(args.device)
    return model

    
def inverse_normalize(normalized_tensor, max_values, min_values, velocity=False):
    ranges = max_values - min_values
    if velocity:
        input_data_pos = normalized_tensor[:, :, ::2]
        input_data_pos = input_data_pos * ranges + min_values
        unnormalized_tensor = normalized_tensor.clone()
        unnormalized_tensor[:, :, ::2] = input_data_pos
    else:
        unnormalized_tensor = normalized_tensor * ranges + min_values
        
    return unnormalized_tensor