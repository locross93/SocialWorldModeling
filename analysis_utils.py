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
    event_models = ['event_model']
    model_type = config['model_type']
    model_class = model_dict[model_type]
    if model_type in gnn_models:
        # set config params in args
        for key in config.keys():
            setattr(args, key, config[key])
        model = model_class(args)
    elif model_type in event_models:
        # make new config with keys that start with mp_ and ep_
        mp_config = {}
        ep_config = {}
        for key in config.keys():
            if key.startswith('mp_'):
                # make new key without mp_
                new_key = key[3:]
                mp_config[new_key] = config[key]
            elif key.startswith('ep_'):
                # make new key without ep_
                new_key = key[3:]
                ep_config[new_key] = config[key]
        model = model_class(ep_config, mp_config)
    else:
        # init class with config
        model = model_class(config)
    return model

def load_trained_model(model_info, args):
    event_models = ['event_model']
    # load config and initialize model class
    config_file = os.path.join(args.model_config_dir, model_info['config'])
    config = load_config(config_file)
    model_type = config['model_type']
    model = init_model_class(config, args)

    if model_type in event_models:
        # load weights
        if 'epoch' in model_info:
            ep_weights_path = os.path.join(args.checkpoint_dir, 'models', model_info['model_dir'], 'ep_model_epoch'+model_info['epoch'])
            mp_weights_path = os.path.join(args.checkpoint_dir, 'models', model_info['model_dir'], 'mp_model_epoch'+model_info['epoch'])
            model.load_weights(ep_weights_path, mp_weights_path)
        else:
            ep_weights_path, _ = get_highest_numbered_file('ep_model', os.path.join(args.checkpoint_dir, 'models', model_info['model_dir']))
            mp_weights_path, _ = get_highest_numbered_file('mp_model', os.path.join(args.checkpoint_dir, 'models', model_info['model_dir']))
            print('Loading from last checkpoint', mp_weights_path)
            model.load_weights(ep_weights_path, mp_weights_path)
        model.mp_model.device = args.device
        model.ep_model.device = args.device
        model.device = args.device
        model.mp_model.to(args.device)
        model.ep_model.to(args.device)
        model.mp_model.eval()
        model.ep_model.eval()
    else:
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