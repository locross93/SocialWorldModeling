# -*- coding: utf-8 -*-
"""
Created on Thu May 11 13:43:20 2023

@author: locro
"""

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
    

if __name__ == "__main__":
    # train test splits
    data_file = data_dir+'train_test_splits_3D_dataset.pkl'
    with open(data_file, 'rb') as f:
        loaded_dataset = pickle.load(f)
    train_dataset, test_dataset = loaded_dataset
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_or_val = 'val'
    
    if train_or_val == 'train':
        input_data = train_dataset.dataset.tensors[0][train_dataset.indices,:,:]
    elif train_or_val == 'val':
        input_data = test_dataset.dataset.tensors[0][test_dataset.indices,:,:]
    
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
        model = load_trained_model(model_info, key, input_data.size(1), DEVICE)
        
        # put on gpu
        input_data = input_data.to(DEVICE)
        
        pickup_timepoints = annotate_pickup_timepoints(train_or_val, pickup_or_move='move')
        goal_timepoints = annotate_goal_timepoints(train_or_val)