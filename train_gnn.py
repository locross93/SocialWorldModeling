# -*- coding: utf-8 -*-
"""
Created on Thu May 25 18:38:01 2023

@author: locro
"""

# example usage: python train_gnn.py --model imma --config imma_default_config.json

import os
import json
import time
import pickle
import platform
import argparse
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from constants_lc import DEFAULT_VALUES, MODEL_DICT_TRAIN
from models import ReplayBuffer

"""Global variables"""
model_dict = MODEL_DICT_TRAIN

def load_args():
    parser = argparse.ArgumentParser()
    # general pipeline parameters
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
                         default='train_test_splits_3D_dataset.pkl', 
                         help='Data directory')
    parser.add_argument('--checkpoint_dir', type=str, 
                        default=DEFAULT_VALUES['checkpoint_dir'], 
                        help='Checkpoint directory')
    # general training parameters
    parser.add_argument('--model', type=str, required=True, help='Model to use for training')
    parser.add_argument('--config', type=str, required=True, help='Config JSON file')
    parser.add_argument('--model_filename', type=str, default=None, help='Filename for saving model')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning Rate')
    parser.add_argument('--epochs', type=int, default=int(1e5), help='Epochs')
    parser.add_argument('--save_every', type=int, default=500, help='Epoch Save Interval')
    parser.add_argument('--batch_size', type=int, help='Epoch Save Interval')
    parser.add_argument('--burn_in_length', type=int, default=50, help='Amount of frames to burn into RNN')
    parser.add_argument('--rollout_length', type=int, default=30, help='Forward rollout length')  

    return parser.parse_args()
    

def load_config(file):
    with open(file) as f:
        config = json.load(f)
    return config

def save_config(config, filename):
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)

if __name__ == '__main__':    
    args = load_args()
    config = load_config(os.path.join(args.model_config_dir, args.config))
    
    for key in config.keys():
        setattr(args, key, config[key])
    
    # Update config with args and keep track of overridden parameters
    overridden_parameters = []
    for k, v in vars(args).items():
        if v is not None and k in config and config[k] != v:
            config[k] = v
            overridden_parameters.append(f"{k}_{v}")
            print(f"{k}_{v}")
            
    # Check that we are using GPU
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(DEVICE)
    setattr(args, 'device', DEVICE)
    
    model_class = model_dict[args.model]
    model = model_class(args)
    
    # filename for saving
    if args.model_filename is None:
        model_filename = config['model_type']
    else:
        model_filename = args.model_filename
    
    # If parameters overwritte, save updated config to new file
    if len(overridden_parameters) > 0:
        new_config_filename = os.path.join(args.model_config_dir, f"{model_filename}_{'_'.join(overridden_parameters)}.json")
        save_config(config, new_config_filename)

    # load data
    data_file = os.path.join(args.data_dir, args.dataset)
    loaded_dataset = pickle.load(open(data_file, 'rb'))
    train_dataset, test_dataset = loaded_dataset
    train_data = train_dataset.dataset.tensors[0][train_dataset.indices,:,:]
    val_data = test_dataset.dataset.tensors[0][test_dataset.indices,:,:]
    
    # initialize the replay buffer
    burn_in_length = args.burn_in_length
    rollout_length = args.rollout_length
    sequence_length = burn_in_length + rollout_length
    replay_buffer = ReplayBuffer(sequence_length)
    replay_buffer.upload_training_set(train_data)
    if args.batch_size:
        batch_size = args.batch_size
    if platform.system() == 'Windows':
        batch_size = 32
    elif platform.system() == 'Linux':
        batch_size = 256
    print(f'Batch size: {batch_size}')
    batches_per_epoch = replay_buffer.buffer_size // batch_size
    
    # make validation data
    val_buffer = ReplayBuffer(sequence_length)
    val_buffer.upload_training_set(val_data)
    seed = 100 # set seed so every model sees the same randomization
    val_batch_size = val_data.size(0)
    val_trajs = val_buffer.sample(val_batch_size, random_seed=seed)
    val_trajs = val_trajs.to(DEVICE)
    val_trajs = val_trajs.to(torch.float32)
    val_trajs = val_trajs.reshape(-1, sequence_length, 5, 7)
        
    print('Starting', model_filename, 'On DS', args.dataset)
    
    # log to tensorboard
    log_dir = os.path.join(args.checkpoint_dir, 'models/training_info/tensorboard/', model_filename)
    writer = SummaryWriter(log_dir=log_dir)
    
    # optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # store losses in dictionary (will vary by model type)
    loss_dict = {}
    loss_dict['train'] = []
    loss_dict['val'] = []
    loss_dict['epoch_times'] = []
    if config['model_type'][:4] == 'rssm':
        loss_dict['recon_loss'] = []
        loss_dict['kl_loss'] = []
    
    model.to(DEVICE)
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        batch_loss = []
        if config['model_type'][:4] == 'rssm':
            batch_recon_loss = []
            batch_kl_loss = []
        nsamples = 0
        for i in range(batches_per_epoch):
            batch_x = replay_buffer.sample(batch_size)
            batch_x = batch_x.to(torch.float32)
            batch_x = batch_x.to(DEVICE)
            batch_x = batch_x.reshape(-1, sequence_length, 5, 7)
            nsamples += batch_x.shape[0]
            opt.zero_grad()
            loss = model.loss(batch_x, burn_in_length, rollout_length)
            loss.backward()
            opt.step()
            
            batch_loss.append(loss.item())
        epoch_loss = np.sum(batch_loss)
        loss_dict['train'].append(epoch_loss)
        loss_dict['epoch_times'].append(time.time()-start_time)
        # test on validation set
        with torch.no_grad():
            model.eval()
            val_loss = model.loss(val_trajs, burn_in_length, rollout_length)
            val_loss = val_loss.item()
            loss_dict['val'].append(val_loss)
        # log to tensorboard
        writer.add_scalar('Train Loss/loss', epoch_loss, epoch)
        writer.add_scalar('Val Loss/val', val_loss, epoch)
        if epoch % args.save_every == 0 or epoch == (args.epochs-1):
            # save checkpoints in checkpoint directory with plenty of storage
            save_dir = os.path.join(args.checkpoint_dir, 'models', model_filename)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model_name = os.path.join(save_dir, f'{model_filename}_epoch{epoch}')
            torch.save(model.state_dict(), model_name)
        print(f'Epoch {epoch}, Train Loss {epoch_loss}, Validation Loss {val_loss}')

