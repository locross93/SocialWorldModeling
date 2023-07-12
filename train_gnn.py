# -*- coding: utf-8 -*-
"""
Created on Thu May 25 18:38:01 2023

@author: locro
"""

# example usage: python train_gnn.py --config imma_encoder_rnn.json --model_filename imma_rnn_enc

import os
import json
import time
import pickle
import platform
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from analysis_utils import init_model_class
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
                         default=DEFAULT_VALUES['dataset'], 
                         help='Dataset name')
    parser.add_argument('--checkpoint_dir', type=str, 
                        default=DEFAULT_VALUES['checkpoint_dir'], 
                        help='Checkpoint directory')
    # general training parameters
    parser.add_argument('--config', type=str,
                        required=True, help='Config JSON file')
    parser.add_argument('--batch_size', type=int, action='store',
                        default=DEFAULT_VALUES['batch_size'],
                        help='Batch size')
    parser.add_argument('--model_filename', type=str,
                        default=None, 
                        help='Filename for saving model')
    parser.add_argument('--lr', type=float, action='store',
                        default=DEFAULT_VALUES['lr'], 
                        help='Learning Rate')
    parser.add_argument('--lr_scheduler', type=int, action='store',
                        default=1, help='Learning Rate Scheduling')
    parser.add_argument('--epochs', type=int, action='store',
                        default=DEFAULT_VALUES['epochs'], 
                        help='Epochs')
    parser.add_argument('--save_every', type=int, action='store',
                        default=DEFAULT_VALUES['save_every'], 
                        help='Epoch Save Interval')
    parser.add_argument('--input_size', type=int, help='Input size')
    parser.add_argument('--env', type=str, default='tdw', help='Environment')
    parser.add_argument('--gt', default=False, action='store_true', help='use ground truth graph')    
    parser.add_argument('--rollout_length', type=int, default=30, help='Forward rollout length') 
    # model parameters
    parser.add_argument('--obs_frames', type=int, help='Number of observation frames')
    parser.add_argument('--num_humans', type=int, default=5, help='Number of humans')
    parser.add_argument('--feat_dim', type=int, default=7, help='Feature dimension')
    parser.add_argument('--skip_first', type=int, default=0, help='Skip the first frame')
    parser.add_argument('--edge_types', type=int, default=2, help='Number of edge types')
    parser.add_argument('--hidden_dim', type=int, help='Hidden dimension')
    parser.add_argument('--plt', type=int, default=0, help='Progressive Layer Training')
    parser.add_argument('--burn_in', type=int, default=0, help='Burn-in flag')
    parser.add_argument('--encoder', choices=['mlp', 'rnn'])

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
    
    model = init_model_class(config, args)
    
    # filename same as config makes it easier for identifying different parameters
    if args.model_filename is None:
        model_filename = '_'.join(args.config.split('.')[0].split('_')[:-1])
    else:
        model_filename = args.model_filename
    
    # If parameters overwritte, save updated config to new file
    if len(overridden_parameters) > 0:
        new_config_filename = os.path.join(args.model_config_dir, f"{model_filename}_{'_'.join(overridden_parameters)}.json")
        save_config(config, new_config_filename)

    # load data    
    dataset_file = os.path.join(args.data_dir, args.dataset)
    loaded_dataset = pickle.load(open(dataset_file, 'rb'))
    train_dataset, test_dataset = loaded_dataset
    train_data = train_dataset.dataset.tensors[0][train_dataset.indices,:,:]
    val_data = test_dataset.dataset.tensors[0][test_dataset.indices,:,:]
    
    # initialize the replay buffer
    burn_in_length = args.obs_frames
    rollout_length = args.rollout_length
    sequence_length = burn_in_length + rollout_length
    replay_buffer = ReplayBuffer(sequence_length)
    replay_buffer.upload_training_set(train_data)    
    print(f'Batch size: {args.batch_size}')
    batches_per_epoch = np.min([replay_buffer.buffer_size // args.batch_size, 50])
    
    # make validation data
    val_buffer = ReplayBuffer(sequence_length)
    val_buffer.upload_training_set(val_data)
    seed = 100 # set seed so every model sees the same randomization
    val_batch_size = np.min([val_data.size(0), 1000])
    val_trajs = val_buffer.sample(val_batch_size, random_seed=seed)
    val_trajs = val_trajs.to(DEVICE)
        
    print('Starting', model_filename, 'On DS', args.dataset)
    
    # log to tensorboard
    log_dir = os.path.join(args.checkpoint_dir, 'models/training_info/tensorboard/', model_filename)
    writer = SummaryWriter(log_dir=log_dir)
    
    # optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)
    
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
    for epoch in tqdm(range(args.epochs)):
        model.train()
        batch_loss = []
        if config['model_type'][:4] == 'rssm':
            batch_recon_loss = []
            batch_kl_loss = []
        nsamples = 0
        for i in tqdm(range(batches_per_epoch)):
            batch_x = replay_buffer.sample(args.batch_size)
            batch_x = batch_x.to(DEVICE)
            nsamples += batch_x.shape[0]
            opt.zero_grad()
            loss = model.loss(batch_x, burn_in_length, rollout_length)
            loss.backward()
            opt.step()
            
            batch_loss.append(loss.item())
        epoch_loss = np.mean(batch_loss)
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
            # save training history in dataframe
            training_info = {
                'Epochs': np.arange(len(loss_dict['train']))}
            for key in loss_dict:
                training_info[key] = loss_dict[key]
            df_training = pd.DataFrame.from_dict(training_info)
            df_training.to_csv(os.path.join(save_dir, f'training_info_{model_filename}.csv'))
        if args.lr_scheduler:
            scheduler.step()
        print(f'Epoch {epoch}, Train Loss {epoch_loss}, Validation Loss {val_loss}')

