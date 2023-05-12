# -*- coding: utf-8 -*-
"""
Created on Tue May  9 14:06:24 2023

@author: locro
"""

# example usage: python train_world_model.py --model dreamerv2 --config rssm_disc_default_config.json --dec_hidden_size 512
# python train_world_model.py --model multistep_predictor --config multistep_predictor_default_config.json --mlp_hidden_size 512

import os
import json
import time
import pickle
import platform
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import DreamerV2, MultistepPredictor, ReplayBuffer


"""Global variables"""
model_dict = {
    'dreamerv2': DreamerV2,
    'multistep_predictor': MultistepPredictor
    }
DEFAULT_VALUES = {
    'analysis_dir': './',
    'data_dir': '/data2/ziyxiang/social_world_model/data',
    'checkpoint_dir': '/data2/ziyxiang/social_world_model/checkpoint',
}


def load_args():
    parser = argparse.ArgumentParser()
    # general pipeline parameters
    parser.add_argument('--analysis_dir', type=str, default=DEFAULT_VALUES['analysis_dir'], help='Analysis directory')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_VALUES['data_dir'], help='Data directory')
    parser.add_argument('--checkpoint_dir', type=str, default=DEFAULT_VALUES['checkpoint_dir'], help='Checkpoint directory')
    # general training parameters
    parser.add_argument('--model', type=str, required=True, help='Model to use for training')
    parser.add_argument('--config', type=str, required=True, help='Config JSON file')
    parser.add_argument('--model_filename', type=str, default=None, help='Filename for saving model')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--epochs', type=int, default=int(1e5), help='Epochs')
    parser.add_argument('--save_every', type=int, default=500, help='Epoch Save Interval')
    # rssm parameters
    parser.add_argument('--deter_size', type=int, help='Deterministic size')
    parser.add_argument('--dec_hidden_size', type=int, help='Decoder hidden size')
    parser.add_argument('--rssm_type', type=str, help='RSSM type')
    parser.add_argument('--rnn_type', type=str, help='RNN type')
    parser.add_argument('--category_size', type=int, help='Category size')
    parser.add_argument('--class_size', type=int, help='Class size')
    # multistep predictor parameters
    parser.add_argument('--mlp_hidden_size', type=int, help='MLP hidden size')
    parser.add_argument('--num_mlp_layers', type=int, help='Number of MLP layers')
    parser.add_argument('--rnn_hidden_size', type=int, help='RNN hidden size')
    parser.add_argument('--num_rnn_layers', type=int, help='Number of RNN layers')
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

def main():
    args = load_args()
    config = load_config(os.path.join(args.analysis_dir, 'models/configs/', args.config))
    
    # Update config with args and keep track of overridden parameters
    overridden_parameters = []
    for k, v in vars(args).items():
        if v is not None and k in config:
            config[k] = v
            overridden_parameters.append(f"{k}_{v}")
            print(f"{k}_{v}")
    
    model_class = model_dict[args.model]
    model = model_class(config)
    
    # filename for saving
    if args.model_filename is None:
        model_filename = config['model_type']
    else:
        model_filename = args.model_filename
    
    # If parameters overwritte, save updated config to new file
    if len(overridden_parameters) > 0:
        new_config_filename = os.path.join(args.analysis_dir, 'models/configs/', f"{model_filename}_{'_'.join(overridden_parameters)}.json")
        save_config(config, new_config_filename)
    
    # Check that we are using GPU
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(DEVICE)

    # load data
    data_file = os.path.join(args.data_dir, 'train_test_splits_3D_dataset.pkl')
    with open(data_file, 'rb') as f:
        loaded_dataset = pickle.load(f)
    train_dataset, test_dataset = loaded_dataset
    train_data = train_dataset.dataset.tensors[0][train_dataset.indices,:,:]
    val_data = test_dataset.dataset.tensors[0][test_dataset.indices,:,:]
    
    # initialize the replay buffer
    burn_in_length = args.burn_in_length
    rollout_length = args.rollout_length
    sequence_length = burn_in_length + rollout_length
    replay_buffer = ReplayBuffer(sequence_length)
    replay_buffer.upload_training_set(train_data)
    if platform.system() == 'Windows':
        batch_size = 64
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
        
    print('Starting', model_filename)
    
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
            batch_x = batch_x.to(DEVICE)
            nsamples += batch_x.shape[0]
            opt.zero_grad()
            if config['model_type'][:4] == 'rssm':
                loss = model.loss(batch_x)
            elif config['model_type'] == 'multistep_predictor':
                loss = model.loss(batch_x, burn_in_length, rollout_length)
            loss.backward()
            opt.step()
            
            batch_loss.append(loss.item())
            if config['model_type'][:4] == 'rssm':
                batch_recon_loss.append(model.recon_loss.item())
                batch_kl_loss.append(model.kl.item())
        epoch_loss = np.sum(batch_loss)
        loss_dict['train'].append(epoch_loss)
        loss_dict['epoch_times'].append(time.time()-start_time)
        if config['model_type'][:4] == 'rssm':
            loss_dict['recon_loss'].append(np.sum(batch_recon_loss))
            loss_dict['kl_loss'].append(np.sum(batch_kl_loss))
        # test on validation set
        with torch.no_grad():
            model.eval()
            if config['model_type'][:4] == 'rssm':
                val_loss = model.loss(val_trajs)
            elif config['model_type'] == 'multistep_predictor':
                val_loss = model.loss(val_trajs, burn_in_length, rollout_length)
            val_loss = val_loss.item()
            loss_dict['val'].append(val_loss)
        # log to tensorboard
        writer.add_scalar('Train Loss/loss', epoch_loss, epoch)
        writer.add_scalar('Val Loss/val', val_loss, epoch)
        if config['model_type'][:4] == 'rssm':
            writer.add_scalar('Train Loss/recon_loss', np.sum(batch_recon_loss), epoch)
            writer.add_scalar('Train Loss/kl_loss', np.sum(batch_kl_loss), epoch)
        if epoch % args.save_every == 0 or epoch == (args.epochs-1):
            # save checkpoints in checkpoint directory with plenty of storage
            save_dir = os.path.join(args.checkpoint_dir, 'models/', model_filename)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model_name = os.path.join(save_dir, f'{model_filename}_epoch{epoch}')
            torch.save(model.state_dict(), model_name)
        print(f'Epoch {epoch}, Train Loss {epoch_loss}, Validation Loss {val_loss}')

if __name__ == '__main__':    
    main()

