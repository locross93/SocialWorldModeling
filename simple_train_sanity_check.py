# -*- coding: utf-8 -*-
"""
Created on Tue May  9 14:06:24 2023

@author: locro
"""

# example usage: python train_world_model.py --model dreamerv2 --config rssm_disc_default_config.json --dec_hidden_size 512
# python train_world_model.py --model multistep_predictor --config multistep_predictor_default_config.json --mlp_hidden_size 512
# python train_world_model.py --model transformer_wm --config transformer_wm_default_config.json

import os
import json
import time
import pickle
import argparse
import numpy as np
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
                         default='dataset_5_25_23.pkl', 
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
    parser.add_argument('--epochs', type=int, action='store',
                        default=DEFAULT_VALUES['epochs'], 
                        help='Epochs')
    parser.add_argument('--save_every', type=int, action='store',
                        default=DEFAULT_VALUES['save_every'], 
                        help='Epoch Save Interval')
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
    args.device = DEVICE
    print(f'Using device: {DEVICE}')
    
    model = init_model_class(config, args)
    
    # filename same as cofnig makes it easier for identifying different parameters
    if args.model_filename is None:
        model_filename = '_'.join(args.config.split('.')[0].split('_')[:-1])       
    else:
        model_filename = args.model_filename
    print(f"model will be saved to {model_filename}")
    
    # If parameters overwritte, save updated config to new file
    if len(overridden_parameters) > 0:
        new_config_filename = os.path.join(args.model_config_dir, f"{model_filename}_{'_'.join(overridden_parameters)}.json")
        save_config(config, new_config_filename)
    
    # Check that we are using GPU
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {DEVICE}')

    # load data    
    dataset_file = os.path.join(args.data_dir, args.dataset)
    loaded_dataset = pickle.load(open(dataset_file, 'rb'))
    train_dataset, test_dataset = loaded_dataset
    train_data = train_dataset.dataset.tensors[0][train_dataset.indices,:,:]
    val_data = test_dataset.dataset.tensors[0][test_dataset.indices,:,:]

    # initialize the replay buffer
    burn_in_length = args.burn_in_length
    rollout_length = args.rollout_length
    sequence_length = burn_in_length + rollout_length
    replay_buffer = ReplayBuffer(sequence_length)
    replay_buffer.upload_training_set(train_data)
    #breakpoint()    
    batch_size = args.batch_size
    
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
    if config['model_type'] == 'dreamerv2':
        loss_dict['recon_loss'] = []
        loss_dict['kl_loss'] = []
    
    model.to(DEVICE)
    start_time = time.time()
    batch_x = replay_buffer.sample(batch_size)
    batch_x = batch_x.to(DEVICE)

    for epoch in tqdm(range(args.epochs)):
        model.train()
        batch_loss = []
        if config['model_type'] == 'dreamerv2':
            batch_recon_loss = []
            batch_kl_loss = []            
            opt.zero_grad()
        if batch_x.dtype == torch.int64:
            batch_x = batch_x.float()
        if config['model_type'] == 'dreamerv2' or \
            config['model_type'] in ['transformer_wm', 'transformer_iris', 'transformer_iris_low_dropout']:
            loss = model.loss(batch_x)
        elif config['model_type'] == 'transformer_mp':
            loss = model.loss(batch_x, burn_in_length, rollout_length, mask_type='triangular')
        else:
            loss = model.loss(batch_x, burn_in_length, rollout_length)
        loss.backward()
        opt.step()    
        batch_loss.append(loss.item())
        if config['model_type'] == 'dreamerv2':
            batch_recon_loss.append(model.recon_loss.item())
            batch_kl_loss.append(model.kl.item())        
        epoch_loss = np.sum(batch_loss)
        loss_dict['train'].append(epoch_loss)
        loss_dict['epoch_times'].append(time.time()-start_time)
        if config['model_type'] == 'dreamerv2':
            loss_dict['recon_loss'].append(np.sum(batch_recon_loss))
            loss_dict['kl_loss'].append(np.sum(batch_kl_loss))
        elif config['model_type'] in ['transformer_wm', 'transformer_iris', 'transformer_iris_low_dropout']:
            loss_dict
        
        # log to tensorboard
        writer.add_scalar('Train Loss/loss', epoch_loss, epoch)
        if config['model_type'] == 'dreamerv2':
            writer.add_scalar('Train_Loss/recon_loss', np.sum(batch_recon_loss), epoch)
            writer.add_scalar('Train_Loss/kl_loss', np.sum(batch_kl_loss), epoch)
        elif config['model_type'] in ['transformer_wm', 'transformer_iris', 'transformer_iris_low_dropout']:
            #writer.add_embedding(model.pos_embds, global_step=epoch, tag='pos_embds')
            one_sample_time_embds = model.embds[0,:,:]
            #writer.add_embedding(one_sample_time_embds, global_step=epoch, tag='one_sample_time_embds')
        if epoch % args.save_every == 0 or epoch == (args.epochs-1):
            # save checkpoints in checkpoint directory with plenty of storage
            save_dir = os.path.join(args.checkpoint_dir, 'models', model_filename)            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model_name = os.path.join(save_dir, f'{model_filename}_epoch{epoch}')
            torch.save(model.state_dict(), model_name)
        print(f'Epoch {epoch}, Train Loss {epoch_loss}')




if __name__ == '__main__':    
    main()
