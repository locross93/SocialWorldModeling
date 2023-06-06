# -*- coding: utf-8 -*-
"""
Created on Tue May  9 14:06:24 2023

@author: locro
"""

# example usage: python train_world_model.py  --config rssm_disc_default_config.json --dec_hidden_size 512
# python train_world_model.py  --config multistep_predictor_default_config.json --mlp_hidden_size 512

import os
import json
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from analysis_utils import init_model_class
from constants import DEFAULT_VALUES, MODEL_DICT_TRAIN
#from models import ReplayBuffer
# temp
from models import ReplayBufferEarly as ReplayBuffer


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
    parser.add_argument('--epochs', type=int, action='store',
                        default=DEFAULT_VALUES['epochs'], 
                        help='Epochs')
    parser.add_argument('--save_every', type=int, action='store',
                        default=DEFAULT_VALUES['save_every'], 
                        help='Epoch Save Interval')
    parser.add_argument('--input_size', type=int, help='Input size')
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

    if config['model_type'] == "sgnet_cvae":
        config['enc_steps'] = args.burn_in_length
        config['dec_steps'] = args.rollout_length
    elif config['model_type'] == "agent_former":        
        config['past_frames'] = args.burn_in_length
        config['future_frames'] = args.rollout_length

    # model_class = model_dict[config['model_type']]        
    # model = model_class(config)
    
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
    batch_size = args.batch_size
    print(f'Batch size: {batch_size}')
    batches_per_epoch = np.min([replay_buffer.buffer_size // batch_size, 50])
    
    # make validation data
    val_buffer = ReplayBuffer(sequence_length)
    val_buffer.upload_training_set(val_data)
    seed = 100 # set seed so every model sees the same randomization
    val_batch_size = np.min([val_data.size(0), 1000])
    val_trajs = val_buffer.sample(val_batch_size, random_seed=seed)
    val_trajs = val_trajs.to(DEVICE)
    val_trajs = val_trajs.to(torch.float32)
        
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
    loss_dict['val_mse'] = []
    loss_dict['epoch_times'] = []

    if config['model_type'] == 'dreamerv2':
        loss_dict['recon_loss'] = []
        loss_dict['kl_loss'] = []
    elif config['model_type'] == 'sgnet_cvae':
        loss_dict['cvae_loss'] = []
        loss_dict['kld_loss'] = []
        loss_dict['goal_loss'] = []
    
    model.to(DEVICE)
    start_time = time.time()
    for epoch in tqdm(range(args.epochs)):
        model.train()
        batch_loss = []
        if config['model_type'] == 'dreamerv2':
            batch_recon_loss = []
            batch_kl_loss = []
        elif config['model_type'] == 'sgnet_cvae':
            batch_cvae_loss = []
            batch_kld_loss = []
            batch_goal_loss = []
        nsamples = 0
        for i in tqdm(range(batches_per_epoch)):
            batch_x = replay_buffer.sample(batch_size)
            batch_x = batch_x.to(torch.float32)
            batch_x = batch_x.to(DEVICE)
            nsamples += batch_x.shape[0]
            opt.zero_grad()
            if batch_x.dtype == torch.int64:
                batch_x = batch_x.float()
            if config['model_type'][:4] == 'rssm' or \
                config['model_type'] in ['dreamerv2', 'transformer_wm', 'transformer_iris', 'transformer_iris_low_dropout']:
                loss = model.loss(batch_x)
            elif config['model_type'] in ['sgnet_cvae']:
                loss, sgnet_loss_dict = model.loss(batch_x)
                batch_cvae_loss.append(sgnet_loss_dict['cvae_loss'])
                batch_goal_loss.append(sgnet_loss_dict['goal_loss'])
                batch_kld_loss.append(sgnet_loss_dict['kld_loss'])
            elif config['model_type'] in ['multistep_predictor', 'multistep_delta']:
                loss = model.loss(batch_x, burn_in_length, rollout_length)
            elif config['model_type'] == 'transformer_mp':
                loss = model.loss(batch_x, burn_in_length, rollout_length, mask_type='triangular')
            loss.backward()
            opt.step()
            
            batch_loss.append(loss.item())
            if config['model_type'] == 'dreamerv2':
                batch_recon_loss.append(model.recon_loss.item())
                batch_kl_loss.append(model.kl.item())
        epoch_loss = np.mean(batch_loss) # TO DO IS THIS REALLY MSE?
        loss_dict['train'].append(epoch_loss)
        loss_dict['epoch_times'].append(time.time()-start_time)

        # loss multiple losses
        if config['model_type'] == 'dreamerv2':
            loss_dict['recon_loss'].append(np.sum(batch_recon_loss))
            loss_dict['kl_loss'].append(np.sum(batch_kl_loss))
        elif config['model_type'] == 'sgnet_cvae':
            loss_dict['cvae_loss'].append(np.mean(batch_cvae_loss))
            loss_dict['goal_loss'].append(np.mean(batch_goal_loss))
            loss_dict['kld_loss'].append(np.mean(batch_kld_loss))
            
        # test on validation set
        with torch.no_grad():
            model.eval()
            if config['model_type'][:4] == 'rssm' or \
                config['model_type'] in ['dreamerv2', 'transformer_wm', 'transformer_iris', 'transformer_iris_low_dropout', 
                                         'agent_former']:
                val_loss = model.loss(val_trajs)
            elif config['model_type'] in ['sgnet_cvae']:
                val_loss, _ = model.loss(val_trajs)
            elif config['model_type'] in ['multistep_predictor', 'multistep_delta']:
                val_loss = model.loss(val_trajs, burn_in_length, rollout_length)
            elif config['model_type'] == 'transformer_mp':
                val_loss = model.loss(val_trajs, burn_in_length, rollout_length, mask_type='triangular')

            val_loss = val_loss.item()
            loss_dict['val'].append(val_loss)
            # get MSE on validation data
            if config['model_type'] == 'dreamerv2':
                val_mse = model.recon_loss.item() / val_trajs[:,-rollout_length:,:].numel()
            else:
                val_mse = val_loss / val_trajs[:,-rollout_length:,:].numel()
            loss_dict['val_mse'].append(val_mse)
        # log to tensorboard
        writer.add_scalar('Train Loss/loss', epoch_loss, epoch)
        writer.add_scalar('Val Loss/val', val_loss, epoch)
        writer.add_scalar('Val Loss/val_mse', val_mse, epoch)
        if config['model_type'] == 'dreamerv2':
            writer.add_scalar('Train_Loss/recon_loss', np.mean(batch_recon_loss), epoch)
            writer.add_scalar('Train_Loss/kl_loss', np.mean(batch_kl_loss), epoch)
        elif config['model_type'] == 'sgnet_cvae':
            writer.add_scalar('Train_Loss/cvae_loss', np.mean(batch_cvae_loss), epoch)
            writer.add_scalar('Train_Loss/goal_loss', np.mean(batch_goal_loss), epoch)
            writer.add_scalar('Train_Loss/kld_loss', np.mean(batch_kld_loss), epoch)
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
        print(f'Epoch {epoch}, Train Loss {epoch_loss}, Validation MSE {val_mse}')


if __name__ == '__main__':    
    main()

