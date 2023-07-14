# example usage: python train_world_model.py --config rssm_disc_default_config.json --dec_hidden_size 512
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
from constants_lc import DEFAULT_VALUES, MODEL_DICT_TRAIN
from models import ReplayBufferEvents


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
    parser.add_argument('--train_seed', type=int, 
                        default=DEFAULT_VALUES['train_seed'], help='Random seed')
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
    # model parameters
    parser.add_argument('--mp_rnn_hidden_size', type=int, help='MP RNN hidden size')
    parser.add_argument('--mp_mlp_hidden_size', type=int, help='MP MLP hidden size')
    parser.add_argument('--ep_rnn_hidden_size', type=int, help='EP RNN hidden size')
    parser.add_argument('--ep_mlp_hidden_size', type=int, help='EP MLP hidden size')
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
    torch.manual_seed(args.train_seed)
    print(f"Train seed: {args.train_seed}") 
    config = load_config(os.path.join(args.model_config_dir, args.config))
    
    # Update config with args and keep track of overridden parameters
    overridden_parameters = []
    for k, v in vars(args).items():
        if v is not None and k in config and config[k] != v:
            config[k] = v
            overridden_parameters.append(f"{k}_{v}")
            print(f"{k}_{v}")

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
    mp_model = init_model_class(mp_config, args)
    ep_model = init_model_class(ep_config, args)
    
    # filename same as config makes it easier for identifying different parameters
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

    # load events data
    events_ds_file = os.path.join(args.analysis_dir,'results/event_inds/event_inds_mp_ds2.pkl')
    events_dataset = pickle.load(open(events_ds_file, 'rb'))

    # initialize the replay buffer
    burn_in_length = args.burn_in_length
    rollout_length = args.rollout_length
    sequence_length = burn_in_length + rollout_length
    replay_buffer = ReplayBufferEvents(burn_in_length, rollout_length, train_data, events_dataset['train'])
    batch_size = args.batch_size
    print(f'Batch size: {batch_size}')
    batches_per_epoch = np.min([replay_buffer.buffer_size // batch_size, 50])
    
    # make validation data
    val_buffer = ReplayBufferEvents(burn_in_length, rollout_length, val_data, events_dataset['val'])
    seed = 100 # set seed so every model sees the same randomization
    val_batch_size = np.min([val_data.size(0), 1000])
    val_trajs, val_event_states, val_event_horizons = val_buffer.sample(val_batch_size, random_seed=seed)
    val_trajs = val_trajs.to(DEVICE)
    val_event_states = val_event_states.to(DEVICE)
    val_event_horizons = val_event_horizons.to(DEVICE)
        
    print('Starting', model_filename, 'On DS', args.dataset)
    
    # log to tensorboard
    log_dir = os.path.join(args.checkpoint_dir, 'models/training_info/tensorboard/', model_filename)
    writer = SummaryWriter(log_dir=log_dir)
    
    # optimizer
    all_parameters = list(mp_model.parameters())+list(ep_model.parameters())
    opt = torch.optim.Adam(all_parameters, lr=args.lr)
    mp_model.to(DEVICE)
    ep_model.to(DEVICE)
    
    # store losses in dictionary (will vary by model type)
    loss_dict = {}
    loss_dict['train'] = []
    loss_dict['val'] = []
    loss_dict['mp_loss'] = []
    loss_dict['ep_loss'] = []
    loss_dict['epoch_times'] = []
    
    start_time = time.time()
    for epoch in tqdm(range(args.epochs)):
        mp_model.train()
        ep_model.train()
        batch_loss = []
        nsamples = 0
        for i in tqdm(range(batches_per_epoch)):
            batch_x, event_states, event_horizons = replay_buffer.sample(batch_size)
            batch_x = batch_x.to(DEVICE)
            nsamples += batch_x.shape[0]
            opt.zero_grad()
            ep_loss, event_loss, horizon_loss, event_hat, event_horizon_hat = ep_model.supervised_loss(batch_x[:,:burn_in_length,:], event_states)
            # condition ms predictor on predicted end states, detach from computational graph so gradients don't flow through esp_model
            if mp_config['input_pred_horizon']:
                # concatenate event_hat and event_horizon_hat
                event_hat = torch.cat([event_hat, event_horizon_hat], dim=-1)
                breakpoint()
            mp_loss = mp_model.supervised_loss(batch_x, event_hat.detach(), burn_in_length, rollout_length)
            loss = ep_loss + mp_loss
            loss.backward()
            opt.step()
            batch_loss.append([ep_loss.item(), mp_loss.item(), loss.item()])
        loss_dict['train'].append(np.mean([b_loss[-1] for b_loss in batch_loss]))
        loss_dict['ep_loss'].append(np.mean([b_loss[0] for b_loss in batch_loss])
        loss_dict['mp_loss'].append(np.mean([b_loss[1] for b_loss in batch_loss]))
        loss_dict['epoch_times'].append(time.time()-start_time)
            
        # test on validation set
        with torch.no_grad():
            mp_model.eval()
            ep_model.eval()
            val_ep_loss, val_event_loss, val_horizon_loss, val_event_hat, val_event_horizon_hat = esp_model.supervised_loss(val_trajs[:,:burn_in_length,:], val_event_states, val_event_horizons)
            val_mp_loss = mp_model.supervised_loss(val_trajs, val_event_hat, burn_in_length, rollout_length)
            val_loss = val_ep_loss + val_mp_loss
            val_loss = val_loss.item()
            loss_dict['val'].append(val_loss)
        # log to tensorboard
        writer.add_scalar('Train Loss/loss', loss_dict['train'][-1], epoch)
        writer.add_scalar('Train Loss/ep_loss', loss_dict['ep_loss'][-1], epoch)
        writer.add_scalar('Train Loss/mp_loss', loss_dict['mp_loss'][-1], epoch')
        writer.add_scalar('Val Loss/val', val_loss, epoch)
        
        if epoch % args.save_every == 0 or epoch == (args.epochs-1):
            # save checkpoints in checkpoint directory with plenty of storage
            save_dir = os.path.join(args.checkpoint_dir, 'models', model_filename)            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            ep_model_name = os.path.join(save_dir, f'ep_model_epoch{epoch}')
            torch.save(ep_model.state_dict(), ep_model_name)
            mp_model_name = os.path.join(save_dir, f'mp_model_epoch{epoch}')
            torch.save(mp_model.state_dict(), mp_model_name)
            # save training history in dataframe
            training_info = {
                'Epochs': np.arange(len(loss_dict['train']))}
            for key in loss_dict:
                training_info[key] = loss_dict[key]
            df_training = pd.DataFrame.from_dict(training_info)
            df_training.to_csv(os.path.join(save_dir, f'training_info_{model_filename}.csv'))
        print(f'Epoch {epoch}, Train Loss {loss_dict["train"][-1]}, Event Loss {loss_dict["ep_loss"][-1]}, MP Loss {loss_dict["mp_loss"][-1]}, Validation Loss {loss_dict["val"][-1]}')


if __name__ == '__main__':    
    main()

