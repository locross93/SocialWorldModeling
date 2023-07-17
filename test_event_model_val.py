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
from models import EventPredictor, MSPredictorEventContext, ReplayBufferEvents

class EventModel:
    def __init__(self, ep_config, mp_config):
        self.ep_model = EventPredictor(ep_config)
        self.mp_model = MSPredictorEventContext(mp_config)
        
    def load_weights(self, ep_weights_path, mp_weights_path):
        # Load the weights of the two models
        self.ep_model.load_state_dict(torch.load(ep_weights_path))
        self.mp_model.load_state_dict(torch.load(mp_weights_path))

    def predict_next_event(self, x):
        # Call EventPredictor's forward function with burn_in steps - x should be burn_in length
        batch_size = x.size(0)
        hidden = self.ep_model.init_hidden(batch_size)
        if self.ep_model.predict_horizon:
            out_event_state, out_event_horizon, hidden = self.ep_model.forward(x, hidden)
            # last output is prediction with entire burn in sequence
            event_hat = out_event_state[:,-1,:]
            event_horizon_hat = out_event_horizon[:,-1,:].squeeze(1)
            return event_hat, event_horizon_hat
        else:
            out_event_state, hidden = self.ep_model.forward(x, hidden)
            # last output is prediction with entire burn in sequence
            event_hat = out_event_state[:,-1,:]
            return event_hat
    
    def forward_rollout(self, x, burn_in_length, rollout_length):
        if self.ep_model.predict_horizon:
            event_hat, event_horizon_hat = self.predict_next_event(x[:,:burn_in_length,:])
        else:
            event_hat = self.predict_next_event(x[:,:burn_in_length,:])

        # condition ms predictor on predicted next event
        if self.mp_model.input_pred_horizon:
            # concatenate event_hat and event_horizon_hat
            event_hat = torch.cat([event_hat, event_horizon_hat.unsqueeze(1)], dim=-1)
        # Call MSPredictorEventContext's forward_rollout with event_hat
        x_hat = self.mp_model.forward_rollout(x, event_hat, burn_in_length, rollout_length)

        return x_hat

    def loss(self, x, burn_in_length, rollout_length):
        loss_fn = torch.nn.MSELoss()
        
        x_hat = self.forward_rollout(x, burn_in_length, rollout_length)
        t_end = burn_in_length + rollout_length
        x_supervise = x[:,burn_in_length:t_end,:]
        
        loss = loss_fn(x_supervise, x_hat)
        
        return loss

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

    model = EventModel(ep_config, mp_config)
    ep_weights_path = os.path.join(args.checkpoint_dir, 'models/event_context_world/ep_model_epoch4200')
    mp_weights_path = os.path.join(args.checkpoint_dir, 'models/event_context_world/mp_model_epoch4200')
    model.load_weights(ep_weights_path, mp_weights_path)

    # Check that we are using GPU
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {DEVICE}')

    model.mp_model.to(DEVICE)
    model.ep_model.to(DEVICE)

    # load data    
    dataset_file = os.path.join(args.data_dir, args.dataset)
    loaded_dataset = pickle.load(open(dataset_file, 'rb'))
    train_dataset, test_dataset = loaded_dataset
    train_data = train_dataset.dataset.tensors[0][train_dataset.indices,:,:]
    val_data = test_dataset.dataset.tensors[0][test_dataset.indices,:,:]

    # load events data
    events_ds_file = os.path.join(args.analysis_dir,'results/event_inds/event_inds_mp_ds2.pkl')
    events_dataset = pickle.load(open(events_ds_file, 'rb'))

    # make validation data
    burn_in_length = args.burn_in_length
    rollout_length = args.rollout_length
    val_buffer = ReplayBufferEvents(burn_in_length, rollout_length, val_data, events_dataset['val'])
    seed = 100 # set seed so every model sees the same randomization
    val_batch_size = np.min([val_data.size(0), 1000])
    val_trajs, val_event_states, val_event_horizons = val_buffer.sample(val_batch_size, random_seed=seed)
    val_trajs = val_trajs.to(DEVICE)
    val_event_states = val_event_states.to(DEVICE)
    val_event_horizons = val_event_horizons.to(DEVICE)

    # test on validation set
    loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        model.mp_model.eval()
        model.ep_model.eval()
        val_loss = model.loss(val_trajs, burn_in_length, rollout_length)
        val_loss = val_loss.item()
    print(f'Validation Loss {val_loss}')

if __name__ == '__main__':    
    main()