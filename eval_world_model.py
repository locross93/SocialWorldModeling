# -*- coding: utf-8 -*-
"""
Created on Thu May 11 13:43:20 2023

@author: locro
"""
import os
import argparse
import pickle
import platform
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
# to enforce type check
from typeguard import typechecked
from typing import List, Tuple, Dict, Optional, Union, Any, cast
import torch

from constants import DEFAULT_VALUES, MODEL_DICT_VAL
from analysis_utils import load_trained_model
from models import DreamerV2
from annotate_pickup_timepoints import annotate_pickup_timepoints
from annotate_goal_timepoints import eval_recon_goals


@typechecked
class Analysis(object):
    """
    Class to perform analysis on the trained models

    Attributes
    ----------
    args : command line arguments including analysis directory, data directory, and checkpoint directory
    which_model : key indicating which model from MODEL_DICT_VAL to analyze
    """
    def __init__(self, args, model_dict) -> None:        
        self.args = args
        self.model_dict = model_dict           
        self.load_data()        
        self.save_plot = True

    def load_data(self) -> None:
        data_file = os.path.join(args.data_dir, 'train_test_splits_3D_dataset.pkl')        
        loaded_dataset = pickle.load(open(data_file, 'rb'))        
        _, test_dataset = loaded_dataset
        self.input_data = test_dataset.dataset.tensors[0][test_dataset.indices,:,:]
        self.num_timepoints = input_data.size(1)

    def load_model(self, model_key) -> torch.nn.Module:
        """
        Load the trained model from the checkpoint directory
        """
        model_info = MODEL_DICT_VAL[model_key]
        self.model_name = model_info['model_label']
        # set device, this assumes only one gpu is available to the script
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = load_trained_model(model_info, DEVICE)
        return model

    def eval_goal_events_in_rollouts(model, input_data, ds='First'):
        if ds == 'First':
            # first dataset
            pickup_timepoints = annotate_pickup_timepoints(train_or_val='val', pickup_or_move='move')
            single_goal_trajs = np.where((np.sum(pickup_timepoints > -1, axis=1) == 1))[0]
        else:
            # use event log for new datasets - TO DO
            pickup_timepoints = []
        
        num_single_goal_trajs = len(single_goal_trajs)
        imagined_trajs = np.zeros([num_single_goal_trajs, input_data.shape[1], input_data.shape[2]])
        num_timepoints = input_data.size(1)
        real_trajs = []
        imag_trajs = []
        for i,row in enumerate(single_goal_trajs):
            if i%50 == 0:
                print(i)
            x = input_data[row,:,:].unsqueeze(0)
            # get the only pick up point in the trajectory
            steps2pickup = np.max(pickup_timepoints[row,:]).astype(int)
            # store the steps before pick up with real frames in imagined_trajs
            imagined_trajs[i,:steps2pickup,:] = x[:,:steps2pickup,:].cpu()
            # rollout model for the rest of the trajectory
            rollout_length = num_timepoints - steps2pickup
            rollout_x = model.forward_rollout(x, steps2pickup, rollout_length).cpu().detach()
            # get end portion of true trajectory to compare to rollout
            real_traj = x[:,steps2pickup:,:].to("cpu")
            assert rollout_x.size() == real_traj.size()
            real_trajs.append(real_traj)
            imag_trajs.append(rollout_x)
            # store the steps after pick up with predicted frames in imagined_trajs
            imagined_trajs[i,steps2pickup:,:] = rollout_x
        x_true = torch.cat(real_trajs, dim=1)
        x_hat = torch.cat(imag_trajs, dim=1)
        mse = ((x_true - x_hat)**2).mean().item()
    
        full_trajs = input_data[single_goal_trajs,:,:].cpu()
        scores, y_labels, y_recon = eval_recon_goals(full_trajs, imagined_trajs, final_location=True, plot=False)
        # evaluate whether only appropriate goals (after object picked up) are reconstructed
        pickup_subset = pickup_timepoints[single_goal_trajs,:]
        indices = np.argwhere(pickup_subset > -1)
        accuracy = np.mean(y_recon[indices[:,0],indices[:,1]])
        result = {'Score': accuracy, 'MSE': mse}
        return result    

    def eval_one_model(self, model_key) -> Dict:
        model = self.load_model(model_key)
        model = model.eval()
        if self.args.eval_type == 'goal_events':
            result = self.eval_goal_events_in_rollouts(model, self.input_data)
        result['Model'] = MODEL_DICT_VAL[model_key]['model_label']
        return result

    def eval_all_models(self) -> List:
        self.results = []
        for model_key in self.args.model_keys:
            print(model_key)            
            result = self.eval_model()
            self.results.append(result)

def load_args():
    parser = argparse.ArgumentParser()
    # general pipeline parameters
    parser.add_argument('--analysis_dir', type=str, default=DEFAULT_VALUES['analysis_dir'], help='Analysis directory')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_VALUES['data_dir'], help='Data directory')
    parser.add_argument('--checkpoint_dir', type=str, default=DEFAULT_VALUES['checkpoint_dir'], help='Checkpoint directory')
    parser.add_argument('--model_keys', type=str, default=list(MODEL_DICT_VAL.keys()), help='Keys for all model to evaluate in MODEL_DICT_VAL')
    parser.add_argument('--eval_type', type=str, choices=DEFAULT_VALUES['eval_types'], default='goal_events', help='Type of evaluation to perform')
    parser.add_argument('--save_plot', action='store_true', help='Whether to save the plot')
    #parser.add_argument('--result_file_name', type=str, default='eval_rollouts_goal_events', help='Name of result file')    
    return parser.parse_args()

if __name__ == "__main__":    
    args = load_args()

    df_plot = pd.DataFrame(results)
    # Plot the results using seaborn
    fig = plt.figure()
    sns.barplot(x='Model', y='Score', data=df_plot) 
    plt.title('Evaluate Single Goal Events From Forward Rollouts', fontsize=14)
    plt.xlabel('Model Name', fontsize=16) 
    plt.ylabel('Accuracy', fontsize=16)
    plt.ylim([0, 1])
    if save_plot:
        df_plot.to_csv(analysis_dir+'results/figures/'+save_file+'.csv')
        plt.savefig(analysis_dir+'results/figures/'+save_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    fig = sns.barplot(x='Model', y='MSE', data=df_plot) 
    plt.title('Prediction Error of Goal Events in Imagination', fontsize=18)
    plt.xlabel('Model Name', fontsize=16) 
    plt.ylabel('MSE', fontsize=16)
    if save_plot:
        plt.savefig(analysis_dir+'results/figures/'+save_file+'_recon_error', dpi=300, bbox_inches='tight')
    plt.show()
            
        
        