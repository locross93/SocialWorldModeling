# -*- coding: utf-8 -*-
"""
Created on Thu May 11 13:43:20 2023

@author: locro
"""
import os
import argparse
import pickle
import torch
import numpy as np
import pandas as pd
# to enforce type checking
from typeguard import typechecked
import typing
from typing import List, Tuple, Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score


from constants_lc import DEFAULT_VALUES, MODEL_DICT_VAL, DATASET_NUMS
from analysis_utils import load_config, get_highest_numbered_file
from models import DreamerV2
from annotate_pickup_timepoints import annotate_pickup_timepoints
from annotate_goal_timepoints import eval_recon_goals, annotate_goal_timepoints


#@typechecked
class Analysis(object):
    """
    Class to perform analysis on the trained models

    Attributes
    ----------
    args : command line arguments including analysis directory, data directory, and checkpoint directory
    which_model : key indicating which model from MODEL_DICT_VAL to analyze
    """
    def __init__(self, args) -> None:        
        self.args = args

    def load_data(self) -> None:
        data_file = os.path.join(self.args.data_dir, self.args.dataset)        
        self.ds_num = DATASET_NUMS[self.args.dataset]
        self.loaded_dataset = pickle.load(open(data_file, 'rb'))        
        _, test_dataset = self.loaded_dataset
        self.input_data = test_dataset.dataset.tensors[0][test_dataset.indices,:,:]
        self.num_timepoints = self.input_data.size(1)
        # if 2+ dataset, load event log
        if self.ds_num > 1:
            # load dataset info
            exp_info_file = data_file[:-4]+'_exp_info.pkl'
            if os.path.isfile(exp_info_file):
                self.exp_info_dict = pickle.load(open(exp_info_file, 'rb'))


    def load_model(self, model_key) -> torch.nn.Module:
        """
        Load the trained model from the checkpoint directory
        """
        model_info = MODEL_DICT_VAL[model_key]
        self.model_name = model_info['model_label']
        # set device, this assumes only one gpu is available to the script
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_class = model_info['class']
        config_file = os.path.join(self.args.model_config_dir, model_info['config'])
        config = load_config(config_file)
        model = model_class(config)
        # load checkpoint weights
        # checkpoints are in folder named after model
        model_dir = os.path.join(self.args.checkpoint_dir, 'models', model_info['model_dir'])
        if 'epoch' in model_info:
            model_file_name = os.path.join(model_dir, model_info['model_dir']+'_epoch'+model_info['epoch'])
            model.load_state_dict(torch.load(model_file_name))
            print('Loading model',model_file_name)
        else:
            latest_checkpoint, _ =  get_highest_numbered_file(model_info['model_dir'], model_dir)
            print('Loading from last checkpoint',latest_checkpoint)
            model.load_state_dict(torch.load(latest_checkpoint))

        model.eval()
        model.to(DEVICE)
        return model


    def eval_goal_events_in_rollouts(self, model, input_data, ds='First') -> Dict[str, typing.Any]:
        if self.ds_num == 1:
            # first dataset
            pickup_timepoints = annotate_pickup_timepoints(self.loaded_dataset, train_or_val='val', pickup_or_move='move')
            single_goal_trajs = np.where((np.sum(pickup_timepoints > -1, axis=1) == 1))[0]
        elif self.ds_num > 1:
            # 2+ dataset use event logger to define events
            pickup_timepoints = self.exp_info_dict[self.args.train_or_val]['pickup_timepoints']
            single_goal_trajs = self.exp_info_dict[self.args.train_or_val]['single_goal_trajs']
        
        num_single_goal_trajs = len(single_goal_trajs)
        imagined_trajs = np.zeros([num_single_goal_trajs, input_data.shape[1], input_data.shape[2]])        
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
            rollout_length = self.num_timepoints - steps2pickup
            rollout_x = model.forward_rollout(x.cuda(), steps2pickup, rollout_length).cpu().detach()
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
        result = {'model': self.model_name, 'score': accuracy, 'MSE': mse}        
        return result
    

    def eval_multigoal_events_in_rollouts(self, model, input_data, ds='First') -> Dict[str, Any]:        
        if ds == 'First':
            # first dataset
            pickup_timepoints = annotate_pickup_timepoints(self.loaded_dataset, train_or_val='val', pickup_or_move='move')
            multi_goal_trajs = np.where((np.sum(pickup_timepoints > -1, axis=1) == 3))[0]
        else:
            # 2+ dataset use event logger to define events
            pickup_timepoints = self.exp_info_dict[self.args.train_or_val]['pickup_timepoints']
            multi_goal_trajs = self.exp_info_dict[self.args.train_or_val]['multi_goal_trajs']
            
        num_multi_goal_trajs = len(multi_goal_trajs)
        imagined_trajs = np.zeros([num_multi_goal_trajs, input_data.shape[1], input_data.shape[2]])
        real_trajs = []
        imag_trajs = []
        for i,row in enumerate(multi_goal_trajs):
            if i%50 == 0:
                print(i)
            x = input_data[row,:,:].unsqueeze(0)
            # burn in to the pick up point of the 2nd object that is picked up, so its unambiguous that all objects will be delivered
            steps2pickup = np.sort(pickup_timepoints[row,:])[1].astype(int)
            # store the steps before pick up with real frames in imagined_trajs
            imagined_trajs[i,:steps2pickup,:] = x[:,:steps2pickup,:].cpu()
            # rollout model for the rest of the trajectory
            rollout_length = self.num_timepoints - steps2pickup
            rollout_x = model.forward_rollout(x.cuda(), steps2pickup, rollout_length).cpu().detach()
            # get end portion of true trajectory to compare to rollout
            real_traj = x[:,steps2pickup:,:].to("cpu")
            assert rollout_x.size() == real_traj.size()
            real_trajs.append(real_traj)
            imag_trajs.append(rollout_x)
            imagined_trajs[i,steps2pickup:,:] = rollout_x
        x_true = torch.cat(real_trajs, dim=1)
        x_hat = torch.cat(imag_trajs, dim=1)
        mse = ((x_true - x_hat)**2).mean().item()
        
        full_trajs = input_data[multi_goal_trajs,:,:].cpu()
        scores, y_labels, y_recon = eval_recon_goals(full_trajs, imagined_trajs, final_location=False, plot=False)
        # 100% accuracy is all goal labels are == 1
        assert np.mean(y_labels) == 1.0
        # get accuracies separately for 2nd object and 3rd object (3rd object the hardest to imagine properly)
        goals_obj1 = []
        goals_obj2 = []
        goals_obj3 = []
        for i,row in enumerate(multi_goal_trajs):
            pickup_seq = np.argsort(pickup_timepoints[row,:])
            goals_obj1.append(y_recon[i,pickup_seq[0]])
            goals_obj2.append(y_recon[i,pickup_seq[1]])
            goals_obj3.append(y_recon[i,pickup_seq[2]])

        acc_g2 = np.mean(goals_obj2)
        acc_g3 = np.mean(goals_obj3)
        
        result = {
            'model': self.model_name,
            'g2_acc': acc_g2,
            'g2_goal_num': '2nd Goal',
            'g2_mse': mse,
            'g3_acc': acc_g3,
            'g3_goal_num': '3rd Goal',
            'g3_mse': mse}
        
        return result
    

    def _eval_move_events(self, input_matrices, recon_matrices, move_thr) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        def detect_object_move(trial_obj_pos, move_thr):
            obj_pos_delta = np.abs(trial_obj_pos[1:,:] - trial_obj_pos[:-1,:])
            obj_pos_delta_sum = obj_pos_delta.sum(axis=1)
            total_movement = obj_pos_delta_sum.sum()            
            if total_movement > move_thr:
                return 1
            else:
                return 0
            
        if input_matrices.shape[1:] != (300, 23):
            input_matrices = input_matrices.reshape(-1, 300, 23)
            
        if recon_matrices.shape[1:] != (300, 23):
            recon_matrices = recon_matrices.reshape(-1, 300, 23)
            
        if hasattr(recon_matrices, 'requires_grad') and recon_matrices.requires_grad:
            recon_matrices = recon_matrices.detach().numpy()
            
        scores = {}
        
        data_columns = ['obj0_x', 'obj0_y', 'obj0_z', 'obj1_x', 'obj1_y', 'obj1_z', 'obj2_x',
            'obj2_y', 'obj2_z', 'agent0_x', 'agent0_y', 'agent0_z', 'agent0_rot_x',
            'agent0_rot_y', 'agent0_rot_z', 'agent0_rot_w', 'agent1_x', 'agent1_y',
            'agent1_z', 'agent1_rot_x', 'agent1_rot_y', 'agent1_rot_z',
            'agent1_rot_w'] 
        dims = ['x', 'y', 'z']

        num_trials = input_matrices.shape[0]
        num_objs = 3
        obj_moved_flag = np.zeros([num_trials, 3])
        # hard threshold to detect movement (unchangable by user)
        move_thr_true = 0.1
        for i in range(num_trials):
            trial_x = input_matrices[i,:,:]
            for j in range(num_objs):
                pos_inds = [data_columns.index('obj'+str(j)+'_'+dim) for dim in dims]
                trial_obj_pos = trial_x[:,pos_inds]
                obj_moved_flag[i,j] = detect_object_move(trial_obj_pos, move_thr_true)
                
        recon_moved_flag = np.zeros([num_trials, 3])
        for i in range(num_trials):
            trial_x = recon_matrices[i,:,:]
            for j in range(num_objs):
                pos_inds = [data_columns.index('obj'+str(j)+'_'+dim) for dim in dims]
                trial_obj_pos = trial_x[:,pos_inds]
                recon_moved_flag[i,j] = detect_object_move(trial_obj_pos, move_thr)
                
        scores['accuracy'] = accuracy_score(obj_moved_flag.reshape(-1), recon_moved_flag.reshape(-1))
        scores['precision'] = precision_score(obj_moved_flag.reshape(-1), recon_moved_flag.reshape(-1))
        scores['recall'] = recall_score(obj_moved_flag.reshape(-1), recon_moved_flag.reshape(-1))        
        return scores, obj_moved_flag, recon_moved_flag
    
    def eval_move_events_in_rollouts(self, model, input_data, ds='First') -> Dict[str, Any]:
        if ds == 'First':
            # first dataset
            pickup_timepoints = annotate_pickup_timepoints(self.loaded_dataset, train_or_val='val', pickup_or_move='move')
            goal_timepoints = annotate_goal_timepoints(self.loaded_dataset, train_or_val='val')
            single_goal_trajs = np.where((np.sum(pickup_timepoints > -1, axis=1) == 1))[0]
            multi_goal_trajs = np.where((np.sum(pickup_timepoints > -1, axis=1) == 3))[0]
        else:
            # use event log for new datasets - TO DO
            pickup_timepoints = []
        
        imagined_trajs = np.zeros(input_data.shape)
        for i in range(input_data.shape[0]):
            if i%50 == 0:
                print(i)
            x = input_data[i,:,:].unsqueeze(0)
            if i in single_goal_trajs:
                # burn in to a few frames past the goal, so it is clear it is a single goal trial - TO DO THIS WILL BE DIFF FOR DS2
                # get the only goal point in the trajectory
                burn_in_length = np.max(goal_timepoints[i,:]).astype(int) + 10
            elif i in multi_goal_trajs:
                # burn in to the pick up point of the 2nd object that is picked up, so its unambiguous that all objects will be delivered
                burn_in_length = np.sort(pickup_timepoints[i,:])[1].astype(int)
            else:                
                burn_in_length = self.args.non_goal_burn_in
            # store the steps of burn in with real frames in imagined_trajs
            imagined_trajs[i,:burn_in_length,:] = x[:,:burn_in_length,:].cpu()
            # rollout model for the rest of the trajectory
            rollout_length = self.num_timepoints - burn_in_length
            rollout_x = model.forward_rollout(x.cuda(), burn_in_length, rollout_length).cpu().detach()
            # store the steps after pick up with predicted frames in imagined_trajs
            imagined_trajs[i,burn_in_length:,:] = rollout_x
                            
        input_data = input_data.cpu().detach().numpy()
        # result = {'Accuracy':_, 'Precision':_, 'Recall':_}
        result, obj_moved_flag, recon_moved_flag = self._eval_move_events(
            input_data, imagined_trajs, self.args.move_threshold)
        result['model'] = self.model_name
        return result    


    def eval_one_model(self, model_key) -> Dict[str, Any]:        
        model = self.load_model(model_key)
        result = {}
        result['Model'] = MODEL_DICT_VAL[model_key]['model_label']

        if self.args.eval_type == 'goal_events':
            result = self.eval_goal_events_in_rollouts(model, self.input_data)            
        elif self.args.eval_type == 'multigoal_events':
            result = self.eval_multigoal_events_in_rollouts(model, self.input_data)
        elif self.args.eval_type == 'move_events':
            result = self.eval_move_events_in_rollouts(model, self.input_data)        
        else:
            raise NotImplementedError(f'Evaluation type {self.args.eval_type} not implemented')    
        return result


    def eval_all_models(self) -> None:
        self.results = []
        print(f'>>>>> Evaluating {self.args.eval_type} for all models <<<<<')
        for model_key in self.args.model_keys:
            print(f'Currently evaluating model {model_key}')
            result = self.eval_one_model(model_key)
            self.results.append(result)
    

    def save_results(self) -> None:
        save_file = f'eval_{self.args.eval_type}'
        save_path = os.path.join(self.args.analysis_dir, 'results', f'{save_file}.csv')
        df_results = pd.DataFrame(self.results)
        df_results.to_csv(save_path)                             


    def run(self) -> None:
        self.load_data()
        self.eval_all_models()
        self.save_results()




def load_args():
    parser = argparse.ArgumentParser()
    # general pipeline parameters
    parser.add_argument('--analysis_dir', type=str, action='store',
                        default=DEFAULT_VALUES['analysis_dir'], 
                        help='Analysis directory')
    parser.add_argument('--model_config_dir', type=str, action='store',
                        default=DEFAULT_VALUES['model_config_dir'],
                        help='Model config directory')   
    parser.add_argument('--data_dir', type=str, action='store',
                        default=DEFAULT_VALUES['data_dir'], 
                        help='Data directory')
    parser.add_argument('--checkpoint_dir', type=str, action='store',
                        default=DEFAULT_VALUES['checkpoint_dir'], 
                        help='Checkpoint directory')
    parser.add_argument('--dataset', type=str,
                         default='dataset_5_25_23.pkl', 
                         help='Dataset')
    parser.add_argument('--model_keys', nargs='+', action='store',
                        default=DEFAULT_VALUES['model_keys'], 
                        help='A list of keys, seperate by spaces, for all model to evaluate')
    parser.add_argument('--eval_type', type=str, action='store',
                        choices=DEFAULT_VALUES['eval_types'], 
                        default='goal_events', 
                        help='Type of evaluation to perform')
    parser.add_argument('--move_threshold', action='store',
                        default=DEFAULT_VALUES['move_threshold'],
                        help='Threshold for move event evaluation')
    parser.add_argument('--non_goal_burn_in', action='store',
                        default=DEFAULT_VALUES['non_goal_burn_in'],
                        help='Number of frames to burn in for non-goal events')    
    return parser.parse_args()




if __name__ == "__main__":    
    args = load_args()
    analysis = Analysis(args)
    analysis.run()