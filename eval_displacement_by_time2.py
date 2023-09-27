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
from functools import reduce
from scipy.spatial import distance
# to enforce type checking
from typeguard import typechecked
import typing
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

from constants import DEFAULT_VALUES, MODEL_DICT_VAL, DATASET_NUMS
from analysis_utils import load_config, get_highest_numbered_file, get_data_columns, init_model_class, inverse_normalize
from annotate_pickup_timepoints import annotate_pickup_timepoints
from annotate_goal_timepoints import eval_recon_goals, annotate_goal_timepoints
from plot_eval_world_model import plot_eval_wm_results


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
        torch.manual_seed(args.eval_seed)
        print(f"Eval seed: {args.eval_seed}")           

    def load_data(self) -> None:
        #if self.args.dataset == 'train_test_splits_3D_dataset.pkl' or self.args.dataset == 'data_norm_velocity.pkl':
        if self.args.dataset == 'train_test_splits_3D_dataset.pkl':
            self.ds_num = 1
        else:
            #self.ds_num = 2
            self.ds_num = DATASET_NUMS[self.args.dataset]
        self.dataset_file = os.path.join(self.args.data_dir, self.args.dataset)
        self.loaded_dataset = pickle.load(open(self.dataset_file, 'rb'))
        train_dataset, test_dataset = self.loaded_dataset
        if self.args.train_or_val == 'train':
            self.input_data = train_dataset.dataset.tensors[0][train_dataset.indices,:,:]
        else:
            self.input_data = test_dataset.dataset.tensors[0][test_dataset.indices,:,:]
        self.num_timepoints = self.input_data.size(1)
        # if 2+ dataset, load event log
        if self.ds_num > 1:
            # load dataset info
            exp_info_file = self.dataset_file[:-4]+'_exp_info.pkl'
            if os.path.isfile(exp_info_file):
                with open(exp_info_file, 'rb') as f:
                    self.exp_info_dict = pickle.load(f)
            else:
                print('DS info dict not found')
            self.data_columns = self.exp_info_dict['data_columns']
        else:
            self.data_columns = get_data_columns(DATASET_NUMS[self.args.dataset])


    def load_model(self, model_key, epoch=None) -> torch.nn.Module:
        """
        Load the trained model from the checkpoint directory
        """
        model_info = MODEL_DICT_VAL[model_key]
        if epoch is not None:
            model_info['epoch'] = epoch
        self.model_name = model_info['model_label']        
        # set device, this assumes only one gpu is available to the script
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        setattr(self.args, 'device', DEVICE)
        config_file = os.path.join(self.args.model_config_dir, model_info['config'])
        config = load_config(config_file)
        model_type = config['model_type']        
        model = init_model_class(config, self.args)

        event_models = ['event_model']
        if model_type in event_models:
            # load weights
            if 'epoch' in model_info:
                self.epoch = model_info['epoch']
                ep_weights_path = os.path.join(self.args.checkpoint_dir, 'models', model_info['model_dir'], 'ep_model_epoch'+model_info['epoch'])
                mp_weights_path = os.path.join(self.args.checkpoint_dir, 'models', model_info['model_dir'], 'mp_model_epoch'+model_info['epoch'])
                model.load_weights(ep_weights_path, mp_weights_path)
            else:
                ep_weights_path, _ = get_highest_numbered_file('ep_model', os.path.join(self.args.checkpoint_dir, 'models', model_info['model_dir']))
                mp_weights_path, _ = get_highest_numbered_file('mp_model', os.path.join(self.args.checkpoint_dir, 'models', model_info['model_dir']))
                print('Loading from last checkpoint', mp_weights_path)
                self.epoch = mp_weights_path
                model.load_weights(ep_weights_path, mp_weights_path)
            model.mp_model.device = self.args.device
            model.ep_model.device = self.args.device
            model.device = self.args.device
            model.mp_model.to(self.args.device)
            model.ep_model.to(self.args.device)
        else:
            # load checkpoint weights
            # checkpoints are in folder named after model
            model_dir = os.path.join(self.args.checkpoint_dir, 'models', model_info['model_dir'])
            if 'epoch' in model_info:
                self.epoch = model_info['epoch']
                model_file_name = os.path.join(model_dir, model_info['model_dir']+'_epoch'+model_info['epoch'])
                model.load_state_dict(torch.load(model_file_name))
                print('Loading model',model_file_name)
            else:
                latest_checkpoint, _ =  get_highest_numbered_file(model_info['model_dir'], model_dir)
                print('Loading from last checkpoint', latest_checkpoint)
                self.epoch = latest_checkpoint
                model.load_state_dict(torch.load(latest_checkpoint))
            model.device = self.args.device
            model.to(self.args.device)
        model.eval()

        return model


    def calculate_ADE_FDE(self, real_trajs, imag_trajs):
        ade_per_trial = []
        fde_per_trial = []

        for real_traj, imag_traj in zip(real_trajs, imag_trajs):
            # Check if both real and imagined trajectories have the same shape
            assert real_traj.shape == imag_traj.shape
            
            # Calculate Euclidean distance between corresponding points
            distance = torch.norm(real_traj - imag_traj, dim=-1)
            
            # Compute ADE for this trial by taking the average over timesteps
            ade_per_trial.append(torch.mean(distance))
            
            # Compute FDE for this trial by taking the distance at the final timestep
            if len(distance.shape) == 2:
                fde_per_trial.append(distance[0, -1])
            elif len(distance.shape) == 1:
                fde_per_trial.append(distance[-1])

        # Compute overall ADE and FDE by averaging across trials
        ADE = torch.mean(torch.tensor(ade_per_trial))
        FDE = torch.mean(torch.tensor(fde_per_trial))

        return ADE.item(), FDE.item()


    def eval_goal_events_in_rollouts(self, model, input_data, offset=5, partial=1.0) -> Dict[str, typing.Any]:
        if self.ds_num == 1:
            # first dataset
            pickup_timepoints = annotate_pickup_timepoints(self.loaded_dataset, train_or_val='val', pickup_or_move='move', ds_num=self.ds_num)
            single_goal_trajs = np.where((np.sum(pickup_timepoints > -1, axis=1) == 1))[0]
        else:
            # 2+ dataset use event logger to define events
            pickup_timepoints = self.exp_info_dict[self.args.train_or_val]['pickup_timepoints']
            single_goal_trajs = self.exp_info_dict[self.args.train_or_val]['single_goal_trajs']
        
        single_goal_trajs = single_goal_trajs[:int(partial*len(single_goal_trajs))]
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
            if offset != 0:
                # burn in to frames before or after pick up point to control difficulty
                steps2pickup = steps2pickup + offset
                if i == 0:
                    print('Offset:',offset)
                # if steps2pickup > 15:
                #     steps2pickup = steps2pickup - 15
                # elif steps2pickup > 10:
                #     steps2pickup = steps2pickup - 5
                # else:
                #     # TO DO - don't include trial if not enough burn in available
                #     steps2pickup = steps2pickup - 1
            # store the steps before pick up with real frames in imagined_trajs
            imagined_trajs[i,:steps2pickup,:] = x[:,:steps2pickup,:].cpu()
            # rollout model for the rest of the trajectory
            rollout_length = self.num_timepoints - steps2pickup        
            rollout_x = model.forward_rollout(x.cuda(), steps2pickup, rollout_length).cpu().detach()
            # Replace any nan, inf, or outliers values with 0
            rollout_x[torch.isnan(rollout_x) | torch.isinf(rollout_x) | (torch.abs(rollout_x) > 1e3)] = 0
            # get end portion of true trajectory to compare to rollout
            real_traj = x[:,steps2pickup:,:].to("cpu")
            assert rollout_x.size() == real_traj.size()
            real_trajs.append(real_traj)
            imag_trajs.append(rollout_x)
            # store the steps after pick up with predicted frames in imagined_trajs
            imagined_trajs[i,steps2pickup:,:] = rollout_x
        # LC 8/8/23 is this correct? Is this for MSE?
        x_true = torch.cat(real_trajs, dim=1)
        x_hat = torch.cat(imag_trajs, dim=1)
        full_trajs = input_data[single_goal_trajs,:,:].cpu()
        if 'min_max_values' in self.exp_info_dict:
            # data is normalize, project it back into regular input space
            min_values, max_values = self.exp_info_dict['min_max_values']
            velocity = any('vel' in s for s in self.data_columns)
            x_true = inverse_normalize(x_true, max_values.astype(np.float32), min_values.astype(np.float32), velocity)
            x_hat = inverse_normalize(x_hat, max_values.astype(np.float32), min_values.astype(np.float32), velocity)
            full_trajs = inverse_normalize(full_trajs, max_values.astype(np.float32), min_values.astype(np.float32), velocity)
        if any('vel' in item for item in self.data_columns):
            # remove velocity columns
            x_true = x_true[:,:,::2]
            x_hat = x_hat[:,:,::2]
            full_trajs = full_trajs[:,:,::2]
        # mse = ((x_true - x_hat)**2).mean().item()
        # compute average displacement error and final displacement error
        ade, fde = self.calculate_ADE_FDE(real_trajs, imag_trajs)

        #scores, y_labels, y_recon = eval_recon_goals(full_trajs, imagined_trajs, final_location=False, plot=False, ds_num=self.ds_num, obj_dist_thr=2.0, agent_dist_thr=1.0)
        # Easier
        scores, y_labels, y_recon = eval_recon_goals(full_trajs, imagined_trajs, final_location=False, plot=False, ds_num=self.ds_num, obj_dist_thr=2.0, agent_dist_thr=None)
        # evaluate whether only appropriate goals (after object picked up) are reconstructed
        pickup_subset = pickup_timepoints[single_goal_trajs,:]
        indices = np.argwhere(pickup_subset > -1)
        accuracy = np.mean(y_recon[indices[:,0],indices[:,1]])
        
        result = {'model': self.model_name, 'score': accuracy, 'ADE': ade, 'FDE': fde}      
        return result
           
    def eval_multigoal_events_in_rollouts(self, model, input_data, partial=1.0) -> Dict[str, Any]:        
        if self.ds_num == 1:
            # first dataset
            pickup_timepoints = annotate_pickup_timepoints(self.loaded_dataset, train_or_val='val', pickup_or_move='move', ds_num=self.ds_num)
            multi_goal_trajs = np.where((np.sum(pickup_timepoints > -1, axis=1) == 3))[0]
        else:
            # 2+ dataset use event logger to define events
            pickup_timepoints = self.exp_info_dict[self.args.train_or_val]['pickup_timepoints']
            multi_goal_trajs = self.exp_info_dict[self.args.train_or_val]['multi_goal_trajs']
        
        multi_goal_trajs = multi_goal_trajs[:int(partial*len(multi_goal_trajs))]
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
            # Replace any nan, inf, or outliers values with 0
            rollout_x[torch.isnan(rollout_x) | torch.isinf(rollout_x) | (torch.abs(rollout_x) > 1e3)] = 0
            # get end portion of true trajectory to compare to rollout
            real_traj = x[:,steps2pickup:,:].to("cpu")                
            assert rollout_x.size() == real_traj.size()
            real_trajs.append(real_traj)
            imag_trajs.append(rollout_x)
            imagined_trajs[i,steps2pickup:,:] = rollout_x
        
        x_true = torch.cat(real_trajs, dim=1)
        x_hat = torch.cat(imag_trajs, dim=1)
        #mse = ((x_true - x_hat)**2).mean().item()
        ade, fde = self.calculate_ADE_FDE(real_trajs, imag_trajs)
        
        full_trajs = input_data[multi_goal_trajs,:,:].cpu()
        scores, y_labels, y_recon = eval_recon_goals(full_trajs, imagined_trajs, final_location=False, plot=False, ds_num=self.ds_num)
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
        
        # result = {
        #     'model': self.model_name,
        #     'g2_acc': acc_g2,
        #     'g2_goal_num': '2nd Goal',
        #     'g2_mse': mse,
        #     'g3_acc': acc_g3,
        #     'g3_goal_num': '3rd Goal',
        #     'g3_mse': mse}
        result = {
            'model': self.model_name,
            'g2_acc': acc_g2,
            'g2_goal_num': '2nd Goal',
            'g3_acc': acc_g3,
            'g3_goal_num': '3rd Goal',
            'ADE': ade,
            'FDE': fde}
        
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
            
        if hasattr(recon_matrices, 'requires_grad') and recon_matrices.requires_grad:
            recon_matrices = recon_matrices.detach().numpy()
            
        scores = {}        
        dims = ['x', 'y', 'z']

        num_trials = input_matrices.shape[0]
        num_objs = 3
        obj_moved_flag = np.zeros([num_trials, 3])
        # hard threshold to detect movement (unchangable by user)
        move_thr_true = 0.1
        for i in tqdm(range(num_trials)):
            trial_x = input_matrices[i,:,:]
            for j in range(num_objs):
                pos_inds = [self.data_columns.index('obj'+str(j)+'_'+dim) for dim in dims]
                trial_obj_pos = trial_x[:,pos_inds]
                obj_moved_flag[i,j] = detect_object_move(trial_obj_pos, move_thr_true)
                
        recon_moved_flag = np.zeros([num_trials, 3])
        for i in tqdm(range(num_trials)):
            trial_x = recon_matrices[i,:,:]
            for j in range(num_objs):
                pos_inds = [self.data_columns.index('obj'+str(j)+'_'+dim) for dim in dims]
                trial_obj_pos = trial_x[:,pos_inds]
                recon_moved_flag[i,j] = detect_object_move(trial_obj_pos, move_thr)
                
        # TO DO - DON'T COUNT TRIALS WERE OBJECT MOVED FROM UNPREDICTABLE COLLISION
                
        scores['accuracy'] = accuracy_score(obj_moved_flag.reshape(-1), recon_moved_flag.reshape(-1))
        scores['precision'] = precision_score(obj_moved_flag.reshape(-1), recon_moved_flag.reshape(-1))
        scores['recall'] = recall_score(obj_moved_flag.reshape(-1), recon_moved_flag.reshape(-1))        
        return scores, obj_moved_flag, recon_moved_flag
    
    def eval_move_events_in_rollouts(self, model, input_data, partial=1.0) -> Dict[str, Any]:
        if self.ds_num == 1:
            # first dataset
            pickup_timepoints = annotate_pickup_timepoints(self.loaded_dataset, train_or_val='val', pickup_or_move='move', ds_num=self.ds_num)
            goal_timepoints = annotate_goal_timepoints(self.loaded_dataset, train_or_val='val', ds_num=self.ds_num)
            single_goal_trajs = np.where((np.sum(pickup_timepoints > -1, axis=1) == 1))[0]
            multi_goal_trajs = np.where((np.sum(pickup_timepoints > -1, axis=1) == 3))[0]
        else:
            # 2+ dataset use event logger to define events
            pickup_timepoints = self.exp_info_dict[self.args.train_or_val]['pickup_timepoints']
            goal_timepoints = self.exp_info_dict[self.args.train_or_val]['goal_timepoints']
            single_goal_trajs = self.exp_info_dict[self.args.train_or_val]['single_goal_trajs']
            multi_goal_trajs = self.exp_info_dict[self.args.train_or_val]['multi_goal_trajs']
        
        imagined_trajs = np.zeros(input_data.shape)
        
        goal_inds = np.concatenate([single_goal_trajs, multi_goal_trajs])        
        non_goal_trajs = [i for i in range(len(input_data)) if i not in goal_inds]
        non_goal_trajs = non_goal_trajs[ :int(len(input_data)*partial)]
        goal_inds =  np.random.choice(goal_inds, size=int(len(goal_inds)*partial), replace=False)     

        real_trajs = []
        imag_trajs = []                               

        # compute non goal trajs first with batches
        counter = 0
        batch_trajs = []
        batch_inds = []
        burn_in_length = self.args.non_goal_burn_in
        rollout_length = self.num_timepoints - burn_in_length
        for i in non_goal_trajs:
            counter += 1
            if counter % 100 == 0:
                print(i)
            x = input_data[i,:,:]#.unsqueeze(0)
            # store the steps of burn in with real frames in imagined_trajs
            imagined_trajs[i,:burn_in_length,:] = x[:burn_in_length,:].cpu()
            batch_inds.append(i)
            batch_trajs.append(x)
            real_trajs.append(x[burn_in_length:,:])
            if counter > 0 and counter % self.args.batch_size == 0:
                batch_x = torch.stack(batch_trajs, dim=0)
                batch_x = torch.stack(batch_trajs, dim=0)
                batch_x = batch_x.to(self.args.device)#model.DEVICE)
                rollout_x = model.forward_rollout(batch_x, burn_in_length, rollout_length).cpu().detach()
                # Replace any nan, inf, or outliers values with 0
                rollout_x[torch.isnan(rollout_x) | torch.isinf(rollout_x) | (torch.abs(rollout_x) > 1e3)] = 0
                batch_inds = np.array(batch_inds)
                imagined_trajs[batch_inds,burn_in_length:,:] = rollout_x
                # append real and imagined trajs to lists
                for rollout_trial in rollout_x:
                    imag_trajs.append(rollout_trial)
                batch_trajs = []
                batch_inds = []
        # compute last batch that is less than batch_size
        batch_x = torch.stack(batch_trajs, dim=0)
        batch_x = batch_x.to(self.args.device)#model.DEVICE)
        rollout_x = model.forward_rollout(batch_x, burn_in_length, rollout_length).cpu().detach()
        batch_inds = np.array(batch_inds)
        imagined_trajs[batch_inds,burn_in_length:,:] =  rollout_x
        # append real and imagined trajs to lists
        for rollout_trial in rollout_x:
            imag_trajs.append(rollout_trial)
        
        for i in tqdm(goal_inds):
            counter += 1
            if counter % 100 == 0:
                print(i)
            x = input_data[i,:,:].unsqueeze(0)
            if i in single_goal_trajs:
                # burn in to a few frames past the goal, so it is clear it is a single goal trial - TO DO THIS WILL BE DIFF FOR DS2
                # get the only goal point in the trajectory
                burn_in_length = np.max(goal_timepoints[i,:]).astype(int) + 10
                if burn_in_length > input_data.size(1):
                    # very late pickup
                    burn_in_length = np.max(goal_timepoints[i,:]).astype(int)
            elif i in multi_goal_trajs:
                # burn in to the pick up point of the 2nd object that is picked up, so its unambiguous that all objects will be delivered
                burn_in_length = np.sort(pickup_timepoints[i,:])[1].astype(int)
            else:                
                burn_in_length = self.args.non_goal_burn_in
            # store the steps of burn in with real frames in imagined_trajs
            imagined_trajs[i,:burn_in_length,:] = x[:,:burn_in_length,:].cpu()
            # rollout model for the rest of the trajectory
            rollout_length = self.num_timepoints - burn_in_length
            if x.dtype == torch.float64:
                x = x.float()
            rollout_x = model.forward_rollout(x.cuda(), burn_in_length, rollout_length).cpu().detach()
            # Replace any nan, inf, or outliers values with 0
            rollout_x[torch.isnan(rollout_x) | torch.isinf(rollout_x) | (torch.abs(rollout_x) > 1e3)] = 0
            # get end portion of true trajectory to compare to rollout
            real_traj = x[:,burn_in_length:,:].to("cpu")                
            assert rollout_x.size() == real_traj.size()
            real_trajs.append(real_traj)
            imag_trajs.append(rollout_x)
            # store the steps after pick up with predicted frames in imagined_trajs
            imagined_trajs[i,burn_in_length:,:] = rollout_x
                            
        input_data = input_data.cpu().detach().numpy()
        # result = {'Accuracy':_, 'Precision':_, 'Recall':_}
        result, obj_moved_flag, recon_moved_flag = self._eval_move_events(
            input_data, imagined_trajs, self.args.move_threshold)
        result['model'] = self.model_name
        ade, fde = self.calculate_ADE_FDE(real_trajs, imag_trajs)
        result['ADE'] = ade
        result['FDE'] = fde
        return result


    def eval_pickup_events(self, input_matrices, recon_matrices):
        def consecutive(data, stepsize=1):
            return np.split(data, np.where(np.diff(data) > stepsize)[0]+1)
        
        def detect_obj_pick_up(trial_agent_pos, trial_obj_pos):
            picked_up = False
            dropped = False
            # calculate how close agent is to object
            agent_obj_dist = np.array([distance.euclidean(trial_agent_pos[t,[0,2]], trial_obj_pos[t,[0,2]]) for t in range(len(trial_agent_pos))])
            # find inds where obj meet criteria
            # 1. obj is above y_thr
            # 2. obj is moving
            # 3. obj is close to agent
            # 4. largest y delta should be beginning or end of the sequence
            #y_thr = pick_up_y_thr[temp_obj_name]
            y_thr = 1e-3
            pick_up_inds = np.where((trial_obj_pos[:,1] > y_thr) & (trial_obj_pos[:,1] < 0.6))[0]
            #pick_up_event_inds = consecutive(pick_up_inds)
            obj_pos_delta = np.zeros(trial_obj_pos.shape)
            obj_pos_delta[1:,:] = np.abs(trial_obj_pos[1:,:] - trial_obj_pos[:-1,:])
            obj_pos_delta_sum = obj_pos_delta.sum(axis=1)
            obj_moving_inds = np.where(obj_pos_delta_sum > 1e-5)[0]
            agent_close_inds = np.where(agent_obj_dist < 0.8)[0]
            pick_up_move_close = reduce(np.intersect1d,(pick_up_inds, obj_moving_inds, agent_close_inds))
            pick_up_event_inds = consecutive(pick_up_move_close)
            for pick_up_event in pick_up_event_inds:
                if len(pick_up_event) > 5:
                    # largest y delta should be beginning or end of the sequence
                    obj_delta_event = obj_pos_delta[pick_up_event,:]
                    amax_delta = np.argmax(obj_delta_event[:,1])
                    index_percentage = amax_delta / len(obj_delta_event)
                    if index_percentage < 0.1 or index_percentage > 0.9:
                        picked_up = True
                        dropped = True
                    
            if picked_up and dropped:
                return True
            else:
                return False
            
        if hasattr(recon_matrices, 'requires_grad') and recon_matrices.requires_grad:
            recon_matrices = recon_matrices.detach().numpy()
            
        data_columns = self.data_columns 
        dims = ['x', 'y', 'z']
    
        num_trials = input_matrices.shape[0]
        num_objs = 3
        obj_pick_up_flag = np.zeros([num_trials, 3])
        for i in range(num_trials):
            trial_x = input_matrices[i,:,:]
            for j in range(num_objs):
                pos_inds = [data_columns.index('obj'+str(j)+'_'+dim) for dim in dims]
                trial_obj_pos = trial_x[:,pos_inds]
                agent_moved = []
                for k in range(2):
                    pos_inds2 = [data_columns.index('agent'+str(k)+'_'+dim) for dim in dims]
                    trial_agent_pos = trial_x[:,pos_inds2]
                    move_bool = detect_obj_pick_up(trial_agent_pos, trial_obj_pos)
                    agent_moved.append(move_bool)
                if any(agent_moved):
                    obj_pick_up_flag[i,j] = 1
                    
        recon_pick_up_flag = np.zeros([num_trials, 3])
        for i in range(num_trials):
            trial_x = recon_matrices[i,:,:]
            for j in range(num_objs):
                pos_inds = [data_columns.index('obj'+str(j)+'_'+dim) for dim in dims]
                trial_obj_pos = trial_x[:,pos_inds]
                agent_moved = []
                for k in range(2):
                    pos_inds2 = [data_columns.index('agent'+str(k)+'_'+dim) for dim in dims]
                    trial_agent_pos = trial_x[:,pos_inds2]
                    move_bool = detect_obj_pick_up(trial_agent_pos, trial_obj_pos)
                    agent_moved.append(move_bool)
                if any(agent_moved):
                    recon_pick_up_flag[i,j] = 1
                    
        pickup_idxs = np.where(obj_pick_up_flag)
        accuracy = np.mean(recon_pick_up_flag[pickup_idxs])
        
        return accuracy, obj_pick_up_flag, recon_pick_up_flag    
    
    
    def eval_pickup_events_in_rollouts(self, model, input_data, partial=1.0) -> Dict[str, Any]:
        if self.ds_num == 1:
            # first dataset
            pickup_timepoints = annotate_pickup_timepoints(self.loaded_dataset, train_or_val='val', pickup_or_move='move', ds_num=self.ds_num)
            goal_timepoints = annotate_goal_timepoints(self.loaded_dataset, train_or_val='val', ds_num=self.ds_num)
            single_goal_trajs = np.where((np.sum(pickup_timepoints > -1, axis=1) == 1))[0]
            multi_goal_trajs = np.where((np.sum(pickup_timepoints > -1, axis=1) == 3))[0]
        else:
            # 2+ dataset use event logger to define events
            pickup_timepoints = self.exp_info_dict[self.args.train_or_val]['pickup_timepoints']
            goal_timepoints = self.exp_info_dict[self.args.train_or_val]['goal_timepoints']
            single_goal_trajs = self.exp_info_dict[self.args.train_or_val]['single_goal_trajs']
            multi_goal_trajs = self.exp_info_dict[self.args.train_or_val]['multi_goal_trajs']
            
        # TO DO, ANALYZE EVERY PICKUP EVENT SEPARATELY, INCLUDING MULTI GOAL TRAJS
        #TEMP
        scores, y_labels, y_recon = self.eval_pickup_events(input_data, input_data)
        pickup_subset = pickup_timepoints[single_goal_trajs,:]
        indices = np.argwhere(pickup_subset > -1)
        accuracy = np.mean(y_recon[indices[:,0],indices[:,1]])
        breakpoint()
        
        single_goal_trajs = single_goal_trajs[:int(partial*len(single_goal_trajs))]
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
            # burn in until right before the pick up point
            if steps2pickup > 15:
                burn_in_length = steps2pickup - 10
            elif steps2pickup > 10:
                burn_in_length = steps2pickup - 5
            else:
                # TO DO - don't include trial if not enough burn in available
                burn_in_length = steps2pickup - 1
            # store the steps before pick up with real frames in imagined_trajs
            imagined_trajs[i,:burn_in_length,:] = x[:,:burn_in_length,:].cpu()
            # rollout model for the rest of the trajectory
            rollout_length = num_timepoints - burn_in_length
            rollout_x = model.forward_rollout(x.cuda(), burn_in_length, rollout_length).cpu().detach()
            # Replace any nan, inf, or outliers values with 0
            rollout_x[torch.isnan(rollout_x) | torch.isinf(rollout_x) | (torch.abs(rollout_x) > 1e3)] = 0
            # get end portion of true trajectory to compare to rollout
            real_traj = x[:,burn_in_length:,:].to("cpu")
            assert rollout_x.size() == real_traj.size()
            real_trajs.append(real_traj)
            imag_trajs.append(rollout_x)
            # store the steps after pick up with predicted frames in imagined_trajs
            imagined_trajs[i,burn_in_length:,:] = rollout_x
            
        full_trajs = input_data[single_goal_trajs,:,:].cpu()
        scores, y_labels, y_recon = self.eval_pickup_events(full_trajs, imagined_trajs)
        # evaluate whether only appropriate goals (after object picked up) are reconstructed
        pickup_subset = pickup_timepoints[single_goal_trajs,:]
        indices = np.argwhere(pickup_subset > -1)
        accuracy = np.mean(y_recon[indices[:,0],indices[:,1]])
        ade, fde = self.calculate_ADE_FDE(real_trajs, imag_trajs)
        result = {'model': self.model_name, 'score': accuracy, 'ADE': ade, 'FDE': fde}        
        return result
    
    def detect_object_move(self, trial_obj_pos, move_thr):
        obj_pos_delta = np.abs(trial_obj_pos[1:,:] - trial_obj_pos[:-1,:])
        obj_pos_delta_sum = obj_pos_delta.sum(axis=1)
        total_movement = obj_pos_delta_sum.sum()        
        if total_movement > move_thr:
            return 1
        else:
            return 0
    
    def get_obj_moved_flags(self, trajs, move_thr=0.1):
        """ Returns flags indicating whether each object moved more than move_thr in the trajectory """        
        data_columns = [
            'obj0_x', 'obj0_y', 'obj0_z', 'obj0_rot_x', 'obj0_rot_y', 'obj0_rot_z', 'obj0_rot_w', 
            'obj1_x', 'obj1_y', 'obj1_z', 'obj1_rot_x', 'obj1_rot_y', 'obj1_rot_z', 'obj1_rot_w', 
            'obj2_x', 'obj2_y', 'obj2_z', 'obj2_rot_x', 'obj2_rot_y', 'obj2_rot_z', 'obj2_rot_w', 
            'agent0_x', 'agent0_y', 'agent0_z', 'agent0_rot_x', 'agent0_rot_y', 'agent0_rot_z', 'agent0_rot_w', 
            'agent1_x', 'agent1_y', 'agent1_z', 'agent1_rot_x', 'agent1_rot_y', 'agent1_rot_z', 'agent1_rot_w'
            ]
        #dims = ['x', 'y', 'z', 'rot_x', 'rot_y', 'rot_z', 'rot_w']        
        dims = ['x', 'y', 'z']
        num_trials = trajs.shape[0]
        num_objs = 3
        obj_moved_flag = np.zeros([num_trials, 3])
        for i in range(num_trials):
            trial_x = trajs[i,:,:]
            for j in range(num_objs):
                pos_inds = [data_columns.index('obj'+str(j)+'_'+dim) for dim in dims]                
                trial_obj_pos = trial_x[:,pos_inds]
                obj_moved_flag[i,j] = self.detect_object_move(trial_obj_pos, move_thr)    
        return obj_moved_flag

    def compute_displacement_error(self, model, batch_size=None):        
        still_obj = self.args.still_obj
        print(f"Still obj: {still_obj}")
        total_trials, traj_length, _ = self.input_data.shape
        behavior_keys = list(self.exp_info_dict['val'].keys())[2: -5]
        burn_in_lengths = [50]    #, 100, 150, 150, 200, 250]        
        result = {}
        for burn_in_length in burn_in_lengths:
            print(f"Burn in length {burn_in_length}")            
            rollout_length = traj_length - burn_in_length        
            real_trajs = self.input_data[:, -rollout_length:, :]
            burn_ins = self.input_data[:, :burn_in_length, :]
            input_data = self.input_data
            result, obj_moved_flag, recon_moved_flag = self._eval_move_events(input_data, input_data, self.args.move_threshold)
            
            with torch.no_grad():
                if batch_size is None: 
                    with torch.no_grad():
                        rollout_x = model.forward_rollout(input_data.cuda(), burn_in_length, rollout_length).cpu()
                    # Replace any nan, inf, or outliers values with 0
                    rollout_x[torch.isnan(rollout_x) | torch.isinf(rollout_x) | (torch.abs(rollout_x) > 1e3)] = 0
                else:
                    rollout_x_list = []
                    move_flags = []
                    for i in tqdm(range(0, total_trials, batch_size)):
                        x = input_data[i: i+batch_size, :burn_in_length, :]
                        with torch.no_grad():
                            y = model.forward_rollout(x.cuda(), burn_in_length, rollout_length)
                        move_flags.append(self.get_obj_moved_flags(x))                        
                        rollout_x_list.append(y)
                        torch.cuda.empty_cache()
                    rollout_x = torch.cat(rollout_x_list, dim=0).cpu()
                    rollout_x[torch.isnan(rollout_x) | torch.isinf(rollout_x) | (torch.abs(rollout_x) > 1e3)] = 0                    
                    move_flags = torch.from_numpy(np.concatenate(move_flags, axis=0))   
                    if still_obj:
                        num_trials = rollout_x.shape[0]
                        num_objs = 3
                        dims = ['x', 'y', 'z']
                        displacement_by_time = []
                        total_disp_by_time = []
                        for i in range(num_trials):
                            trial_x = rollout_x[i,:,:]
                            real_x = real_trajs[i,:,:]
                            for j in range(num_objs):
                                if not obj_moved_flag[i,j]:
                                    pos_inds = [self.data_columns.index('obj'+str(j)+'_'+dim) for dim in dims]
                                    trial_obj_pos = trial_x[:,pos_inds]
                                    real_obj_pos = real_x[:,pos_inds]
                                    step_de = torch.norm(trial_obj_pos - real_obj_pos, p=2, dim=-1)
                                    displacement_by_time.append(step_de)
                                    # get the total delta from sum(d0, d1...dt) at each timepoint - what's used for detect_object_move and jittering
                                    first_disp = torch.norm(trial_obj_pos[0,:] - real_obj_pos[0,:], p=2, dim=-1)
                                    obj_pos_delta = torch.norm(trial_obj_pos[1:,:] - trial_obj_pos[:-1,:], p=2, dim=-1)
                                    # concatenate first disp with the rest of the displacements
                                    all_disp = torch.cat([first_disp.unsqueeze(0), obj_pos_delta], dim=0)
                                    total_disp_by_time = all_disp.cumsum(dim=0)
                        time_disp_array = torch.stack(displacement_by_time, dim=0)
                        avg_disp_by_time = time_disp_array.mean(dim=0)
                        result['all_trials'] = {'step_de': avg_disp_by_time, 'cum_disp': total_disp_by_time} 
                    elif still_obj and len(obj_moved_flag) == 0:                                                
                        # only compute displacement for still objects, and only compute displacement with positions
                        rollout_x = rollout_x.reshape(rollout_x.shape[0], rollout_x.shape[1], -1, 7)
                        rollout_x = rollout_x[:, :, :3, :3]
                        real_trajs = real_trajs.reshape(real_trajs.shape[0], real_trajs.shape[1], -1, 7)
                        real_trajs = real_trajs[:, :, :3, :3].detach().cpu()
                        move_flags = move_flags.unsqueeze(1).expand(-1, rollout_length, -1).cpu()
                        # get stable objects
                        rollout_x = rollout_x[move_flags == 0]
                        #rollout_x = rollout_x.reshape(-1, rollout_length, 9)
                        rollout_x = rollout_x.reshape(-1, rollout_length, 3)
                        real_trajs = real_trajs[move_flags == 0]
                        #real_trajs = real_trajs.reshape(-1, rollout_length, 9)
                        real_trajs = real_trajs.reshape(-1, rollout_length, 3)
                        # compute displacement error                        
                        step_de = torch.norm(rollout_x - real_trajs, p=2, dim=-1)
                        #breakpoint()
                        # for i in range(100):
                        #     outname = f"tmp_figs/{self.model_name}_step_de_{i}.png"                            
                        #     plt.plot(step_de[i])
                        #     plt.savefig(outname, dpi=800)
                        #     plt.close()
                        step_de = step_de.mean(dim=0)
                        # step_de = []
                        # for n_trial in tqdm(range(rollout_x.shape[0])):
                        #     trial_flags = move_flags[n_trial].reshape(-1)
                        #     trial_rollout = rollout_x[n_trial]
                        #     trial_real = real_trajs[n_trial]
                        #     trial_diffs = []
                        #     for n_step in range(rollout_length):
                        #         step_rollout = trial_rollout[n_step].reshape(-1)
                        #         step_real = trial_real[n_step].reshape(-1)  
                        #         step_rollout = step_rollout[trial_flags == 0]
                        #         step_real = step_real[trial_flags == 0]
                        #         diff = torch.norm(rollout_x - real_trajs, p=2, dim=-1)
                        #         trial_diffs.append(diff)
                        #     step_de.append(trial_diffs)
                        # step_de = np.array(step_de).mean(axis=0)
                        result['all_trials'] = {'step_de': step_de}                               
                    else:                        
                        # Replace any nan, inf, or outliers values with 0
                        step_de = torch.norm(rollout_x - real_trajs, p=2, dim=-1).mean(dim=0)
                        # compute average displacement error by computing euclidean distance between predicted and real trajectories
                        ade = torch.mean(torch.norm(rollout_x - real_trajs, p=2, dim=-1))
                        # compute final displacement error
                        fde = torch.mean(torch.norm(rollout_x[:, -1, :] - real_trajs[:, -1, :], p=2, dim=-1))
                        result['all_trials'] = {
                            'ade': ade, 'fde': fde, 'step_de': step_de}                    
                        print(f"Total trials used for computing displacement error: {rollout_x.size(0)}")
                        #rollout_x = rollout_x.reshape(rollout_x.size(0), -1)
                        #real_trajs = real_trajs.reshape(real_trajs.size(0), -1)
                        #behavior_result = {}
                        
                        # for behavior_key in behavior_keys:
                        #     behavior_idxs = self.exp_info_dict['val'][behavior_key]
                        #     behavior_rollout_x = rollout_x[behavior_idxs]
                        #     behavior_real_trajs = real_trajs[behavior_idxs]
                        #     # compute average displacement error by computing euclidean distance between predicted and real trajectories
                        #     ade = torch.mean(torch.norm(behavior_rollout_x - behavior_real_trajs, p=2, dim=-1))
                        #     # compute final displacement error
                        #     fde = torch.mean(torch.norm(behavior_rollout_x[:, -1, :] - behavior_real_trajs[:, -1, :], p=2, dim=-1))
                        #     result[behavior_key] = {'ade': ade, 'fde': fde}
                        #result[burn_in_length] = behavior_result
        result['model'] = self.model_name              
        return result

    def eval_one_model(self, model_key, epoch=None) -> Dict[str, Any]:        
        model = self.load_model(model_key, epoch)        
        result = {}
        result['Model'] = MODEL_DICT_VAL[model_key]['model_label']

        if self.args.eval_type == 'goal_events':
            result = self.eval_goal_events_in_rollouts(model, self.input_data, partial=self.args.partial)      
        elif self.args.eval_type == 'goal_events_level2':
            result = self.eval_goal_events_in_rollouts(model, self.input_data, level=2, partial=self.args.partial) 
        elif self.args.eval_type == 'multigoal_events':
            result = self.eval_multigoal_events_in_rollouts(model, self.input_data, partial=self.args.partial)
        elif self.args.eval_type == 'move_events':
            result = self.eval_move_events_in_rollouts(model, self.input_data, partial=self.args.partial)        
        elif self.args.eval_type == 'pickup_events':
            result = self.eval_pickup_events_in_rollouts(model, self.input_data, partial=self.args.partial)
        elif self.args.eval_type == 'displacement':    # mean/final displacement error
            if 'sgnet' in model_key:
                batch_size = 32
            else:
                batch_size = args.batch_size          
            result = self.compute_displacement_error(model, batch_size=batch_size)
        else:
            raise NotImplementedError(f'Evaluation type {self.args.eval_type} not implemented')    
        
        if self.args.append_results:
            # add more information for this analysis
            result['epoch'] = self.epoch
            result['dataset'] = self.args.dataset
        return result


    def eval_all_models(self) -> None:
        self.results = []
        print(f'>>>>> Evaluating {self.args.eval_type} for all models <<<<<')
        for model_key in self.args.model_keys:
            print(f'Currently evaluating model {model_key}')
            result = self.eval_one_model(model_key)
            print(f'result: {result}')
            self.results.append(result)
    

    def save_results(self) -> None:
        if args.save_file is None:
            save_file = f'eval_{self.args.eval_type}'
        else:
            save_file = args.save_file
        # if a training set eval, add suffix
        if self.args.train_or_val == 'train':
            save_file = save_file+'_train'
        result_save_dir = os.path.join(self.args.analysis_dir, 'results')
        if not os.path.exists(result_save_dir):
            os.makedirs(result_save_dir)
        if self.args.eval_type == 'displacement':
            if self.args.still_obj:
                save_path = os.path.join(result_save_dir, f'{save_file}_still_obj.pkl')
            else:
                save_path = os.path.join(result_save_dir, f'{save_file}.pkl')            
            print(f"Saving results to {save_path}")
            with open(save_path, 'wb') as f:
                pickle.dump(self.results, f)
        else:  
            save_path = os.path.join(result_save_dir, f'{save_file}.csv')
            df_results = pd.DataFrame(self.results)
            df_results.to_csv(save_path)

        if self.args.plot and self.args.eval_type != 'displacement':
            figure_save_dir = os.path.join(self.args.analysis_dir, 'results', 'figures')
            if not os.path.exists(figure_save_dir):
                os.makedirs(figure_save_dir)                
            plot_save_file = os.path.join(self.args.analysis_dir, 'results', 'figures', save_file)
            plot_eval_wm_results(df_results, self.args, plot_save_file)
            
        if self.args.append_results and self.args.partial == 1.0 and self.args.eval_type != 'displacement':
            all_results_file = os.path.join(result_save_dir, 'all_results_'+self.args.eval_type+'.csv')
            if os.path.exists(all_results_file):
                df_all_results = pd.read_csv(all_results_file, index_col=0)
                df_all_results = pd.concat([df_all_results, df_results], ignore_index=True)
                df_all_results.to_csv(all_results_file)
            else:
                df_results.to_csv(all_results_file)


    def sweep_checkpoints(self, args, start_epoch, end_epoch, interval):
        # Placeholder for storing results
        sweep_results = []
        self.results = []

        # Iterating through the checkpoints at specific intervals
        for epoch in range(start_epoch, end_epoch + 1, interval):
            print(f"==== Evaluating checkpoint at epoch {epoch} ====")
            
            # Iterating through the provided model keys
            for model_key in args.model_keys:
                # Loading model information
                model_info = MODEL_DICT_VAL[model_key]
                
                # Constructing the model file name based on the epoch
                model_dir = os.path.join(args.checkpoint_dir, 'models', model_info['model_dir'])
                model_file_name = os.path.join(model_dir, f"model_{epoch}.pth")  # Assuming this naming convention
                
                epoch_str = str(epoch)
                result = self.eval_one_model(model_key, epoch_str)
                
                # Storing the result
                sweep_results.append((epoch, model_key, result))
                print(f"Result: {result}")
        
        # Finding the checkpoint that maximizes evaluation accuracy
        optimal_result = max(sweep_results, key=lambda x: x[2]['score'])
        print(f"Optimal checkpoint found at epoch {optimal_result[0]} for model {optimal_result[1]} with accuracy {optimal_result[2]['score']}")
        self.results.append(optimal_result)


    def run(self) -> None:
        self.load_data()
        if self.args.sweep_checkpoints is not None:
            self.sweep_checkpoints(self.args, self.args.sweep_checkpoints[0], self.args.sweep_checkpoints[1], self.args.sweep_checkpoints[2])
            self.save_results()
        else:
            self.eval_all_models()
            self.save_results()


def load_args():
    parser = argparse.ArgumentParser()
    # general pipeline parameters
    parser.add_argument('--batch_size', type=int, action='store',
                        default=DEFAULT_VALUES['batch_size'],
                        help='Batch size')
    parser.add_argument('--eval_seed', type=int, 
                        default=DEFAULT_VALUES['eval_seed'], help='Random seed')
    parser.add_argument('--analysis_dir', type=str, action='store',
                        default=DEFAULT_VALUES['analysis_dir'], 
                        help='Analysis directory')
    parser.add_argument('--model_config_dir', type=str, action='store',
                        default=DEFAULT_VALUES['model_config_dir'],
                        help='Model config directory')   
    parser.add_argument('--data_dir', type=str,
                         default=DEFAULT_VALUES['data_dir'], 
                         help='Data directory')
    parser.add_argument('--dataset', type=str,
                         default=DEFAULT_VALUES['dataset'], 
                         help='Dataset name')
    parser.add_argument('--checkpoint_dir', type=str, action='store',
                        default=DEFAULT_VALUES['checkpoint_dir'], 
                        help='Checkpoint directory')
    parser.add_argument('--save_file', type=str,
                        default=None, 
                        help='Filename for saving model')
    parser.add_argument('--append_results', type=int, default=1, help='Append to master dataframe of results')
    parser.add_argument('--train_or_val', type=str, default='val', help='Training or Validation Set')
    parser.add_argument('--model_keys', nargs='+', action='store',
                        default=DEFAULT_VALUES['model_keys'], 
                        help='A list of keys, seperate by spaces, for all model to evaluate')
    parser.add_argument('--eval_type', type=str, action='store',
                        choices=DEFAULT_VALUES['eval_types'], 
                        default='goal_events', 
                        help='Type of evaluation to perform')
    parser.add_argument('--plot', type=int, default=1, help='Plot Results')
    parser.add_argument('--move_threshold', action='store',
                        default=DEFAULT_VALUES['move_threshold'],
                        help='Threshold for move event evaluation')
    parser.add_argument('--non_goal_burn_in', action='store',
                        default=DEFAULT_VALUES['non_goal_burn_in'],
                        help='Number of frames to burn in for non-goal events')
    parser.add_argument('--partial', type=float, default=1.0,         
                        help='Partial evaluation')
    parser.add_argument('--sweep_checkpoints', nargs=3, type=int, default=None, help='Sweep through checkpoints [start, end, interval]')
    parser.add_argument('--still_obj', action='store_true', default=False, help='Only evaluate still objects')
    return parser.parse_args()


if __name__ == "__main__":    
    args = load_args()
    analysis = Analysis(args)
    analysis.run()