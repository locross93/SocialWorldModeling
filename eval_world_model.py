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
from sklearn.metrics import accuracy_score, precision_score, recall_score


from constants import DEFAULT_VALUES, MODEL_DICT_VAL, DATASET_NUMS
from analysis_utils import load_config, get_highest_numbered_file, get_data_columns, init_model_class
from annotate_pickup_timepoints import annotate_pickup_timepoints
from annotate_goal_timepoints import eval_recon_goals, annotate_goal_timepoints
from plot_eval_world_model import plot_eval_wm_results


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
        if self.args.dataset == 'train_test_splits_3D_dataset.pkl' or self.args.dataset == 'data_norm_velocity.pkl':
            self.ds_num = 1
        else:
            self.ds_num = 2
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


    def load_model(self, model_key) -> torch.nn.Module:
        """
        Load the trained model from the checkpoint directory
        """
        model_info = MODEL_DICT_VAL[model_key]
        self.model_name = model_info['model_label']        
        # set device, this assumes only one gpu is available to the script
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        setattr(args, 'device', DEVICE)
        config_file = os.path.join(self.args.model_config_dir, model_info['config'])
        config = load_config(config_file)        
        model = init_model_class(config, args)
            
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
            print('Loading from last checkpoint',latest_checkpoint)
            self.epoch = latest_checkpoint
            model.load_state_dict(torch.load(latest_checkpoint))

        model.eval()
        model.to(DEVICE)
        return model


    def eval_goal_events_in_rollouts(self, model, input_data, level=1, partial=1.0) -> Dict[str, typing.Any]:
        if self.ds_num == 1:
            # first dataset
            pickup_timepoints = annotate_pickup_timepoints(self.loaded_dataset, train_or_val='val', pickup_or_move='move', ds_num=self.ds_num)
            single_goal_trajs = np.where((np.sum(pickup_timepoints > -1, axis=1) == 1))[0]
        else:
            # 2+ dataset use event logger to define events
            pickup_timepoints = self.exp_info_dict[self.args.train_or_val]['pickup_timepoints']
            single_goal_trajs = self.exp_info_dict[self.args.train_or_val]['single_goal_trajs']
        
        single_goal_trajs = single_goal_trajs[:int(partial*len(single_goal_trajs))]
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
            if level == 2:
                # burn in to several frames before pick up point
                if steps2pickup > 15:
                    steps2pickup = steps2pickup - 15
                elif steps2pickup > 10:
                    steps2pickup = steps2pickup - 5
                else:
                    # TO DO - don't include trial if not enough burn in available
                    steps2pickup = steps2pickup - 1
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
        scores, y_labels, y_recon = eval_recon_goals(full_trajs, imagined_trajs, final_location=False, plot=False, ds_num=self.ds_num)
        # evaluate whether only appropriate goals (after object picked up) are reconstructed
        pickup_subset = pickup_timepoints[single_goal_trajs,:]
        indices = np.argwhere(pickup_subset > -1)
        accuracy = np.mean(y_recon[indices[:,0],indices[:,1]])
        print(np.where(y_recon[indices[:,0],indices[:,1]])[0])
        result = {'model': self.model_name, 'score': accuracy, 'MSE': mse}        
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
        print('Goal 2 successes:',np.where(goals_obj2)[0])
        acc_g3 = np.mean(goals_obj3)
        print('Goal 3 successes:',np.where(goals_obj3)[0])

        
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
            
        if hasattr(recon_matrices, 'requires_grad') and recon_matrices.requires_grad:
            recon_matrices = recon_matrices.detach().numpy()
            
        scores = {}
        
        dims = ['x', 'y', 'z']

        num_trials = input_matrices.shape[0]
        num_objs = 3
        obj_moved_flag = np.zeros([num_trials, 3])
        # hard threshold to detect movement (unchangable by user)
        move_thr_true = 0.1
        for i in range(num_trials):
            trial_x = input_matrices[i,:,:]
            for j in range(num_objs):
                pos_inds = [self.data_columns.index('obj'+str(j)+'_'+dim) for dim in dims]
                trial_obj_pos = trial_x[:,pos_inds]
                obj_moved_flag[i,j] = detect_object_move(trial_obj_pos, move_thr_true)
                
        recon_moved_flag = np.zeros([num_trials, 3])
        for i in range(num_trials):
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
    
    def eval_move_events_in_rollouts(self, model, input_data) -> Dict[str, Any]:
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
        for i in range(input_data.shape[0]):
            if i%50 == 0:
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
            # store the steps after pick up with predicted frames in imagined_trajs
            imagined_trajs[i,burn_in_length:,:] = rollout_x
                            
        input_data = input_data.cpu().detach().numpy()
        # result = {'Accuracy':_, 'Precision':_, 'Recall':_}
        result, obj_moved_flag, recon_moved_flag = self._eval_move_events(
            input_data, imagined_trajs, self.args.move_threshold)
        result['model'] = self.model_name
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
                    index_percentage = amax_delta / len(obj_delta_event) * 100
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
    
    
    def eval_pickup_events_in_rollouts(self, model, input_data) -> Dict[str, Any]:
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
        result = {'model': self.model_name, 'score': accuracy}        
        return result

    def compute_displacement_error(self, model, batch_size=None):
        total_trials, traj_length, _ = self.input_data.shape
        behavior_keys = list(self.exp_info_dict['val'].keys())[2: -5]
        burn_in_lengths = [50]    #, 100, 150, 150, 200, 250]        
        result = {}        
        for burn_in_length in burn_in_lengths:
            print(f"Burn in length {burn_in_length}")            
            rollout_length = traj_length - burn_in_length        
            real_trajs = self.input_data[:, -rollout_length:, :]
            with torch.no_grad():
                if batch_size is None: 
                    rollout_x = model.forward_rollout(real_trajs.cuda(), burn_in_length, rollout_length).cpu()
                else:
                    rollout_x = []
                    for i in range(0, total_trials, batch_size):
                        x = real_trajs[i: i+batch_size, :, :]
                        y = model.forward_rollout(x.cuda(), burn_in_length, rollout_length).cpu()
                        rollout_x.append(y)
                        torch.cuda.empty_cache()
                    rollout_x = torch.cat(rollout_x, dim=0)

                    #rollout_x = rollout_x.reshape(rollout_x.size(0), -1)
                    #real_trajs = real_trajs.reshape(real_trajs.size(0), -1)
                    behavior_result = {}
                    for behavior_key in behavior_keys:
                        behavior_idxs = self.exp_info_dict['val'][behavior_key]
                        behavior_rollout_x = rollout_x[behavior_idxs]
                        behavior_real_trajs = real_trajs[behavior_idxs]                        
                        # compute mean displacement error by computing euclidean distance between predicted and real trajectories
                        mde = torch.mean(torch.norm(behavior_rollout_x - behavior_real_trajs, p=2, dim=-1))
                        # compute final displacement error
                        fde = torch.mean(torch.norm(behavior_rollout_x[:, -1] - behavior_real_trajs[:, -1], p=2, dim=-1))
                        behavior_result[behavior_key] = {'mde': mde, 'fde': fde}
                    result[burn_in_length] = behavior_result        
        result['model'] = self.model_name              
        return result

    def eval_one_model(self, model_key) -> Dict[str, Any]:        
        model = self.load_model(model_key)        
        result = {}
        result['Model'] = MODEL_DICT_VAL[model_key]['model_label']

        if self.args.eval_type == 'goal_events':
            result = self.eval_goal_events_in_rollouts(model, self.input_data, partial=self.args.partial) 
        elif self.args.eval_type == 'multigoal_events':
            result = self.eval_multigoal_events_in_rollouts(model, self.input_data, partial=self.args.partial)
        elif self.args.eval_type == 'move_events':
            result = self.eval_move_events_in_rollouts(model, self.input_data)        
        elif self.args.eval_type == 'pickup_events':
            result = self.eval_pickup_events_in_rollouts(model, self.input_data)
        elif self.args.eval_type == 'displacement':    # mean/final displacement error
            if  'sgnet' in model_key:
                batch_size = 32
            else:
                batch_size = 2048          
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
            save_path = os.path.join(result_save_dir, f'{save_file}.pkl')
            breakpoint()
            with open(save_path, 'wb') as f:
                pickle.dump(self.results, f)
        else:  
            save_path = os.path.join(result_save_dir, f'{save_file}.csv')
            df_results = pd.DataFrame(self.results)
            df_results.to_csv(save_path)

        if self.args.plot:
            figure_save_dir = os.path.join(self.args.analysis_dir, 'results', 'figures')
            if not os.path.exists(figure_save_dir):
                os.makedirs(figure_save_dir)                
            plot_save_file = os.path.join(self.args.analysis_dir, 'results', 'figures', save_file)
            plot_eval_wm_results(df_results, self.args, plot_save_file)     
        if self.args.append_results and self.args.partial == 1.0:
            all_results_file = os.path.join(result_save_dir, 'all_results_'+self.args.eval_type+'.csv')
            if os.path.exists(all_results_file):
                df_all_results = pd.read_csv(all_results_file, index_col=0)
                df_all_results = pd.concat([df_all_results, df_results], ignore_index=True)
                df_all_results.to_csv(all_results_file)
            else:
                df_results.to_csv(all_results_file)


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
    parser.add_argument('--plot', type=bool, default=True, help='Plot Results')
    parser.add_argument('--move_threshold', action='store',
                        default=DEFAULT_VALUES['move_threshold'],
                        help='Threshold for move event evaluation')
    parser.add_argument('--non_goal_burn_in', action='store',
                        default=DEFAULT_VALUES['non_goal_burn_in'],
                        help='Number of frames to burn in for non-goal events')
    parser.add_argument('--partial', type=float, default=1.0,         
                        help='Partial evaluation')        
    return parser.parse_args()




if __name__ == "__main__":    
    args = load_args()
    analysis = Analysis(args)
    analysis.run()
