import argparse
import pickle
from constants_lc import DEFAULT_VALUES
import os
import torch
from torch.utils.data import TensorDataset, Subset

from constants_lc import DEFAULT_VALUES

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=DEFAULT_VALUES['data_dir'])
    parser.add_argument('--dataset', type=str, default=DEFAULT_VALUES['dataset'])
    parser.add_argument('--new_ds_size', type=int, default='New Dataset Size')
    return parser.parse_args()

if __name__ == '__main__':
    args = load_args()
    # load data
    dataset_file = os.path.join(args.data_dir, args.dataset)
    loaded_dataset = pickle.load(open(dataset_file, 'rb'))
    train_dataset, test_dataset = loaded_dataset
    train_data = train_dataset.dataset.tensors[0][train_dataset.indices,:,:]
    """ if args.train_or_val == 'train':
        input_data = train_dataset.dataset.tensors[0][train_dataset.indices,:,:]
    else:
        input_data = test_dataset.dataset.tensors[0][test_dataset.indices,:,:] """
    # load dataset info
    exp_info_file = dataset_file[:-4]+'_exp_info.pkl'
    if os.path.isfile(exp_info_file):
        with open(exp_info_file, 'rb') as f:
            exp_info_dict = pickle.load(f)
    else:
        print('DS info dict not found')

    keys2ignore = ['event_logs', 'exp_names', 'goal_timepoints', 'pickup_timepoints', 'indices', 'single_goal_trajs', 'multi_goal_trajs']

    # get fraction of new ds size from old ds size
    partial = args.new_ds_size / len(train_data)

    new_ds_inds = []
    # new exp_info_dict
    new_exp_info_dict = {}
    new_exp_info_dict['train'] = {}
    fewest_trial_num = args.new_ds_size
    for key in exp_info_dict['train'].keys():
        if key in keys2ignore:
            continue    

        trial_type_inds = exp_info_dict['train'][key]
        trial_type_inds = trial_type_inds[:int(partial*len(trial_type_inds))]
        new_ds_inds.extend(trial_type_inds)
        new_exp_info_dict['train'][key] = trial_type_inds

        if len(trial_type_inds) < fewest_trial_num:
            fewest_trial_num = len(trial_type_inds)
            fewest_trial_type = key

    # add more trials from fewest_trial_type to get to new_ds_size
    leftover_trials = args.new_ds_size - len(new_ds_inds)
    if leftover_trials > 0:
        trial_type_inds = exp_info_dict['train'][fewest_trial_type]
        trial_type_inds = trial_type_inds[int(partial*len(trial_type_inds)):(int(partial*len(trial_type_inds))+leftover_trials)]
        new_ds_inds.extend(trial_type_inds)
        new_exp_info_dict['train'][fewest_trial_type].extend(trial_type_inds)
    assert len(new_ds_inds) == args.new_ds_size, 'New DS size is not correct'

    # sort new_ds_inds
    new_ds_inds.sort()
    new_input_data = train_data[new_ds_inds,:,:]
    val_data = test_dataset.dataset.tensors[0][test_dataset.indices,:,:]
    # concat new_input_data and val_data
    new_data = torch.cat((new_input_data, val_data), dim=0)
    new_dataset = torch.utils.data.TensorDataset(new_data)
    train_idx = list(range(len(new_input_data)))
    val_idx = list(range(len(new_input_data), len(new_data)))
    train_dataset = Subset(new_dataset, train_idx) # a subset of the dataset with train indices
    test_dataset = Subset(new_dataset, val_idx) # a subset of the dataset with test indices
    new_exp_info_dict['train']['indices'] = train_idx
    new_exp_info_dict['train']['goal_timepoints'] = exp_info_dict['train']['goal_timepoints'][train_idx,:]
    new_exp_info_dict['train']['pickup_timepoints'] = exp_info_dict['train']['pickup_timepoints'][train_idx,:]
    new_exp_info_dict['train']['single_goal_trajs'] = [idx for idx in train_idx if idx in exp_info_dict['train']['single_goal_trajs']]
    new_exp_info_dict['train']['multi_goal_trajs'] = [idx for idx in train_idx if idx in exp_info_dict['train']['multi_goal_trajs']]
    new_exp_info_dict['val'] = exp_info_dict['val']

    save_file = dataset_file[:-4]+'_size'+str(args.new_ds_size)+'.pkl'
    full_dataset = [train_dataset, test_dataset]
    with open(save_file, 'wb') as handle:
        pickle.dump(full_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    exp_info_file = save_file[:-4]+'_exp_info.pkl'
    with open(exp_info_file, 'wb') as handle:
        pickle.dump(exp_info_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)