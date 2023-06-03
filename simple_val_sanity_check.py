import os
import torch
import pickle
import numpy as np

from models import ReplayBuffer
from constants import DEFAULT_VALUES, MODEL_DICT_VAL, DATASET_NUMS
from analysis_utils import load_config, get_highest_numbered_file, get_data_columns, init_model_class


data_dir = DEFAULT_VALUES['data_dir']
dataset = "dataset_5_25_23.pkl"
dataset_file = os.path.join(data_dir, dataset)
loaded_dataset = pickle.load(open(dataset_file, 'rb'))
train_dataset, test_dataset = loaded_dataset
train_data = train_dataset.dataset.tensors[0][train_dataset.indices,:,:]
val_data = test_dataset.dataset.tensors[0][test_dataset.indices,:,:]

# load model
model_key = 'transformer_iris_default'
model_config_dir = DEFAULT_VALUES['model_config_dir']
model_info = MODEL_DICT_VAL[model_key]
model_name = model_info['model_label']
# set device, this assumes only one gpu is available to the script
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
config_file = os.path.join(model_config_dir, model_info['config'])
config = load_config(config_file)
model = init_model_class(config)
# sam
burn_in_length = 50
rollout_length = 30
sequence_length = burn_in_length + rollout_length
val_buffer = ReplayBuffer(sequence_length)
val_buffer.upload_training_set(val_data)
seed = 100 # set seed so every model sees the same randomization
val_batch_size = np.min([val_data.size(0), 1000])
val_trajs = val_buffer.sample(val_batch_size, random_seed=seed)
val_trajs = val_trajs.to(DEVICE)
val_trajs = val_trajs.to(torch.float32)

# eval on val data
model = model.to(DEVICE)
model.eval()
val_loss = model.loss(val_trajs)
print("val loss: {}".format(val_loss))