# -*- coding: utf-8 -*-
"""
Created on Mon May 15 18:03:38 2023

@author: locro
"""

import numpy as np
import os
import pandas as pd
import pickle
import platform
from pytorch_forecasting.models import RecurrentNetwork
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import TorchNormalizer, MultiNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

if platform.system() == 'Windows':
    # We are running on Windows
    analysis_dir = '/Users/locro/Documents/Stanford/analysis/'
    checkpoint_dir = analysis_dir
elif platform.system() == 'Linux':
    # We are running on Linux
    analysis_dir = '/home/locross/analysis/'
    checkpoint_dir = '/mnt/fs2/locross/analysis/'
    
os.chdir(analysis_dir)
    
from analysis_utils import data_columns

burn_in_length = 50
rollout_length = 30
if platform.system() == 'Windows':
    batch_size = 64
elif platform.system() == 'Linux':
    batch_size = 128
print(f'Batch size: {batch_size}')

model_type = 'tft_model'
model_filename = model_type
print('Starting', model_filename)

# load data
data_file = analysis_dir+'data/train_test_splits_3D_dataset.pkl'
with open(data_file, 'rb') as f:
    loaded_dataset = pickle.load(f)
train_dataset, test_dataset = loaded_dataset
train_data = train_dataset.dataset.tensors[0][train_dataset.indices,:,:]
val_data = test_dataset.dataset.tensors[0][test_dataset.indices,:,:]

# make training data
df_train = pd.DataFrame(train_data.view(train_data.size(0)*train_data.size(1),train_data.size(2)), columns=data_columns)

trial_column = np.repeat(np.arange(train_data.size(0)), train_data.size(1))
trial_column = pd.Series(trial_column, name="Trial #")
df_train["Trial #"] = trial_column

time_column = np.tile(np.arange(train_data.size(1)), train_data.size(0))
time_column = pd.Series(time_column, name="Trial #")
df_train["time_idx"] = time_column

# create the dataset from the pandas dataframe
train_dataset = TimeSeriesDataSet(
    df_train,
    group_ids=["Trial #"],
    target=data_columns,
    time_idx="time_idx",
    min_encoder_length=burn_in_length,
    max_encoder_length=burn_in_length,
    min_prediction_length=rollout_length,
    max_prediction_length=rollout_length,
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=data_columns,
    target_normalizer=MultiNormalizer([TorchNormalizer(method="identity") for _ in data_columns])
)

# convert the dataset to a dataloader
train_dataloader = train_dataset.to_dataloader(train=True, batch_size=batch_size)

# make validation data
df_val = pd.DataFrame(val_data.view(val_data.size(0)*val_data.size(1),val_data.size(2)), columns=data_columns)

trial_column = np.repeat(np.arange(val_data.size(0)), val_data.size(1))
trial_column = pd.Series(trial_column, name="Trial #")
df_val["Trial #"] = trial_column

time_column = np.tile(np.arange(val_data.size(1)), val_data.size(0))
time_column = pd.Series(time_column, name="Trial #")
df_val["time_idx"] = time_column

# create the dataset from the pandas dataframe
val_dataset = TimeSeriesDataSet(
    df_val,
    group_ids=["Trial #"],
    target=data_columns,
    time_idx="time_idx",
    min_encoder_length=burn_in_length,
    max_encoder_length=burn_in_length,
    min_prediction_length=rollout_length,
    max_prediction_length=rollout_length,
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=data_columns,
    target_normalizer=MultiNormalizer([TorchNormalizer(method="identity") for _ in data_columns])
)

# convert the dataset to a dataloader
val_dataloader = val_dataset.to_dataloader(train=False, batch_size=batch_size)

# Train model
# configure network and trainer
#early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=50, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
log_dir = checkpoint_dir+'models/training_info/tensorboard/'+model_filename
logger = TensorBoardLogger(log_dir)  # logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs=5000,
    gpus=1,
    enable_model_summary=True,
    gradient_clip_val=0.1,
    limit_train_batches=0.1,  # coment in for training, running valiation every 30 batches
    limit_val_batches=0.1,
    # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
    callbacks=[lr_logger],
    #callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)


tft = TemporalFusionTransformer.from_dataset(
    train_dataset,
    learning_rate=0.03,
    hidden_size=256,
    attention_head_size=2,
    dropout=0.1,
    hidden_continuous_size=8,
    #output_size=7,  # 7 quantiles by default
    loss=QuantileLoss(),
    log_interval=1,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    #reduce_on_plateau_patience=4,
)

print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# fit network
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
