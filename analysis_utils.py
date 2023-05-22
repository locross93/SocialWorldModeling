# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 12:00:58 2023

@author: locro
"""

import json
import os
import pickle
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

    
data_columns = ['obj0_x', 'obj0_y', 'obj0_z', 'obj1_x', 'obj1_y', 'obj1_z', 'obj2_x',
       'obj2_y', 'obj2_z', 'agent0_x', 'agent0_y', 'agent0_z', 'agent0_rot_x',
       'agent0_rot_y', 'agent0_rot_z', 'agent0_rot_w', 'agent1_x', 'agent1_y',
       'agent1_z', 'agent1_rot_x', 'agent1_rot_y', 'agent1_rot_z',
       'agent1_rot_w'] 


def load_config(file):
    with open(file) as f:
        config = json.load(f)
    return config


def get_highest_numbered_file(model_filename, directory):
    highest_number = -1
    highest_numbered_file = None
    for filename in os.listdir(directory):
        if model_filename in filename:
            prefix, number = filename.rsplit('_', 1) if '_epoch' not in filename else filename.rsplit('_epoch', 1)
            if prefix == model_filename and int(number) > highest_number:
                highest_number = int(number)
                highest_numbered_file = os.path.join(directory, filename)
    assert highest_number > -1,'No checkpoints found'
    return highest_numbered_file, highest_number


# @TODO plot functions here
# @TODO will need more work depending on the evaluation type
def plot_results(result_file):
    df_plot = pd.read_csv(result_file)
    # Plot the results using seaborn
    fig = plt.figure()
    sns.barplot(x='Model', y='Score', data=df_plot) 
    plt.title('Evaluate Single Goal Events From Forward Rollouts', fontsize=14)
    plt.xlabel('Model Name', fontsize=16) 
    plt.ylabel('Accuracy', fontsize=16)
    plt.ylim([0, 1])
    save_name = os.path.basename(result_file).split('.')[0]
    plt.savefig(f'{save_name}_acc.png', dpi=300, bbox_inches='tight')
    plt.show()

    fig = sns.barplot(x='Model', y='MSE', data=df_plot) 
    plt.title('Prediction Error of Goal Events in Imagination', fontsize=18)
    plt.xlabel('Model Name', fontsize=16) 
    plt.ylabel('MSE', fontsize=16)
    plt.savefig(f'{save_name}_mse.png', dpi=300, bbox_inches='tight')
