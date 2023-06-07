# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 12:00:58 2023

@author: locro
"""
#%%
import json
import os
import torch
import pickle
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

from constants import MODEL_DICT_TRAIN
#%%



#%%
""" Plot functions for the paper """
# Plot goal  events
def plot_goal_events(df, save_file, sn_palette='Set2'):
    plt.figure()
    sns.barplot(x='model', y='score', data=df, palette=sn_palette)
    plt.title('Evaluate Single Goal Events From Forward Rollouts', fontsize=14)
    plt.xlabel('Model Name', fontsize=16) 
    plt.ylabel('Accuracy', fontsize=16)
    plt.ylim([0, 1])
    if save_file is not None:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()

def plot_multigoal_events(df, save_file, sn_palette='Set2'):
    df_plot = pd.melt(df, id_vars="model", value_vars=["g2_acc", "g3_acc"], var_name="Goal Num", value_name="Accuracy")
    df_plot["Goal Num"] = df_plot["Goal Num"].replace({"g2_acc": "2nd Goal", "g3_acc": "3rd Goal"})

    plt.figure()
    sns.barplot(x='model', y='Accuracy', hue='Goal Num', data=df_plot, palette=sn_palette) 
    plt.title('Evaluate Multistep Goal Events From Forward Rollouts', fontsize=14)
    plt.xlabel('Model Name', fontsize=16) 
    plt.ylabel('Accuracy', fontsize=16)
    plt.ylim([0, 1])
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    if save_file is not None:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()

def plot_move_events(df, save_file, sn_palette='Set2'):    
    f1_score = 2 * (df['precision'] * df['recall']) / (df['precision'] + df['recall'])
    df['f1_score'] = f1_score
    df_plot = pd.melt(
        df, id_vars="model", value_vars=["f1_score", "precision", "recall"], 
        var_name="Metric", value_name="Score")

    plt.figure()
    sns.barplot(x='Metric', y='Score', hue='model', data=df_plot, palette=sn_palette) 
    plt.title('Evaluate Move Events From Forward Rollouts', fontsize=14)
    plt.xlabel('Metric', fontsize=16) 
    plt.ylabel('Score', fontsize=16)
    plt.ylim([0, 1])
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    if save_file is not None:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()


def plot_pickup_events(df, save_file, sn_palette='Set2'):
    plt.figure()
    sns.barplot(x='model', y='score', data=df, palette=sn_palette) 
    plt.title('Evaluate Pick Up Events From Forward Rollouts', fontsize=14)
    plt.xlabel('Model Name', fontsize=16)  
    plt.ylabel('Accuracy', fontsize=16)
    plt.ylim([0, 1])
    if save_file is not None:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()


def plot_eval_wm_results(result_path, save_figs=True):
    event_type = result_path.split('_')[1]
    df = pd.read_csv(result_path, index_col=0)
    if save_figs:
        save_file = result_path.split('.')[0] + '.png'
    else:
        save_file = None

    if event_type == 'goal':
        plot_goal_events(df, save_file)       
    elif event_type == 'multigoal':
        plot_multigoal_events(df, save_file)
    elif event_type == 'move':
        plot_move_events(df, save_file)
    elif event_type == 'pickup':
        plot_pickup_events(df, save_file)
#%%

#%%
result_path = 'results/eval_goal_events.csv'
plot_eval_wm_results(result_path)
#%%

#%%
result_path = 'results/eval_multigoal_events.csv'
plot_eval_wm_results(result_path)
#%%

#%%
result_path = 'results/eval_move_events.csv'
plot_eval_wm_results(result_path)
#%%
#%%
def plot_displacement_results(result_path, save_figs=True):
    results = pickle.load(open(result_path, 'rb'))    
    return 0
#%%