# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 12:00:58 2023

@author: locro
"""

import json
import os
import torch
import pickle
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns


""" Plot functions for individual evals """
# Plot goal  events
def plot_goal_events(df, save_file=None, sn_palette='Set2'):
    inds2plot = [25, 18, 28, 26, 22]
    df = df.loc[inds2plot]

    plt.figure()
    bar_plot = sns.barplot(x='model', y='score', data=df, palette=sn_palette)
    #bar_plot.set_xticklabels(bar_plot.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.title('Evaluate Single Goal Events', fontsize=14)
    plt.xlabel('Model Name', fontsize=13) 
    xlabels = ['RSSM Discrete', 'RSSM Continuous', 'Multistep Predictor', 'Multistep Delta', 'IRIS Transformer']
    plt.xticks(range(len(xlabels)), xlabels, fontsize=12, rotation=45, horizontalalignment='right')
    plt.ylabel('Accuracy', fontsize=16)    
    plt.ylim([0, 1])
    if save_file is not None:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')    
    plt.show()

def plot_multigoal_events(df, save_file=None, sn_palette='Set2'):
    inds2plot = [0, 1, 2, 4, 10]
    df = df.loc[inds2plot]  

    df_plot = pd.melt(df, id_vars="model", value_vars=["g2_acc", "g3_acc"], var_name="Goal Num", value_name="Accuracy")
    df_plot["Goal Num"] = df_plot["Goal Num"].replace({"g2_acc": "2nd Goal", "g3_acc": "3rd Goal"})

    plt.figure()
    bar_plot = sns.barplot(x='model', y='Accuracy', hue='Goal Num', data=df_plot, palette=sn_palette) 
    #bar_plot.set_xticklabels(bar_plot.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.title('Evaluate Multistep Goal Events', fontsize=14)
    plt.xlabel('Model Name', fontsize=16)
    xlabels = ['RSSM Discrete', 'RSSM Continuous', 'Multistep Predictor', 'Multistep Delta', 'IRIS Transformer']
    plt.xticks(range(len(xlabels)), xlabels, fontsize=12, rotation=45, horizontalalignment='right')
    plt.ylabel('Accuracy', fontsize=16)
    plt.ylim([0, 1])
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    sns.despine(top=True, right=True)
    if save_file is not None:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()

def plot_move_events(df, save_file=None, sn_palette='Set2'):    
    f1_score = 2 * (df['precision'] * df['recall']) / (df['precision'] + df['recall'])
    df['f1_score'] = f1_score
    df_plot = pd.melt(df, id_vars="model", value_vars=["f1_score", "precision", "recall"], var_name="Metric", value_name="Score")
    plt.figure()
    bar_plot = sns.barplot(x='Metric', y='Score', hue='model', data=df_plot, palette=sn_palette)
    bar_plot.set_xticklabels(bar_plot.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.title('Evaluate Move Events', fontsize=14)
    plt.xlabel('Metric', fontsize=16)    
    plt.ylabel('Score', fontsize=16)
    plt.ylim([0, 1])
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    if save_file is not None:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()


def plot_pickup_events(df, save_file=None, sn_palette='Set2'):
    inds2plot = [0, 1, 2, 4, 6]
    df = df.loc[inds2plot] 
    plt.figure()
    bar_plot = sns.barplot(x='model', y='score', data=df, palette=sn_palette) 
    bar_plot.set_xticklabels(bar_plot.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.title('Evaluate Pick Up Events', fontsize=14)
    plt.xlabel('Model Name', fontsize=16)  
    plt.ylabel('Accuracy', fontsize=16)

    xlabels = ['RSSM Discrete', 'RSSM Continuous', 'Multistep Predictor', 'Multistep Delta', 'IRIS Transformer']
    plt.xticks(range(len(xlabels)), xlabels, fontsize=12, rotation=45, horizontalalignment='right')

    plt.ylim([0, 1])
    if save_file is not None:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()

def plot_eval_wm_results(result_path, save_figs=True):
    event_type = result_path.split('_')[-2]
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