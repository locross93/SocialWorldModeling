# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 12:00:58 2023

@author: locro
"""

import json
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import scipy.stats as stats

title_fontsize=28
xtick_fontsize=12
label_fontsize=20
legend_fontsize=40
""" Plot functions for individual evals """
# Plot goal  events
def plot_goal_events(df, save_file=None, sn_palette='Set2', 
                     title_fontsize=40, xtick_fontsize=22,
                     label_fontsize=36, legend_fontsize=28):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='base_model', y='score', data=df, palette=sn_palette, errorbar='se', capsize=0.1)
    sns.despine(top=True, right=True)    
    #bar_plot.set_xticklabels(bar_plot.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.title('Single Goal Events Evaluation', fontsize=title_fontsize)
    plt.xlabel('Model Name', fontsize=label_fontsize) 
    xlabels = ['Multistep Predictor', 'RSSM Discrete', 'RSSM Continuous', 'Multistep Delta', 'Transformer']
    plt.xticks(range(len(xlabels)), xlabels, fontsize=xtick_fontsize, rotation=0, horizontalalignment='center')
    plt.ylabel('Accuracy', fontsize=label_fontsize)    
    plt.ylim([0, 1])
    if save_file is not None:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')    
    plt.show()

def plot_multigoal_events(df, save_file=None, sn_palette='Set2',
                          title_fontsize=40, xtick_fontsize=22,
                          label_fontsize=36, legend_fontsize=28):
     
    df_plot = pd.melt(df, id_vars="base_model", value_vars=["g2_acc", "g3_acc"], var_name="Goal Num", value_name="Accuracy")
    df_plot["Goal Num"] = df_plot["Goal Num"].replace({"g2_acc": "2nd Goal", "g3_acc": "3rd Goal"})

    plt.figure(figsize=(10, 6))
    sns.barplot(x='base_model', y='Accuracy', hue='Goal Num', data=df_plot,
                palette=sn_palette, errorbar='se', capsize=0.1)    
    sns.despine(top=True, right=True)
    plt.title('Multistep Goal Events Evaluation', fontsize=title_fontsize)
    plt.xlabel('Model Name', fontsize=label_fontsize)
    xlabels = ['Multistep Predictor', 'RSSM Discrete', 'RSSM Continuous', 'Multistep Delta', 'Transformer']
    plt.xticks(range(len(xlabels)), xlabels, fontsize=xtick_fontsize, rotation=0, horizontalalignment='center')
    plt.ylabel('Accuracy', fontsize=label_fontsize)
    plt.ylim([0, 1])
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=legend_fontsize)
    sns.despine(top=True, right=True)
    if save_file is not None:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()

def plot_move_events(df, save_file=None, sn_palette='Set2',
                     title_fontsize=40, xtick_fontsize=22,
                     label_fontsize=36, legend_fontsize=28):
    # print(df)
    xlabels = ['F1-Score', 'Precision', 'Recall']    
    df['base_model'] = df['base_model'].replace('MP', 'Multistep Predictor')
    df['base_model'] = df['base_model'].replace('RSSM', 'RSSM Discrete')
    df['base_model'] = df['base_model'].replace('RSSM_Cont', 'RSSM Continuous')
    df['base_model'] = df['base_model'].replace('MD', 'Multistep Delta')
    df['base_model'] = df['base_model'].replace('TF_Emb2048', 'Transformer')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Metric', y='Score', hue='base_model', data=df, 
                palette=sn_palette, errorbar='se', capsize=0.1)
    sns.despine(top=True, right=True)
    plt.title('Move Events Evaluation', fontsize=title_fontsize)
    plt.xlabel('Metric', fontsize=label_fontsize)
    plt.xticks(range(len(xlabels)), xlabels, fontsize=xtick_fontsize, rotation=0, horizontalalignment='center')
    plt.ylabel('Score', fontsize=label_fontsize)
    plt.ylim([0, 1])
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', fontsize=legend_fontsize)
    if save_file is not None:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()


def plot_pickup_events(df, save_file=None, sn_palette='Set2',
                       title_fontsize=40, xtick_fontsize=22,
                       label_fontsize=36, legend_fontsize=28):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='base_model', y='score', data=df, palette=sn_palette, 
                errorbar='se', capsize=0.1)    
    sns.despine(top=True, right=True)
    plt.title('Pick Up Events Evaluation', fontsize=title_fontsize)
    plt.xlabel('Model Name', fontsize=label_fontsize)  
    plt.ylabel('Accuracy', fontsize=label_fontsize)

    xlabels = ['Multistep Predictor', 'RSSM Discrete', 'RSSM Continuous', 'Multistep Delta', 'Transformer']
    plt.xticks(range(len(xlabels)), xlabels, fontsize=xtick_fontsize, rotation=0, horizontalalignment='center')

    plt.ylim([0, 1])
    if save_file is not None:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()

def process_plot_data(result_path):
    event_type = result_path.split('_')[-2]
    df = pd.read_csv(result_path, index_col=0)
    df['base_model'] = df['model'].apply(lambda x: '_'.join(x.split('-')[: -1]))#x.split('-')[0])

    if event_type == 'goal':
        df = df        
    elif event_type == 'multigoal':
        df = df
    elif event_type == 'move':
        f1_score = 2 * (df['precision'] * df['recall']) / (df['precision'] + df['recall'])
        df['f1_score'] = f1_score
        df = pd.melt(df, id_vars="base_model", value_vars=["f1_score", "precision", "recall"], var_name="Metric", value_name="Score")
    elif event_type == 'pickup':
        df = df
    return df


def plot_eval_wm_results(result_path, save_figs=True, title_fontsize=title_fontsize, 
                     xtick_fontsize=xtick_fontsize, 
                     label_fontsize=label_fontsize,
                     legend_fontsize=label_fontsize):
    event_type = result_path.split('_')[-2]
    #df = pd.read_csv(result_path, index_col=0)
    df = process_plot_data(result_path)
    if save_figs:
        save_file = result_path.split('.')[0] + '.png'
    else:
        save_file = None

    if event_type == 'goal':
        plot_goal_events(df, save_file, title_fontsize=title_fontsize, 
                     xtick_fontsize=xtick_fontsize, 
                     label_fontsize=label_fontsize,
                     legend_fontsize=label_fontsize)       
    elif event_type == 'multigoal':
        plot_multigoal_events(df, save_file, sn_palette='Paired',
                              title_fontsize=title_fontsize, 
                              xtick_fontsize=xtick_fontsize, 
                              label_fontsize=label_fontsize,
                              legend_fontsize=label_fontsize)
    elif event_type == 'move':
        plot_move_events(df, save_file, title_fontsize=title_fontsize, 
                     xtick_fontsize=xtick_fontsize, 
                     label_fontsize=label_fontsize,
                     legend_fontsize=label_fontsize)
    elif event_type == 'pickup':
        plot_pickup_events(df, save_file, title_fontsize=title_fontsize, 
                     xtick_fontsize=xtick_fontsize, 
                     label_fontsize=label_fontsize,
                     legend_fontsize=label_fontsize)

#goal_path = 'results/all_results_goal_events.csv'
#result_path = 'results/all_results_goal_events_submission.csv'
#result_path = 'results/goal_events_10_more_burnin.csv'
result_path = 'results/all_results_multigoal_events_submission.csv'
event_type = result_path.split('_')[-2]
#df = pd.read_csv(result_path, index_col=0)
df = process_plot_data(result_path)

model_keys = {
    'GT End State S1' : 'Hierarchical Oracle Model',
    'GT End State S2' : 'Hierarchical Oracle Model',
    'GT End State S3' : 'Hierarchical Oracle Model',
    'MP-S1' : 'Multistep Predictor',
    'MP-S2' : 'Multistep Predictor',
    'MP-S3' : 'Multistep Predictor',
    'RSSM-S1' : 'RSSM Discrete',
    'RSSM-S2' : 'RSSM Discrete',
    'RSSM-S3' : 'RSSM Discrete',
    'RSSM-Cont-S1' : 'RSSM Continuous',
    'RSSM-Cont-S2' : 'RSSM Continuous',
    'RSSM-Cont-S3' : 'RSSM Continuous',
    'MD-S1' : 'Multistep Delta',
    'MD-S2' : 'Multistep Delta',
    'MD-S3' : 'Multistep Delta',
    'TF-Emb2048-S1' : 'Transformer',
    'TF-Emb2048-S2' : 'Transformer',
    'TF-Emb2048-S3' : 'Transformer',
    'TF Emb2048 S1' : 'Transformer',
    'TF Emb2048 S2' : 'Transformer',
    'TF Emb2048 S3' : 'Transformer',
    }


# Filtering rows where 'model' is in the keys of the model_keys dictionary
df_results = df[df['model'].isin(model_keys.keys())].copy()

# Replacing the 'model' values using the model_keys dictionary
df_results['model'] = df_results['model'].map(model_keys)

save_figs = True
if save_figs:
    save_file = result_path.split('.')[0] + '.png'
else:
    save_file = None

plot_eval_wm_results(result_path, title_fontsize=title_fontsize, 
                     xtick_fontsize=xtick_fontsize, 
                     label_fontsize=label_fontsize,
                     legend_fontsize=label_fontsize)

# multigoal_path = 'results/all_results_multigoal_events.csv'
# plot_eval_wm_results(multigoal_path)
# #%%

# #%%
# move_path = 'results/all_results_move_events.csv'
# plot_eval_wm_results(move_path)
# #%%

# #%%
# pickup_path = 'results/all_results_pickup_events.csv'
# plot_eval_wm_results(pickup_path)