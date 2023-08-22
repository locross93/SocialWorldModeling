# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 15:08:12 2023

@author: locro
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import pickle
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import scipy.stats as stats

os.chdir('/Users/locro/Documents/Stanford/SocialWorldModeling')

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
    'SGNet 10': 'SGNet',
    'SGNet 10 S2': 'SGNet',
    'SGNet 10 S3': 'SGNet',
    }


# GOAL EVENTS
result_path = 'results/goal_events2plot.csv'
df = pd.read_csv(result_path, index_col=0)

result_path2 = 'results/sgnet_goal_events.csv'
df2 = pd.read_csv(result_path2, index_col=0)

save_file = 'results/figures/all_results_goal_events_submission'

#concat df and df2
df = pd.concat([df, df2], axis=0)

# Filtering rows where 'model' is in the keys of the model_keys dictionary
df_results = df[df['model'].isin(model_keys.keys())].copy()

# Replacing the 'model' values using the model_keys dictionary
df_results['model'] = df_results['model'].map(model_keys)

sn_palette='Set2'
title_fontsize=40
xtick_fontsize=18
label_fontsize=36
legend_fontsize=28
model_order = ['Hierarchical Oracle Model', 'Multistep Predictor', 'RSSM Discrete', 'RSSM Continuous', 'Multistep Delta', 'Transformer', 'SGNet']

# Split each model name into two lines if it contains two words
model_labels = ['\n'.join(model.split()) if len(model.split()) > 1 else model for model in model_order]

plt.figure(figsize=(13, 6))
bar_plot = sns.barplot(x='model', y='score', data=df_results, palette=sn_palette, errorbar="se", capsize=0.1, order=model_order)
sns.despine(top=True, right=True)    
#bar_plot.set_xticklabels(bar_plot.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title('Single Goal Events Evaluation', fontsize=title_fontsize)
#plt.title('Single Goal Events - Post Pickup Event Context', fontsize=30)
plt.xlabel('Model Name', fontsize=label_fontsize) 
#xlabels = ['Multistep Predictor', 'RSSM Discrete', 'RSSM Continuous', 'Multistep Delta', 'Transformer']
#plt.xticks(range(len(xlabels)), xlabels, fontsize=xtick_fontsize, rotation=0, horizontalalignment='center')
plt.ylabel('Accuracy', fontsize=label_fontsize)    
plt.ylim([0, 1])

# Set the x-tick labels with the modified model names
bar_plot.set_xticklabels(model_labels, fontsize=xtick_fontsize, rotation=0, horizontalalignment='center')
if save_file is not None:
    plt.savefig(save_file, dpi=300, bbox_inches='tight')    
plt.show()