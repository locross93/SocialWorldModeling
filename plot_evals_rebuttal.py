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

# Function to compute the mean and standard error for each model
def compute_mean_std_err(df, value_column):
    mean_std_err_results = df.groupby('model')[value_column].agg(['mean', 'std'])
    mean_std_err_results['StdErr'] = mean_std_err_results['std'] / (3 ** 0.5) # Assuming 3 instances
    mean_std_err_results.reset_index(inplace=True)
    mean_std_err_results.rename(columns={'model': 'Model', 'mean': 'Mean'}, inplace=True)
    mean_std_err_results.drop(columns=['std'], inplace=True)
    return mean_std_err_results

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

# # GOAL EVENTS + 5
# result_path = 'results/goals_5_more_submission_full.csv'
# df = pd.read_csv(result_path, index_col=0)

# save_file = 'results/figures/all_results_goal_events_submission_5_more'

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
#plt.title('Single Goal Events Evaluation', fontsize=title_fontsize)
plt.title('Single Goal Events - Post Pickup Event Context', fontsize=30)
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

# get ade_df and fde_df that only take model and ade, fde columns
ade_df = df_results[['model', 'ADE']].copy()
ade_df_results = ade_df.groupby('model')['ADE'].agg(['mean', 'std'])
fde_df = df_results[['model', 'FDE']].copy()
fde_df_results = fde_df.groupby('model')['FDE'].agg(['mean', 'std'])

# Define the desired order of behaviors
model_order = [
    'Multistep Predictor',
    'RSSM Discrete',
    'RSSM Continuous',
    'Multistep Delta',
    'Transformer',
    'SGNet',
    'Hierarchical Oracle Model'
]

# Reorder the DataFrame based on the specified sequence of behaviors
ade_df_results = pd.concat([ade_df_results[ade_df_results.index == model] for model in model_order])
fde_df_results = pd.concat([fde_df_results[fde_df_results.index == model] for model in model_order])

ade_df_results.to_csv('results/eval_goal_events_ade.csv')
fde_df_results.to_csv('results/eval_goal_events_fde.csv')

# MULTI GOAL EVENTS
result_path = 'results/mgevents_submission_full.csv'
df_results = pd.read_csv(result_path, index_col=0)

save_file = 'results/figures/all_results_multigoal_events_submission'

df_plot = pd.melt(df_results, id_vars="model", value_vars=["g2_acc", "g3_acc"], var_name="Goal Num", value_name="Accuracy")
df_plot["Goal Num"] = df_plot["Goal Num"].replace({"g2_acc": "2nd Goal", "g3_acc": "3rd Goal"})

# Replacing the 'model' values using the model_keys dictionary
df_plot['model'] = df_plot['model'].map(model_keys)

model_order = ['Hierarchical Oracle Model', 'Multistep Predictor', 'RSSM Discrete', 'RSSM Continuous', 'Multistep Delta', 'Transformer', 'SGNet']

# Split each model name into two lines if it contains two words
model_labels = ['\n'.join(model.split()) if len(model.split()) > 1 else model for model in model_order]

plt.figure(figsize=(13, 6))
bar_plot = sns.barplot(x='model', y='Accuracy', hue='Goal Num', data=df_plot,
            palette="Paired", errorbar='se', capsize=0.1, order=model_order)    
sns.despine(top=True, right=True)
plt.title('Multistep Goal Events Evaluation', fontsize=title_fontsize)
plt.xlabel('Model Name', fontsize=label_fontsize)
#plt.xticks(range(len(model_order)), model_order, fontsize=xtick_fontsize, rotation=0, horizontalalignment='center')
plt.ylabel('Accuracy', fontsize=label_fontsize)
plt.ylim([0, 1])
plt.legend(fontsize=20)
bar_plot.set_xticklabels(model_labels, fontsize=xtick_fontsize, rotation=0, horizontalalignment='center')
sns.despine(top=True, right=True)
if save_file is not None:
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
plt.show()

# PICKUP EVENTS
result_path = 'results/pickup_events_submission_full.csv'
df = pd.read_csv(result_path, index_col=0)

# Filtering rows where 'model' is in the keys of the model_keys dictionary
df_results = df[df['model'].isin(model_keys.keys())].copy()

# Replacing the 'model' values using the model_keys dictionary
df_results['model'] = df_results['model'].map(model_keys)

save_file = 'results/figures/all_results_pickup_events_submission'

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
plt.title('Pick Up Events Evaluation', fontsize=30)
plt.xlabel('Model Name', fontsize=label_fontsize) 
plt.ylabel('Accuracy', fontsize=label_fontsize)
plt.ylim([0, 1])
# Set the x-tick labels with the modified model names
bar_plot.set_xticklabels(model_labels, fontsize=xtick_fontsize, rotation=0, horizontalalignment='center')
if save_file is not None:
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
plt.show()
