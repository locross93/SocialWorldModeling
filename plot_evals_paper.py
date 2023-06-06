# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 12:38:38 2023

@author: locro
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


#####################################################
################ GOAL EVENTS ########################
#####################################################

results_dir = '/Users/locro/Documents/Stanford/SocialWorldModeling/results/'
results_file = 'all_results_goal_events.csv'
df_all_results = pd.read_csv(os.path.join(results_dir, results_file), index_col=0)

inds2plot = [25, 18, 28, 26, 22]
df_results = df_all_results.loc[inds2plot]

xlabels = ['RSSM Discrete', 'RSSM Continuous', 'Multistep Predictor', 'Multistep Delta', 'IRIS Transformer']

save_file = '/Users/locro/Documents/Stanford/figures/evals/eval_goal_events_6_5_23'

plt.figure(figsize=(10, 6))
sns.barplot(x='model', y='score', data=df_results, palette='Set2') 
plt.title('Evaluate Single Goal Events', fontsize=28)
plt.xlabel('Model Name', fontsize=20) 
plt.xticks(range(len(xlabels)), xlabels, fontsize=12)
plt.ylabel('Accuracy', fontsize=20)
plt.ylim([0, 1])
# Hide the right and top spines
sns.despine(top=True, right=True)
if save_file is not None:
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
plt.show()


results_dir = '/Users/locro/Documents/Stanford/SocialWorldModeling/results/'
results_file = 'all_results_goal_events_level2.csv'
df_all_results = pd.read_csv(os.path.join(results_dir, results_file), index_col=0)


#####################################################
################ PICK UP EVENTS #####################
#####################################################

results_dir = '/Users/locro/Documents/Stanford/SocialWorldModeling/results/'
results_file = 'all_results_pickup_events.csv'
df_all_results = pd.read_csv(os.path.join(results_dir, results_file), index_col=0)

inds2plot = [3, 2, 0, 1, 22]
df_results = df_all_results.loc[inds2plot]

xlabels = ['RSSM Discrete', 'RSSM Continuous', 'Multistep Predictor', 'Multistep Delta', 'IRIS Transformer']