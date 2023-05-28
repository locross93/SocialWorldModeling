# -*- coding: utf-8 -*-
"""
Created on Sat May 27 22:58:09 2023

@author: locro
"""

import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from constants_lc import DEFAULT_VALUES

def plot_goal_events(df_results, save_file):
    plt.figure()
    sns.barplot(x='model', y='score', data=df_results) 
    plt.title('Evaluate Single Goal Events From Forward Rollouts', fontsize=14)
    plt.xlabel('Model Name', fontsize=16) 
    plt.ylabel('Accuracy', fontsize=16)
    plt.ylim([0, 1])
    plt.show()
    if save_file is not None:
        plt.savefig(save_file, dpi=300)

def plot_eval_wm_results(df_results, args, save_file=None):
    if args.eval_type == 'goal_events':
        plot_goal_events(df_results, save_file)
        
def load_args():
    parser = argparse.ArgumentParser()
    # general pipeline parameters
    parser.add_argument('--results_dir', type=str, action='store',
                        default=DEFAULT_VALUES['results_dir'], 
                        help='Analysis directory')
    parser.add_argument('--results_file', type=str, help='Results filename')
    parser.add_argument('--save_file', type=str, default=None, help='Save filename')
    parser.add_argument('--eval_type', type=str, action='store',
                        choices=DEFAULT_VALUES['eval_types'], 
                        default='goal_events', 
                        help='Type of evaluation to perform')
    return parser.parse_args()
        
if __name__ == "__main__":    
    args = load_args()
    df_results = pd.read_csv(os.path.join(args.results_dir, args.results_file))
    plot_eval_wm_results(df_results, args, args.save_file)