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
    if save_file is not None:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()
        
def plot_multigoal_events(df_results, save_file):
    df_plot = pd.melt(df_results, id_vars="model", value_vars=["g2_acc", "g3_acc"], var_name="Goal Num", value_name="Accuracy")
    df_plot["Goal Num"] = df_plot["Goal Num"].replace({"g2_acc": "2nd Goal", "g3_acc": "3rd Goal"})

    plt.figure()
    sns.barplot(x='model', y='Accuracy', hue='Goal Num', data=df_plot) 
    plt.title('Evaluate Multistep Goal Events From Forward Rollouts', fontsize=14)
    plt.xlabel('Model Name', fontsize=16) 
    plt.ylabel('Accuracy', fontsize=16)
    plt.ylim([0, 1])
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    if save_file is not None:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()
        
def plot_move_events(df_results, save_file):
    df_plot = pd.melt(df_results, id_vars="model", value_vars=["accuracy", "precision", "recall"], var_name="Metric", value_name="Score")
    
    plt.figure()
    sns.barplot(x='Metric', y='Score', hue='model', data=df_plot) 
    plt.title('Evaluate Move Events From Forward Rollouts', fontsize=14)
    plt.xlabel('Metric', fontsize=16) 
    plt.ylabel('Score', fontsize=16)
    plt.ylim([0, 1])
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    if save_file is not None:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_pickup_events(df_results, save_file):
    plt.figure()
    sns.barplot(x='model', y='score', data=df_results) 
    plt.title('Evaluate Pick Up Events From Forward Rollouts', fontsize=14)
    plt.xlabel('Model Name', fontsize=16) 
    plt.ylabel('Accuracy', fontsize=16)
    plt.ylim([0, 1])
    if save_file is not None:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()

def plot_eval_wm_results(df_results, args, save_file=None):
    if args.eval_type == 'goal_events':
        plot_goal_events(df_results, save_file)
    elif args.eval_type == 'multigoal_events':
        plot_multigoal_events(df_results, save_file)
    elif args.eval_type == 'move_events':
        plot_move_events(df_results, save_file)
    elif args.eval_type == 'pickup_events':
        plot_pickup_events(df_results, save_file)
        
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