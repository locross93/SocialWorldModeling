# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 12:00:58 2023

@author: locro
"""
#%%
import pickle
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from constants import BEHV_CATE_DICT
#%%


#%%
def process_plot_data(result_path):
    event_type = result_path.split('_')[-2]
    df = pd.read_csv(result_path, index_col=0)
    if event_type == 'goal':
        inds2plot = [25, 18, 28, 26, 22]
        df = df.loc[inds2plot]
    elif event_type == 'multigoal':
        inds2plot = [0, 1, 2, 4, 10]
        df = df.loc[inds2plot]  
        df = pd.melt(df, id_vars="model", value_vars=["g2_acc", "g3_acc"], var_name="Goal Num", value_name="Accuracy")
        df["Goal Num"] = df["Goal Num"].replace({"g2_acc": "2nd Goal", "g3_acc": "3rd Goal"})
    elif event_type == 'move':
        f1_score = 2 * (df['precision'] * df['recall']) / (df['precision'] + df['recall'])
        df['f1_score'] = f1_score
        df = pd.melt(df, id_vars="model", value_vars=["f1_score", "precision", "recall"], var_name="Metric", value_name="Score")
    elif event_type == 'pickup':
        inds2plot = [0, 1, 2, 4, 6]
        df = df.loc[inds2plot]
    return df

def plot_paper_eval_results(goal_path, multigoal_path, move_path, pickup_path):
    goal_df = process_plot_data(goal_path)
    multigoal_df = process_plot_data(multigoal_path)
    move_df = process_plot_data(move_path)
    pickup_df = process_plot_data(pickup_path)

    "Plot"
    xlabels = ['RSSM Discrete', 'RSSM Continuous', 'Multistep Predictor', 'Multistep Delta', 'IRIS Transformer']
    # style
    sn_palette='Set2'
    title_fontsize = 40
    xtick_fontsize = 22
    label_fontsize = 36
    legend_fontsize = 28
    fig, axs = plt.subplots(2, 2, figsize=(32, 21))
    # goal events plot
    sns.barplot(ax=axs[0, 0], x='model', y='score', data=goal_df, palette=sn_palette)
    sns.despine(top=True, right=True)
    axs[0, 0].set_title('Single Goal Events', fontsize=title_fontsize)
    axs[0, 0].set_xlabel('Model Name', fontsize=label_fontsize)
    axs[0, 0].set_xticklabels(xlabels, fontsize=xtick_fontsize, rotation=0, horizontalalignment='center')
    axs[0, 0].set_ylabel('Accuracy', fontsize=label_fontsize)
    axs[0, 0].set_ylim([0, 1])
    # multigoal events plot
    sns.barplot(ax=axs[0, 1], x='model', y='Accuracy', hue='Goal Num', data=multigoal_df, palette="Paired")
    axs[0, 1].set_title('Multi Goal Events', fontsize=title_fontsize)
    axs[0, 1].set_xlabel("Model Name", fontsize=label_fontsize)
    axs[0, 1].set_xticklabels(xlabels, fontsize=xtick_fontsize, rotation=0, horizontalalignment='center')
    axs[0, 1].set_ylabel('Accuracy', fontsize=label_fontsize)
    axs[0, 1].set_ylim([0, 1])
    axs[0, 1].legend(loc='upper right', bbox_to_anchor=(1, 1.05))
    axs[0, 1].get_legend().remove()
    # pickup events plot
    sns.barplot(ax=axs[1, 0], x='model', y='score', data=pickup_df, palette=sn_palette)
    axs[1, 0].set_title("Move Events", fontsize=title_fontsize)
    axs[1, 0].set_xlabel('Model Name', fontsize=label_fontsize)
    axs[1, 0].set_xticklabels(xlabels, fontsize=xtick_fontsize, rotation=0, horizontalalignment='center')
    axs[1, 0].set_ylabel('Accuracy', fontsize=label_fontsize)
    axs[1, 0].set_ylim([0, 1])
    # move events plot
    sns.barplot(ax=axs[1, 1], x='Metric', y='Score', hue='model', data=move_df, palette=sn_palette)
    axs[1, 1].set_title('Move Events', fontsize=title_fontsize)
    axs[1, 1].set_xlabel('Metric', fontsize=label_fontsize)
    axs[1, 1].set_xticklabels(['F1 Score', 'Precision', 'Recall'], fontsize=xtick_fontsize, rotation=0, horizontalalignment='center')
    axs[1, 1].set_ylabel('Score', fontsize=label_fontsize)
    axs[1, 1].set_ylim([0, 1])
    axs[1, 1].get_legend().remove()


    mg_handles, mg_labels = axs[0, 1].get_legend_handles_labels()
    fig.legend(mg_handles, mg_labels, loc='upper right', bbox_to_anchor=(1.1, 1.0), fontsize=legend_fontsize, title='Goal Num')
    sg_handles, sg_labels = axs[1, 1].get_legend_handles_labels()
    fig.legend(sg_handles, sg_labels, loc='center right', bbox_to_anchor=(1.2, 0.45), fontsize=legend_fontsize, title='Model Name')
    #move_handles, move_labels = axs[1, 0].get_legend_handles_labels()
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.2)  # Adjust vertical spacing
    plt.savefig('results/all_results.png', dpi=300, bbox_inches='tight')
    plt.show()
#%%


#%%
def get_displacement(displacement_vals, dis_type='mde'):
    disp_by_cate = {}
    for behavior in displacement_vals:
        ade = displacement_vals[behavior][dis_type].item()
        fde = displacement_vals[behavior][dis_type].item()
        behavior_category = CATE_BEHV_DICT[behavior]
        if behavior_category not in model_seed_ade_dict[model_name]:
            disp_by_cate[behavior_category] = [ade]
            disp_by_cate[behavior_category] = [fde]
        else:
            disp_by_cate[behavior_category].append(ade)
            disp_by_cate[behavior_category].append(fde)
    return disp_by_cate

def create_data_df(data_dict):
    data = []
    for model, behaviors in data_dict.items():
        for behavior, (mean, sterr) in behaviors.items():
            data.append({'Model': model, 'Behavior': behavior, 'Mean': mean, 'StdErr': sterr})
    return pd.DataFrame(data)

def plot_placement(df):
    behavior_list = df['Behavior'].unique()

    fig, axs = plt.subplots(nrows=len(behavior_list), figsize=(10, 10))

    for ax, behavior in zip(axs, behavior_list):
        df_behavior = df[df['Behavior'] == behavior]
        sns.barplot(x='Model', y='Mean', data=df_behavior, ax=ax, errorbar=None, palette="Set2")        
        ax.errorbar(x=df_behavior['Model'], y=df_behavior['Mean'], yerr=df_behavior['StdErr'], fmt='none', c='b', capsize=5)
        ax.set_title(behavior)

    plt.tight_layout()
    plt.show()

CATE_BEHV_DICT = {}#{v: k for k, vs in BEHV_CATE_DICT.items() for v in vs}
for k, vs in BEHV_CATE_DICT.items():
    for v in vs:
        if v in CATE_BEHV_DICT:
            CATE_BEHV_DICT[v].append(k)
        else:
            CATE_BEHV_DICT[v] = [k]

result_path = 'results/eval_displacement.pkl'
results = pickle.load(open(result_path, 'rb'))

# {model_name: {behavior: [val_by_seed]}
model_seed_ade_dict = {}
model_seed_fde_dict = {}
for result in results:
    if '-' in result['model']:
        model_name = '_'.join(result['model'].split('-')[: -1])
    elif ' ' in result['model']:
        model_name = '_'.join(result['model'].split(' ')[: -1])
    print(result['model'])
    displacement_vals = result[50]    
    if model_name not in model_seed_ade_dict:
        model_seed_ade_dict[model_name] = {}
        model_seed_fde_dict[model_name] = {}
        for behavior in displacement_vals:            
            ade = displacement_vals[behavior]['mde'].item()
            fde = displacement_vals[behavior]['fde'].item()
            for cat in CATE_BEHV_DICT[behavior]:
                if cat in model_seed_ade_dict[model_name]:
                    model_seed_ade_dict[model_name][cat].append(ade)
                    model_seed_fde_dict[model_name][cat].append(fde)
                else:
                    model_seed_ade_dict[model_name][cat] = [ade]
                    model_seed_fde_dict[model_name][cat] = [fde]
    else:
        for behavior in displacement_vals:
            ade = displacement_vals[behavior]['mde'].item()
            fde = displacement_vals[behavior]['fde'].item()
            for cat in CATE_BEHV_DICT[behavior]:            
                if cat in model_seed_ade_dict[model_name]:
                    model_seed_ade_dict[model_name][cat].append(ade)
                    model_seed_fde_dict[model_name][cat].append(fde)
                else:
                    model_seed_ade_dict[model_name][cat] = [ade]
                    model_seed_fde_dict[model_name][cat] = [fde]

for model in model_seed_ade_dict:
    for cat in model_seed_ade_dict[model]:
        model_seed_ade_dict[model][cat] = (
            np.mean(model_seed_ade_dict[model][cat]), 
            np.std(model_seed_ade_dict[model][cat]) / np.sqrt(len(model_seed_ade_dict[model][cat])))
        model_seed_fde_dict[model][cat] = (
            np.mean(model_seed_fde_dict[model][cat]),
            np.std(model_seed_fde_dict[model][cat]) / np.sqrt(len(model_seed_fde_dict[model][cat])))



#%%

#%%
ade_df = create_data_df(model_seed_ade_dict)
plot_placement(ade_df)
#%%

#%%
fde_df = create_data_df(model_seed_fde_dict)
plot_placement(fde_df)  
#%%