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
import matplotlib.patches as mpatches 
import seaborn as sns
from constants import BEHV_CATE_DICT
#%%

# Evaluation results
#%%
def process_plot_data(result_path):
    event_type = result_path.split('_')[-2]
    df = pd.read_csv(result_path, index_col=0)
    df['base_model'] = df['model'].apply(lambda x: '_'.join(x.split('-')[: -1]))#x.split('-')[0])


    if event_type == 'goal':
        grouped_df = df.groupby('base_model').agg(
             mean_score=('score', np.mean),
             standard_error=('score', lambda x: np.std(x, ddof=1)/np.sqrt(x.size)))
        # Reset the index
        df = grouped_df.reset_index()
    elif event_type == 'multigoal':
        df = pd.melt(df, id_vars="base_model", value_vars=["g2_acc", "g3_acc"], var_name="Goal Num", value_name="Accuracy")
        df["Goal Num"] = df["Goal Num"].replace({"g2_acc": "2nd Goal", "g3_acc": "3rd Goal"})
    elif event_type == 'move':
        f1_score = 2 * (df['precision'] * df['recall']) / (df['precision'] + df['recall'])
        df['f1_score'] = f1_score
        df = pd.melt(df, id_vars="base_model", value_vars=["f1_score", "precision", "recall"], var_name="Metric", value_name="Score")
    elif event_type == 'pickup':
        df = df
    return df

def plot_paper_eval_results(goal_path, multigoal_path, move_path, pickup_path):
    goal_df = process_plot_data(goal_path)
    print(goal_df)
    multigoal_df = process_plot_data(multigoal_path)
    move_df = process_plot_data(move_path)
    pickup_df = process_plot_data(pickup_path)

    "Plot"
    xlabels = ['Multistep Predictor', 'RSSM Discrete', 'RSSM Continuous', 'Multistep Delta', 'Transformer']
    # style
    sn_palette='Set2'
    title_fontsize = 40
    xtick_fontsize = 22
    label_fontsize = 36
    legend_fontsize = 28
    fig, axs = plt.subplots(2, 2, figsize=(32, 21))
    # goal events plot
    sns.barplot(ax=axs[0, 0], x='base_model', y='mean_score', data=goal_df, palette=sn_palette)
    axs[0, 0].errorbar(x=goal_df['base_model'], y=goal_df['mean_score'], yerr=goal_df['standard_error'], fmt='none', c='b', capsize=5)
    sns.despine(top=True, right=True)
    axs[0, 0].set_title('Single Goal Events', fontsize=title_fontsize)
    axs[0, 0].set_xlabel('Model Name', fontsize=label_fontsize)
    axs[0, 0].set_xticklabels(xlabels, fontsize=xtick_fontsize, rotation=10, horizontalalignment='center')
    axs[0, 0].set_ylabel('Accuracy', fontsize=label_fontsize)
    axs[0, 0].set_ylim([0, 1])
    # multigoal events plot
    sns.barplot(ax=axs[0, 1], x='base_model', y='Accuracy', hue='Goal Num', data=multigoal_df, palette="Paired")
    axs[0, 1].set_title('Multi Goal Events', fontsize=title_fontsize)
    axs[0, 1].set_xlabel("Model Name", fontsize=label_fontsize)
    axs[0, 1].set_xticklabels(xlabels, fontsize=xtick_fontsize, rotation=0, horizontalalignment='center')
    axs[0, 1].set_ylabel('Accuracy', fontsize=label_fontsize)
    axs[0, 1].set_ylim([0, 1])
    axs[0, 1].legend(loc='upper right', bbox_to_anchor=(1, 1.05))
    axs[0, 1].get_legend().remove()
    # pickup events plot
    sns.barplot(ax=axs[1, 0], x='base_model', y='score', data=pickup_df, palette=sn_palette)
    axs[1, 0].set_title("Move Events", fontsize=title_fontsize)
    axs[1, 0].set_xlabel('Model Name', fontsize=label_fontsize)
    axs[1, 0].set_xticklabels(xlabels, fontsize=xtick_fontsize, rotation=0, horizontalalignment='center')
    axs[1, 0].set_ylabel('Accuracy', fontsize=label_fontsize)
    axs[1, 0].set_ylim([0, 1])
    # move events plot
    sns.barplot(ax=axs[1, 1], x='Metric', y='Score', hue='base_model', data=move_df, 
                palette=sn_palette, capsize=.2, ci='68')
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
goal_path = 'results/all_results_goal_events.csv'
multigoal_path = 'results/all_results_multigoal_events.csv'
move_path = 'results/eval_move_events.csv' #'results/all_results_move_events.csv'
pickup_path = 'results/all_results_pickup_events.csv'
plot_paper_eval_results(goal_path, multigoal_path, move_path, pickup_path)
#%%



# Displatment results
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
        for behavior, (mean, std) in behaviors.items():
            data.append({'Model': model, 'Behavior': behavior, 'Mean': mean, 'Std': std})
    return pd.DataFrame(data)

def plot_placement(df, legend=False):
    title_dict = {
        'chasing': 'Chasing',
        'ss gathering': 'Single-step Gathering',
        'ms gatherinng': 'Multi-step Gathering',
        'random': 'Random',
        'mimicry': 'Mimicry',
        'colab_gathereing': 'Collaborative Gathering',
        'adversarial_gathering': 'Adversarial Gathering'
    }
    behavior_list = df['Behavior'].unique()
    xlabels = ['Multistep Predictor', 'RSSM Discrete', 'RSSM Continuous', 'Multistep Delta', 'Transformer']

    df['Model'] = df['Model'].replace('MP', 'Multistep Predictor')
    df['Model'] = df['Model'].replace('RSSM', 'RSSM Discrete')
    df['Model'] = df['Model'].replace('RSSM_Cont', 'RSSM Continuous')
    df['Model'] = df['Model'].replace('MD', 'Multistep Delta')
    df['Model'] = df['Model'].replace('TF_Emb2048', 'Transformer')

    fig, axs = plt.subplots(nrows=len(behavior_list), figsize=(10, 11))
    palette = sns.color_palette('Set2', len(df['Model'].unique()))
    bar_width = 0.3

    handles = []  # to hold the legend handles
    for i, (ax, behavior) in enumerate(zip(axs, behavior_list)):
        df_behavior = df[df['Behavior'] == behavior]
        
        #sns.barplot(x='Model', y='Mean', data=df_behavior, ax=ax, errorbar=None, palette="Set2", width=0.25)
        if i != 6:
            ax.set_xlabel('')
            ax.set_xticklabels([])
        else:   
            ax.set_xlabel('Model', fontsize=20) 
            ax.set_xticks(xs)
            ax.set_xticklabels(xlabels, fontsize=12, rotation=10, horizontalalignment='center')
        if i != 3:
            ax.set_ylabel('')
        else:
            ax.set_ylabel('Displacement Error', fontsize=20)

        df_behavior = df[df['Behavior'] == behavior]
        xs = [0, 0.5, 1, 1.5, 2]
        bars = ax.bar(xs, df_behavior['Mean'], color=palette, width=bar_width, )
        ax.errorbar(x=xs, y=df_behavior['Mean'],
                    yerr=df_behavior['StdErr'], fmt='none', c='black', capsize=5)
        ax.set_title(title_dict[behavior])
        ax.set_ylim([0, 15.0])
        # Add to the legend handles
        if i == 0:  # we only need to add handles once
            for bar, model in zip(bars, df_behavior['Model']):
                handles.append(mpatches.Patch(color=bar.get_facecolor(), label=model))

    if legend:
        plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.15, 1.0), fontsize=12, title='Model Name')
    plt.tight_layout()
    plt.show()
    
BEHV_CATE_DICT = {
            'colab_gathereing': ['leader_follower', 'follower_leader'],
            'adversarial_gathering': ['adversarial_gathering', 'gathering_adversarial'],
            'ss gathering': ['random_gathering', 'gathering_static', 'static_gathering', 'gathering_random'],
            'ms gatherinng': ['static_multistep', 'multistep_static', 'random_multistep', 'multistep_random'],
            'chasing': ['chaser_runner', 'runner_chaser'],
            'mimicry': ['random_mimic', 'mimic_random'],
            'random': ['random_gathering', 'random_mimic', 'mimic_random', 'random_multistep',
                       'random_random', 'multistep_random', 'gathering_random']
}

CATE_BEHV_DICT = {}#{v: k for k, vs in BEHV_CATE_DICT.items() for v in vs}
for k, vs in BEHV_CATE_DICT.items():
    for v in vs:
        if v in CATE_BEHV_DICT:
            CATE_BEHV_DICT[v].append(k)
        else:
            CATE_BEHV_DICT[v] = [k]
            
os.chdir('/Users/locro/Documents/Stanford/SocialWorldModeling')

result_path = 'results/displacement_submission.pkl'
results = pickle.load(open(result_path, 'rb'))

result_path2 = 'results/sgnet_displacement.pkl'
results2 = pickle.load(open(result_path2, 'rb'))

result_path3 = 'results/gt_end_state_displacement.pkl'
results3 = pickle.load(open(result_path3, 'rb'))

results = results + results2 + results3

#result = results[0]
#displacement_vals = result[50]  
#all_behaviors = [behavior for behavior in displacement_vals]
all_behaviors = ['leader_follower', 'follower_leader', 'adversarial_gathering', 'gathering_adversarial',
                 'random_gathering', 'gathering_static', 'static_gathering', 'gathering_random',
                 'static_multistep', 'multistep_static', 'random_multistep', 'multistep_random',
                 'chaser_runner', 'runner_chaser','random_mimic', 'mimic_random','random_random']

# {model_name: {behavior: [val_by_seed]}
model_seed_ade_dict = {}
model_seed_fde_dict = {}
for result in results:
    if '-' in result['model']:
        model_name = '_'.join(result['model'].split('-')[: -1])
    elif ' ' in result['model']:
        model_name = '_'.join(result['model'].split(' ')[: -1])
    print(result['model'])
    displacement_vals = result
    ade_key = 'ade'
    # if model_name == 'GT_End_State':
    #     displacement_vals = result
    #     ade_key = 'ade'
    # else:
    #     displacement_vals = result[50]   
    #     ade_key = 'mde'
    if model_name not in model_seed_ade_dict:
        model_seed_ade_dict[model_name] = {}
        model_seed_fde_dict[model_name] = {}
        for behavior in all_behaviors:            
            ade = displacement_vals[behavior][ade_key].item()
            fde = displacement_vals[behavior]['fde'].item()
            for cat in CATE_BEHV_DICT[behavior]:
                if cat in model_seed_ade_dict[model_name]:
                    model_seed_ade_dict[model_name][cat].append(ade)
                    model_seed_fde_dict[model_name][cat].append(fde)
                else:
                    model_seed_ade_dict[model_name][cat] = [ade]
                    model_seed_fde_dict[model_name][cat] = [fde]
    else:
        for behavior in all_behaviors:
            ade = displacement_vals[behavior][ade_key].item()
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
#plot_placement(ade_df)
#%%

#%%
fde_df = create_data_df(model_seed_fde_dict)
#plot_placement(fde_df, legend=True)
#%%

model_keys = {
    'GT_End_State' : 'Hierarchical Oracle Model',
    'MP' : 'Multistep Predictor',
    'RSSM' : 'RSSM Discrete',
    'RSSM_Cont' : 'RSSM Continuous',
    'MD' : 'Multistep Delta',
    'TF_Emb2048' : 'Transformer',
    'SGNet': 'SGNet'
    }

behavior_keys = {
    'colab_gathereing': 'Collaborative Gathering',
    'adversarial_gathering': 'Adversarial Gathering',
    'ss gathering': 'Single-step Gathering',
    'ms gatherinng': 'Multi-step Gathering',
    'chasing': 'Chasing',
    'mimicry': 'Mimicry',
    'random': 'Random',
    }

# Filtering rows where 'model' is in the keys of the model_keys dictionary
ade_df_results = ade_df[ade_df['Model'].isin(model_keys.keys())].copy()

# Replacing the 'model' values using the model_keys dictionary
ade_df_results['Model'] = ade_df_results['Model'].map(model_keys)

# Replace the 'behavior' values using the behavior_keys dictionary
ade_df_results['Behavior'] = ade_df_results['Behavior'].map(behavior_keys)

# repeat for fde_df
fde_df_results = fde_df[fde_df['Model'].isin(model_keys.keys())].copy()
fde_df_results['Model'] = fde_df_results['Model'].map(model_keys)
fde_df_results['Behavior'] = fde_df_results['Behavior'].map(behavior_keys)

# Define the desired order of behaviors
behavior_order = [
    "Single-step Gathering",
    "Multi-step Gathering",
    "Collaborative Gathering",
    "Adversarial Gathering",
    "Random",
    "Mimicry",
    "Chasing"
]

# Reorder the DataFrame based on the specified sequence of behaviors
ade_df_results = pd.concat([ade_df_results[ade_df_results['Behavior'] == behavior] for behavior in behavior_order])
fde_df_results = pd.concat([fde_df_results[fde_df_results['Behavior'] == behavior] for behavior in behavior_order])

# save dataframes
ade_df_results.to_csv('results/ade_results.csv')
fde_df_results.to_csv('results/fde_results.csv')

# """ Displacement error through time"""
# #%%
# # Define a list of distinct colors
# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
# def plot_displacement_errors_distinct_colors(data):
#     plt.figure(figsize=(14, 8))
    
#     # Iterate over each model's data and plot the step_de values with distinct colors
#     for i, model_data in enumerate(data):
#         model_name = model_data['model']
#         step_de = model_data['all_trials']['step_de'].numpy()
#         plt.plot(step_de, label=model_name, color=colors[i % len(colors)])
    
#     plt.title('Displacement Errors Across Time Steps')
#     plt.xlabel('Time Steps')
#     plt.ylabel('Displacement Error')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# Plotting the displacement errors for each model with distinct colors
# result_path = 'results/eval_displacement.pkl'
# data = pickle.load(open(result_path, "rb"))
# plot_displacement_errors_distinct_colors(data)



