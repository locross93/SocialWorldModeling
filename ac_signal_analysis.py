import argparse
from eval_world_model import Analysis
import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from constants import DEFAULT_VALUES, MODEL_DICT_VAL

class AC_Signal_Analysis(Analysis):
    def mse_peaks(self):
        self.load_data()
        pickup_timepoints = self.exp_info_dict[self.args.train_or_val]['pickup_timepoints']
        goal_timepoints = self.exp_info_dict[self.args.train_or_val]['goal_timepoints']
        single_goal_trajs = self.exp_info_dict[self.args.train_or_val]['single_goal_trajs']
        multi_goal_trajs = self.exp_info_dict[self.args.train_or_val]['multi_goal_trajs']

        if self.args.goal_type == 'single_step':
            trajs2rollout = single_goal_trajs
        elif self.args.goal_type == 'multi_step':
            trajs2rollout = multi_goal_trajs

        results = []
        num_timebins = 21
        rollout_length = 30
        for model_key in self.args.model_keys:
            print(f'Currently evaluating model {model_key}')
            self.model = self.load_model(model_key)
            model_info = MODEL_DICT_VAL[model_key]
            self.model_name = model_info['model_label']  
        
            for i, row in enumerate(trajs2rollout):
                if i % 50 == 0:
                    print(f'Currently evaluating trial {i}')
                x = self.input_data[row,:,:].unsqueeze(0)
                if self.args.goal_type == 'single_step':
                    steps2pickup = np.max(pickup_timepoints[row,:]).astype(int)
                    steps2goal = np.max(goal_timepoints[row,:]).astype(int)
                    all_ts = [(steps2pickup, steps2goal)]
                elif self.args.goal_type == 'multi_step':
                    pickup_ts = np.sort(pickup_timepoints[row,:].astype(int))
                    goal_ts = np.sort(goal_timepoints[row,:].astype(int))
                    all_ts = list(zip(pickup_ts, goal_ts))

                init_step = 5
                for goal_num, goal_ts in enumerate(all_ts):
                    steps2pickup = goal_ts[0]
                    steps2goal = goal_ts[1]
                    assert steps2pickup < steps2goal
                    
                    prev_steps = np.linspace(init_step, steps2pickup, (num_timebins//2)+1).astype(int)
                    after_steps = np.linspace(steps2pickup, steps2goal, (num_timebins//2)).astype(int)
                    #pick up ind is repeated, remove
                    after_steps = after_steps[1:]
                    all_steps = np.concatenate([prev_steps, after_steps])
    
                    for t, t_ind in enumerate(all_steps):
                        # after first goal, init_step is also a repeat
                        if goal_num > 0 and t == 0:
                            continue
                        rollout = np.min([rollout_length, 300-t_ind])
                        x_hat_imag = self.model.forward_rollout(x.cuda(), t_ind, rollout).cpu().detach()
                        x_supervise = x[:,t_ind:(t_ind+rollout),:]
                        mse = ((x_supervise - x_hat_imag)**2).mean().item()

                        if t == (num_timebins-2):
                            time2pickup = 1 + goal_num
                        else:
                            time2pickup = (t/(num_timebins-1)) + goal_num
                        results.append({'Time To Goal': time2pickup, 'MSE': mse, 'Model': self.model_name, 'Trial': i})
                    init_step = steps2goal
    
        df_plot = pd.DataFrame(results)
        df_plot['MSE Normalized'] = df_plot.groupby('Model')['MSE'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

        # Replacing the 'Model' values using the model_keys dictionary
        model_keys = {
        'MP-S1' : 'Multistep Predictor',
        'RSSM-S1': 'RSSM Discrete',
        'TF Emb2048 S1' : 'Transformer',
        }
        
        df_plot['Model'] = df_plot['Model'].map(model_keys)

        result_save_dir = os.path.join(self.args.analysis_dir, 'results')
        if self.args.save_file is None:
            save_file = 'ac_signal_curves'
        else:
            save_file = self.args.save_file
        save_path = os.path.join(result_save_dir, f'{save_file}.csv')
        save_path_plot = os.path.join(result_save_dir, f'{save_file}.png')
        df_plot.to_csv(save_path)

        if self.args.goal_type == 'single_step':
            sns.lineplot(x='Time To Goal', y='MSE Normalized', hue='Model', data=df_plot)
            plt.xlabel('Time To Goal %', fontsize=16)
            plt.ylabel('MSE - 30 Step Rollout', fontsize=18)
            plt.xticks(np.arange(0, 1.25, step=0.25)) 
            plt.axvline(x = 0.5, color = 'k', linestyle='--')
            plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
            plt.savefig(save_path_plot, dpi=300, bbox_inches='tight')
            plt.show()
        elif self.args.goal_type == 'multi_step':
            sns.lineplot(x='Time To Goal', y='MSE Normalized', hue='Model', data=df_plot)
            plt.xlabel('Time To Goal %', fontsize=16)
            plt.ylabel('MSE - 30 Step Rollout', fontsize=18)
            plt.xticks(np.arange(0, 3.25, step=0.25)) 
            plt.axvline(x = 0.5, color = 'k', linestyle='--')
            plt.axvline(x = 1.0, color = 'g', linestyle='--')
            plt.axvline(x = 1.5, color = 'k', linestyle='--')
            plt.axvline(x = 2.0, color = 'g', linestyle='--')
            plt.axvline(x = 2.5, color = 'k', linestyle='--')
            plt.axvline(x = 3.0, color = 'g', linestyle='--')
            plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
            plt.savefig(save_path_plot, dpi=300, bbox_inches='tight')
            plt.show()

def load_args():
    parser = argparse.ArgumentParser()
    # general pipeline parameters
    parser.add_argument('--batch_size', type=int, action='store',
                        default=DEFAULT_VALUES['batch_size'],
                        help='Batch size')
    parser.add_argument('--eval_seed', type=int, 
                        default=DEFAULT_VALUES['eval_seed'], help='Random seed')
    parser.add_argument('--analysis_dir', type=str, action='store',
                        default=DEFAULT_VALUES['analysis_dir'], 
                        help='Analysis directory')
    parser.add_argument('--model_config_dir', type=str, action='store',
                        default=DEFAULT_VALUES['model_config_dir'],
                        help='Model config directory')   
    parser.add_argument('--data_dir', type=str,
                         default=DEFAULT_VALUES['data_dir'], 
                         help='Data directory')
    parser.add_argument('--dataset', type=str,
                         default=DEFAULT_VALUES['dataset'], 
                         help='Dataset name')
    parser.add_argument('--checkpoint_dir', type=str, action='store',
                        default=DEFAULT_VALUES['checkpoint_dir'], 
                        help='Checkpoint directory')
    parser.add_argument('--save_file', type=str,
                        default=None, 
                        help='Filename for saving model')
    parser.add_argument('--train_or_val', type=str, default='val', help='Training or Validation Set')
    parser.add_argument('--model_keys', nargs='+', action='store',
                        default=DEFAULT_VALUES['model_keys'], 
                        help='A list of keys, seperate by spaces, for all model to evaluate')
    parser.add_argument('--goal_type', type=str, action='store',
                        default='single_step',
                        help='Type of evaluation to perform')
    parser.add_argument('--partial', type=float, default=1.0,         
                        help='Partial evaluation')
    return parser.parse_args()


if __name__ == "__main__":    
    args = load_args()
    analysis = AC_Signal_Analysis(args)
    analysis.mse_peaks()