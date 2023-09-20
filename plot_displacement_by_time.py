import seaborn as sns
import pickle
import numpy as np
import matplotlib.pyplot as plt

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

def plot_combined_displacement_errors(data, results_path):
    # Set up the Seaborn colorblind palette
    #sns.set_palette("colorblind")
    
    # Create a 2x1 grid of plots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))
    
    # Step-wise Displacement Errors
    for entry in data:
        model_name = entry['model']
        step_de = np.array(entry['all_trials']['step_de'])
        axes[0].plot(step_de, label=model_name)
        
    axes[0].set_title("Step-wise Displacement Errors")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Displacement Error")
    axes[0].legend()
    axes[0].grid(True)

    # # Cumulative Displacement Errors    
    # for entry in data:
    #     model_name = entry['model']
    #     step_de = np.array(entry['all_trials']['step_de'])
    #     cumulative_de = np.cumsum(step_de, axis=0)
    #     axes[1].plot(cumulative_de, label=model_name)

    # Cumulative Displacement Errors    
    for entry in data:
        model_name = entry['model']
        cum_disp = np.array(entry['all_trials']['cum_disp'])
        axes[1].plot(cum_disp, label=model_name)

    axes[1].set_title("Cumulative Displacement Errors")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Cumulative Displacement Error")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    # Save the combined plot as a high-quality PNG
    # file path replace pkl with png
    file_path_combined = results_path.replace('pkl', 'png')
    # if 'still_obj' in result_path:
    #     file_path_combined = "./results/combined_displacement_errors_still_obj_gt.png"
    # else:
    #     file_path_combined = "./results/combined_displacement_errors.png"
    plt.savefig(file_path_combined, dpi=500)

    plt.show()
    return file_path_combined

def plot_displacement_errors(data, results_path):
    #sns.set_palette("Set1")
    sns.set_palette("colorblind")
    title_fontsize=24
    xtick_fontsize=18
    label_fontsize=36
    
    fig = plt.figure(figsize=(9, 8))  # Adjust the size as needed
    
    # Step-wise Displacement Errors
    for entry in data:
        model_name = model_keys[entry['model']]
        #step_de = np.array(entry['all_trials']['step_de'])
        step_de = np.array(entry['all_trials']['cum_disp'])
        plt.plot(step_de, label=model_name)
        
    #plt.title("Average Displacement Error by Timestep", fontsize=title_fontsize)
    plt.title("Total Displacement of Stable Objects by Timestep", fontsize=title_fontsize)
    plt.xlabel("Timestep", fontsize=label_fontsize)
    plt.ylabel("Total Displacement", fontsize=label_fontsize)
    plt.legend(loc='upper left')  # Place legend in the top left corner
    plt.grid(True)
    sns.despine(top=True, right=True)
    plt.xticks(np.arange(0, 300, 50), np.arange(50, 350, 50), fontsize=xtick_fontsize, rotation=0, horizontalalignment='center')

    plt.tight_layout()

    # Save the plot as a high-quality PNG
    # file path replace pkl with png
    file_path_combined = results_path.replace('pkl', 'png')
    # if 'still_obj' in result_path:
    #     file_path_combined = "./results/combined_displacement_errors_still_obj_gt.png"
    # else:
    #     file_path_combined = "./results/combined_displacement_errors.png"
    plt.savefig(file_path_combined, dpi=500)

    plt.show()
    return file_path_combined

# Plotting the displacement errors for each model with distinct colors
#result_path = './results/eval_displacement_still_obj.pkl'
#result_path = './results/mp_disp_by_t4_still_obj.pkl'
#result_path = './results/disp_by_time_submission_still_obj.pkl'
#result_path = './results/disp_by_time_burninflag_still_obj.pkl'
# result_path = './results/eval_displacement_by_t.pkl'
# #result_path = './results/norm_models_disp_by_time_still_obj.pkl'
# #result_path = './results/gt_disp_by_time_still_obj.pkl'
# data = pickle.load(open(result_path, "rb"))

# result_path2 = './results/gt_disp_by_time.pkl'
# data2 = pickle.load(open(result_path2, "rb"))
# data = data + data2

# plot_displacement_errors(data, result_path)

########################################
# ade by time for stable objects
result_path = './results/norm_models_disp_by_time_still_obj.pkl'
data = pickle.load(open(result_path, "rb"))
result_path2 = './results/tf_disp_by_time_still_obj.pkl'
data2 = pickle.load(open(result_path2, "rb"))
data = data + data2
result_path3 = './results/sgnet_disp_by_time_still_obj.pkl'
data3 = pickle.load(open(result_path3, "rb"))
data = data + data3
result_path4 = './results/gt_disp_by_time_still_obj.pkl'
data4 = pickle.load(open(result_path4, "rb"))
data = data + data4

plot_displacement_errors(data, result_path)