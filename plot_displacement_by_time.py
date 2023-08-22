import seaborn as sns
import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_combined_displacement_errors(data):
    # Set up the Seaborn colorblind palette
    sns.set_palette("colorblind")
    
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

    # Cumulative Displacement Errors    
    for entry in data:
        model_name = entry['model']
        step_de = np.array(entry['all_trials']['step_de'])
        cumulative_de = np.cumsum(step_de, axis=0)
        axes[1].plot(cumulative_de, label=model_name)

    axes[1].set_title("Cumulative Displacement Errors")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Cumulative Displacement Error")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    # Save the combined plot as a high-quality PNG
    if 'still_obj' in result_path:
        file_path_combined = "./results/combined_displacement_errors_still_obj_new.png"
    else:
        file_path_combined = "./results/combined_displacement_errors.png"
    plt.savefig(file_path_combined, dpi=500)

    plt.show()
    return file_path_combined


# Plotting the displacement errors for each model with distinct colors
#result_path = './results/eval_displacement_still_obj.pkl'
result_path = './results/mp_disp_by_t4_still_obj.pkl'
#result_path = './results/disp_by_time_submission_still_obj.pkl'
#result_path = './results/disp_by_time_burninflag_still_obj.pkl'
# result_path = './results/eval_displacement.pkl'
data = pickle.load(open(result_path, "rb"))
plot_combined_displacement_errors(data)