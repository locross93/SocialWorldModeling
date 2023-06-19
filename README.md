# SocialWorldModeling

This repository contains code for training various types of world models on the Social World Modeling Benchmark, built in TDW.

## Download dataset

Data can be downloaded at https://drive.google.com/file/d/1sHiTBOCtfqFPk1P_5rzZ8Cl61B_5Nj-A/view - this zip file includes swn_data.pkl with training and validation splits and swm_data_exp_info.pkl a dictionary of information about trial type, events, and more which is used in the evaluations.

## Training World Models

You can train a world model using the `train_world_model.py` script. This script takes command line arguments to specify the model type and the configuration file.

Example usage:

```bash
python train_world_model.py --config rssm_disc_default_config.json
```

In this example, dreamerv2 is the model class and rssm_disc_default_config.json is the configuration file.

Paths for the dataset directory, checkpoint directory, etc. should be specified in constants.py

### Model Configuration

Model configurations are specified using JSON files. These files contain the hyperparameters and model type that is used during training. You can find the configuration files in the config directory.

You can modify these files to change the configuration, or you can create new ones if you want to experiment with different settings. If you specify a configuration parameter via the command line, that will override the value in the configuration file.

Here is an example of what a configuration file might look like:

```bash
{
    "input_size": 35,
    "deter_size": 1024,
    "dec_hidden_size": 2048,
    "rssm_type": "discrete",
    "rnn_type": "GRU",
    "category_size": 32,
    "class_size": 32,
    "model_type": "dreamerv2"
}
```

## Evaluation Pipeline

The evaluation pipeline contains various tests on validation data, to probe the world model's understanding of events and goal-directed behavior, in addition to more traditional metrics for trajectory prediction.

Example usage:

```bash
python eval_world_model.py --eval_type goal_events --model_key rssm_disc
```

1. **Prediction error of forward rollouts on validation set (ADE and FDE)**: --eval_type displacement
2. **Simulating events in forward rollouts** 
    1. Evaluate single goal events: --eval_type goal_events
    2. Evaluate multi goal events: --eval_type multigoal_events
    2. Evaluate move events: --eval_type move_events
    3. Evaluate pick up events: --eval_type pickup_events
3. **Visualizing forward rollouts**: visualize_model_rollouts.py

In the constants.py file, specify the models you want to test in a dictionary of dicts, MODEL_DICT_VAL. Here's an example of what this might look like:

```
MODEL_DICT_VAL = {
        'rssm_disc': {'class': DreamerV2, 'config': 'rssm_disc_default_config.json', 
                      'model_dir': 'rssm_disc', 'epoch': '1000', 'model_label': 'RSSM Discrete'},
        }
```

## Generating More Data

Data generating code is located in the **swm_simulation_env** folder

To install code and necessary dependencies:

```bash
cd swm_simulation_env
pip install -r requirements.txt
pip install -e .
```

To generate data:

```bash
cd swm_simulation_env
python data_generation/generate_swm_data.py
```

Data is stored in **swm_simulation_env/img_out folder**

To generate data of particular behavior types, use the scenario_num flag with the corresponding numbers for each scenario. Each scenario includes two agents and behavior types.

0 - gathering + random  
1 - multistep + random  
2 - leader + follower (collaborative gathering)  
3 - gathering + adversarial (adversarial gathering)  
4 - random + random  
5 - random + mimic  
6 - runner + chaser  
7 - gathering + static  
8 - multistep + static  

For example, to generate 100 collaborative gathering trials

```bash
python data_generation/generate_swm_data.py --scenario_num 2 --num_trials2gen 100
```