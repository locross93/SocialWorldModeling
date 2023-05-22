# SocialWorldModeling

This repository contains code for training various types of world models on the Social World Modeling Benchmark, built in TDW.

## Training World Models

You can train a world model using the `train_world_model.py` script. This script takes command line arguments to specify the model type and the configuration file.

Example usage:

```bash
python train_world_model.py --model dreamerv2 --config rssm_disc_default_config.json
```

In this example, dreamerv2 is the model class and rssm_disc_default_config.json is the configuration file.


### Model Configuration

Model configurations are specified using JSON files. These files contain the hyperparameters and model type that is used during training. You can find the configuration files in the config directory.

You can modify these files to change the configuration, or you can create new ones if you want to experiment with different settings. If you specify a configuration parameter via the command line, that will override the value in the configuration file.

Here is an example of what a configuration file might look like:

```bash
{
    "input_size": 64,
    "deter_size": 256,
    "dec_hidden_size": 512,
    "rssm_type": "discrete",
    "rnn_type": "GRU",
    ...
}
```

## Evaluation Pipeline

The evaluation pipeline contains various tests on validation data, to probe the world model's understanding of events and goal-directed behavior, in addition to more traditional metrics for trajectory prediction.

1. Prediction error of forward rollouts on validation set - eval_val_rollouts.py
2. Simulating events in forward rollouts 
    1. Evaluate single goal events - eval_rollouts_goal_events.py
    2. Evaluate multi goal events - eval_rollouts_multigoal_events.py
    2. Evaluate move events - eval_rollouts_move_events.py
    3. Evaluate pick up events
3. Visualizing forward rollouts - visualize_model_rollouts.py

In each evaluation file, specify the models you want to test in a dictionary of dicts, model_dict. Here's an example of what this might look like:

```
model_dict = {
        'rssm_disc': {'class': DreamerV2, 'config': 'rssm_disc_default_config.json', 
                      'model_dir': 'rssm_disc', 'epoch': '1000', 'model_label': 'RSSM Discrete'},
        }
```



