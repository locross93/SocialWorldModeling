# SocialWorldModeling

This repository contains code for training various types of world models on the Social World Modeling Benchmark, built in TDW.

## Training World Models

You can train a world model using the `train_world_model.py` script. This script takes command line arguments to specify the model type and the configuration file.

Example usage:

```bash
python train_world_model.py --model dreamerv2 --config rssm_disc_default_config.json
```

In this example, dreamerv2 is the model type and rssm_disc_default_config.json is the configuration file.


### Model Configuration

Model configurations are specified using JSON files. These files contain the hyperparameters and model class that is used during training. You can find the configuration files in the config directory.

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
2. Reconstructing events in forward rollouts 
    1. Evaluate goal events
    2. Evaluate move events
    3. Evaluate pick up events




