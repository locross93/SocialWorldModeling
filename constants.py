import os
from models import DreamerV2, MultistepPredictor, MultistepDelta, \
    TransformerMSPredictor, TransformerIrisWorldModel, TransformerWorldModel    
from agent_former.agentformer import AgentFormer
from sgnet_models.SGNet_CVAE import SGNet_CVAE
from gnn_models.imma import IMMA
from gnn_models.gat import GAT
from gnn_models.rfm import RFM


"""Values for training"""
MODEL_DICT_TRAIN = {
    'rssm_disc': DreamerV2,
    'dreamerv2': DreamerV2,
    'multistep_predictor': MultistepPredictor,
    'multistep_delta':  MultistepDelta,
    'transformer_wm': TransformerWorldModel,   
    'transformer_mp': TransformerMSPredictor,
    'transformer_iris': TransformerIrisWorldModel,
    'imma': IMMA,
    'gat': GAT,
    'rfm': RFM,
    'sgnet_cvae': SGNet_CVAE,
    'agent_former': AgentFormer
}
"""Values for validation"""
# Only using 5-31-23 for the paper
MODEL_DICT_VAL=  {
    # 'mp_mlp_2048_lr3e-4': {
    #     'class': MultistepPredictor, 'config': 'multistep_predictor_input_size_35_mlp_hidden_size_2048.json',
    #     'model_dir': 'mp_d3_mlp_hidden_size_2048_lr3e-4', 'model_label': 'MP DS3 MLP HS2048 LR3e-4'},    # running N5
    # 'rssm_disc_h_2048_lr3e-4': {
    #     'class': DreamerV2, 'config': 'rssm_disc_ds3_dec_hidden_size_2048.json', 
    #     'model_dir': 'rssm_disc_ds3', 'model_label': 'RSSM Discrete DS3'},    # running N5
    # 'rssm_cont_h_2048_lr3e-4': {
    #     'class': DreamerV2, 'config': 'rssm_cont_ds2_dec_hidden_size_2048.json', 
    #     'model_dir': 'rssm_cont_replay_early', 'model_label': 'RSSM Continuous Replay Early'},    # running N5, N1
    # 'md_mlp_2048_lr3e-4': {
    #     'class': MultistepDelta, 'config': 'multistep_delta_ds2_mlp_hidden_size_2048.json',
    #     'model_dir': 'multistep_delta_ds3', 'model_label': 'Multistep Delta DS3'},    # running N1, N2
    # 'tf_emb512_lr1e-4': {
    #     'class': TransformerWorldModel, 'config': 'tf_concat_pos_embd_emb_512_config.json',
    #     'model_dir': 'tf_emb512_lr1e-4', 'model_label': 'TF WM Emb512 LR1e-4'},    # running N2, N4
    # 'tf_emb1024_lr1e-4': {
    #     'class': TransformerWorldModel, 'config': 'tf_concat_pos_embd_emb_1024_config.json',
    #     'model_dir': 'tf_emb1024_lr1e-4', 'model_label': 'TF WM Emb1024 LR1e-4'},    # running N2, N4
    # 'tf_emb2048_lr1e-4': {
    #     'class': TransformerWorldModel, 'config': 'tf_concat_pos_embd_emb_2048_config.json', 
    #     'model_dir': 'tf_emb2048_lr1e-4', 'model_label': 'TF WM DS3 Emb2048 LR1e-4'},  
      
    'sgnet_cvae_lr1e-4': {
        'class': SGNet_CVAE, 'config': 'sgnet_cvae_default_config.json',
        'model_dir': 'sgnet_cvae_default_lr1e-4', 'model_label': 'SGNet-H512'},
    'sgnet_cvae_hidden64_lr1e-4': {
        'class': SGNet_CVAE, 'config': 'sgnet_cvae_hidden_size64_config.json',
        'model_dir': 'sgnet_cvae_hidden_size64_lr1e-4', 'model_label': 'SGNet-H64'},
    'sgnet_cvae_hidden128_lr1e-4': {
        'class': SGNet_CVAE, 'config': 'sgnet_cvae_hidden_size128_config.json',
        'model_dir': 'sgnet_cvae_hidden_size128_lr1e-4', 'model_label': 'SGNet-H128'},
    'sgnet_cvae_hidden256_lr1e-4': {
        'class': SGNet_CVAE, 'config': 'sgnet_cvae_hidden_size256_config.json',
        'model_dir': 'sgnet_cvae_hidden_size256_lr1e-4', 'model_label': 'SGNet-H256'},  
}
data_dir_ccn = '/mnt/fs2/ziyxiang/swm_data_and_results/data/'
checkpoint_dir_ccn = '/mnt/fs2/ziyxiang/swm_data_and_results/checkpoint/'
data_dir_ccn2 = '/ccn2/u/ziyxiang/swm_data_and_results/data/'
checkpoint_dir_ccn2 = '/ccn2/u/ziyxiang/swm_data_and_results/checkpoint/'
#checkpoint_dir_ccn2 = '/ccn2/u/ziyxiang/swm_data_and_results/submission_ckpt/'


DEFAULT_VALUES = {
    # general pipeline parameters
    'analysis_dir': './',    
    'data_dir': data_dir_ccn2 if os.path.isdir(data_dir_ccn2) else data_dir_ccn,
    'checkpoint_dir': checkpoint_dir_ccn2 if os.path.isdir(checkpoint_dir_ccn2) else checkpoint_dir_ccn,
    'model_config_dir': './model_configs',
    # general training parameters for all models
    'batch_size': 2048,
    'lr': 1e-5,
    'epochs': int(3e4),
    'save_every': 200,
    # eval parameters
    #'model_keys': [ 'trans_wm_replay_early', 'mp_replay_early', 'transformer_iris_concat_pos_embd_lr1e-4', 'mp_4096_ds3', 'md_4096_ds3',],
    'model_keys': list(MODEL_DICT_VAL.keys()),
    'eval_types': ['goal_events', 'multigoal_events', 'move_events', 'pickup_events', 'displacement'],
    'move_threshold': 4.0,
    'non_goal_burn_in': 50,
    'dataset': 'data_5_31_23.pkl'
}

DATASET_NUMS = {
    'train_test_splits_3D_dataset.pkl': 1,
    'dataset_5_25_23.pkl': 2, 
}

"""Valus for discretization data"""
DISCRETIZATION_DICT_SMALL = {
    'max': [
        6.7625, 1.8597, 6.8753, 6.7313, 1.6808, 6.8558, 6.3460, 1.0356, 
        6.8317, 6.8851, 0.3761, 6.8929, 0.7643, 1.0000, 0.7607, 1.0000, 
        6.8764, 0.3719, 6.8717, 0.7659, 1.0000, 0.7655, 1.0000],
    'min': [
        -6.7398, -0.0123, -6.8303, -6.8838, -0.0220, -6.7574, -6.8617, -0.0117, 
        -6.7947, -6.8901, -0.0426, -6.8758, -0.7560, -1.0000, -0.7604, -1.0000,
        -6.8851, -0.0326, -6.8671, -0.7600, -1.0000, -0.7641, -1.0000]
}
DISCRETIZATION_DICT_BIG = {}
