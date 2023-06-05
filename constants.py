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
MODEL_DICT_VAL=  {
    # 'sgnet_cvae_lr1e-4': {
    #     'class': SGNet_CVAE, 'config': 'sgnet_cvae_default_config.json',
    #     'model_dir': 'sgnet_cvae_default_lr1e-4', 'epoch': '200', 'model_label': 'SGNet-H512'},
    # 'sgnet_cvae_hidden64_lr1e-4': {
    #     'class': SGNet_CVAE, 'config': 'sgnet_cvae_hidden_size64_config.json',
    #     'model_dir': 'sgnet_cvae_hidden_size64_lr1e-4', 'epoch': '200', 'model_label': 'SGNet-H64'},
    # 'sgnet_cvae_hidden128_lr1e-4': {
    #     'class': SGNet_CVAE, 'config': 'sgnet_cvae_hidden_size128_config.json',
    #     'model_dir': 'sgnet_cvae_hidden_size128_lr1e-4', 'epoch': '200', 'model_label': 'SGNet-H128'},
    # 'sgnet_cvae_hidden256_lr1e-4': {
    #     'class': SGNet_CVAE, 'config': 'sgnet_cvae_hidden_size64_config.json',
    #     'model_dir': 'sgnet_cvae_hidden_size256_lr1e-4', 'epoch': '200', 'model_label': 'SGNet-H256'},
    # 'agent_former': {
    #     'class': AgentFormer, 'config': 'agent_former_default_config.json',
    #     'model_dir': 'agent_former_default', 'epoch': '200', 'model_label': 'AgentFormer'},
    # 'rssm_disc': {
    #     'class': DreamerV2, 'config': 'rssm_disc_default_config.json', 
    #     'model_dir': 'rssm_disc_default', 'epoch': '3000', 'model_label': 'RSSM Discrete'},
    # 'multistep_predictor': {
    #     'class': MultistepPredictor, 'config': 'multistep_predictor_default_config.json',
    #     'model_dir': 'multistep_predictor_default', 'epoch': '3000', 'model_label': 'Multistep Predictor'},
    # 'multistep_delta': {
    #     'class': MultistepDelta, 'config': 'multistep_delta_default_config.json',
    #     'model_dir': 'multistep_delta_default', 'epoch': '3000', 'model_label': 'Multistep Delta'},
    # # 'transformer_iris_default': {
    # #     'class': TransformerIrisWorldModel, 'config': 'transformer_iris_default_config.json',
    # #     'model_dir': 'transformer_iris_default', 'epoch': '29000', 'model_label': 'Transformer Iris'},
    # 'gat_rnn_norm_vel': {
    #    'class': GAT, 'config': 'gat_encoder_rnn_config.json', 
    #    'model_dir': 'gat_rnn_norm_vel', 'model_label': 'GAT RNN Encoder'},
    # 'transformer_iris_concat_pos_embd_lr3e-4': {
    #     'class': TransformerIrisWorldModel, 'config': 'transformer_iris_concat_pos_embd_default_config.json',
    #     'model_dir': 'transformer_iris_concat_pos_embd_lr3e-4', 'epoch': '8000', 'model_label': 'Iris Concat lr3e-4'},
    # """Best transformer model (5-23-23)"""
    # 'transformer_iris_concat_pos_embd_lr1e-4': {
    #     'class': TransformerIrisWorldModel, 'config': 'transformer_iris_concat_pos_embd_default_config.json',
    #     'model_dir': 'transformer_iris_concat_pos_embd_lr1e-4', 'model_label': 'Transformer Iris Concat Pos Embd lr1e-4'},
    # 'transformer_iris_concat_pos_embd_lr1e-5': {
    #     'class': TransformerIrisWorldModel, 'config': 'transformer_iris_concat_pos_embd_default_config.json',
    #     'model_dir': 'transformer_iris_concat_pos_embd_lr3e-5', 'epoch': '8000', 'model_label': 'Iris Concat lr3e-5'},
    # 'transformer_iris_concat_pos_embd_lr1e-5': {
    #     'class': TransformerIrisWorldModel, 'config': 'transformer_iris_concat_pos_embd_default_config.json',
    #     'model_dir': 'transformer_iris_concat_pos_embd_lr1e-5', 'epoch': '8000', 'model_label': 'Iris Concat lr1e-5'},
    # 'transformer_iris_default': {
    #     'class': TransformerIrisWorldModel, 'config': 'transformer_iris_default_config.json',
    #     'model_dir': 'transformer_iris_default', 'epoch': '8000', 'model_label': 'Iris Default'},
    # 'transformer_iris_lr1e-4': {
    #     'class': TransformerIrisWorldModel, 'config': 'transformer_iris_default_config.json',
    #     'model_dir': 'transformer_iris_lr1e-4', 'epoch': '8000', 'model_label': 'Iris lr1e-4'},
    # 'transformer_iris_lr1e-5/': {
    #     'class': TransformerIrisWorldModel, 'config': 'transformer_iris_default_config.json',
    #     'model_dir': 'transformer_iris_lr1e-5', 'epoch': '8000', 'model_label': 'Iris lr1e-5'},
    # 'transformer_iris_lr3e-4': {
    #     'class': TransformerIrisWorldModel, 'config': 'transformer_iris_default_config.json',
    #     'model_dir': 'transformer_iris_lr3e-4', 'epoch': '8000', 'model_label': 'Iris lr3e-4'},
    # 'rfm_rnn': {
    #    'class': RFM, 'config': 'rfm_rnn_config.json', 
    #    'model_dir': 'rfm_rnn', 'model_label': 'RFM RNN'},
    # 'gat': {
    #    'class': GAT, 'config': 'gat_default_config.json', 
    #    'model_dir': 'gat', 'model_label': 'GAT'},
    # 'imma': {
    #    'class': IMMA, 'config': 'imma_default_config.json', 
    #    'model_dir': 'imma', 'model_label': 'IMMA'},
    # 'mp_4096_ds3': {
    #     'class': MultistepPredictor, 'config': 'mp_4096_ds3_rnn_hidden_size_4096.json',
    #     'model_dir': 'mp_4096_ds3', 'model_label': 'MP 4096 DS3'},
    # 'md_4096_ds3': {
    #     'class': MultistepPredictor, 'config': 'multistep_delta_ds3_rnn_hidden_size_4096.json',
    #     'model_dir': 'multistep_delta_ds3', 'model_label': 'MD 4096 DS3'},
    # 'mp_replay_early': {
    #     'class': MultistepPredictor, 'config': 'multistep_predictor_input_size_35.json',
    #     'model_dir': 'mp_replay_early', 'model_label': 'MP Replay Early'},
    # 'trans_wm_replay_early': {
    #    'class': TransformerWorldModel, 'config': 'transformer_wm_ds2.json', 
    #    'model_dir': 'transformer_wm_replay_early', 'model_label': 'Transformer WM Replay Early'},

    'md_4096_d3_lr1e-5': {
        'class': MultistepPredictor, 'config': 'multistep_delta_ds2_mlp_hidden_size_4096.json', 
        'model_dir': 'multistep_delta_ds3_mlp_hidden_size_4096_lr1e-5', 'model_label': 'MD 4096 D3 lr1e-5'},
    'md_4096_d3_lr1e-4': {
        'class': MultistepPredictor, 'config': 'multistep_delta_ds2_mlp_hidden_size_4096.json',
        'model_dir': 'multistep_delta_ds3_mlp_hidden_size_4096_lr1e-4', 'model_label': 'MD 4096 D3 lr1e-4'},
    'md_4096_d3_lr3e-4': {
        'class': MultistepPredictor, 'config': 'multistep_delta_ds2_mlp_hidden_size_4096.json',
        'model_dir': 'multistep_delta_ds3_mlp_hidden_size_4096_lr3e-4', 'model_label': 'MD 4096 D3 lr3e-4'},
    'md_4096_d3_lr3e-5': {
        'class': MultistepPredictor, 'config': 'multistep_delta_ds2_mlp_hidden_size_4096.json',
        'model_dir': 'multistep_delta_ds3_mlp_hidden_size_4096_lr3e-5', 'model_label': 'MD 4096 D3 lr3e-5'},
    'md_2048_d3_lr1e-5': {
        'class': MultistepPredictor, 'config': 'multistep_delta_ds2_mlp_hidden_size_2048.json',
        'model_dir': 'multistep_delta_ds3_mlp_hidden_size_2048_lr1e-5', 'model_label': 'MD 2048 D3 lr1e-5'},
    'md_2048_d3_lr1e-4': {
        'class': MultistepPredictor, 'config': 'multistep_delta_ds2_mlp_hidden_size_2048.json',
        'model_dir': 'multistep_delta_ds3_mlp_hidden_size_2048_lr1e-4', 'model_label': 'MD 2048 D3 lr1e-4'},
    'md_2048_d3_lr3e-4': {
        'class': MultistepPredictor, 'config': 'multistep_delta_ds2_mlp_hidden_size_2048.json',
        'model_dir': 'multistep_delta_ds3_mlp_hidden_size_2048_lr3e-4', 'model_label': 'MD 2048 D3 lr3e-4'},
    'md_2048_d3_lr3e-5': {
        'class': MultistepPredictor, 'config': 'multistep_delta_ds2_mlp_hidden_size_2048.json',
        'model_dir': 'multistep_delta_ds3_mlp_hidden_size_2048_lr3e-5', 'model_label': 'MD 2048 D3 lr3e-5'},
    'md_512_d3_lr1e-5': {
        'class': MultistepPredictor, 'config': 'multistep_delta_ds2_mlp_hidden_size_512.json',
        'model_dir': 'multistep_delta_ds3_mlp_hidden_size_512_lr1e-5', 'model_label': 'MD 512 D3 lr1e-5'},
    'md_512_d3_lr1e-4': {
        'class': MultistepPredictor, 'config': 'multistep_delta_ds2_mlp_hidden_size_512.json',
        'model_dir': 'multistep_delta_ds3_mlp_hidden_size_512_lr1e-4', 'model_label': 'MD 512 D3 lr1e-4'},
    'md_512_d3_lr3e-4': {
        'class': MultistepPredictor, 'config': 'multistep_delta_ds2_mlp_hidden_size_512.json',
        'model_dir': 'multistep_delta_ds3_mlp_hidden_size_512_lr3e-4', 'model_label': 'MD 512 D3 lr3e-4'},
    'md_512_d3_lr3e-5': {
        'class': MultistepPredictor, 'config': 'multistep_delta_ds2_mlp_hidden_size_512.json',
        'model_dir': 'multistep_delta_ds3_mlp_hidden_size_512_lr3e-5', 'model_label': 'MD 512 D3 lr3e-5'},    
    'md_d3_lr1e-4': {
        'class': MultistepPredictor, 'config': 'multistep_delta_ds2.json',
        'model_dir': 'multistep_delta_ds3_lr1e-4', 'model_label': 'MD D3 lr1e-4'},
    'md_d3_lr3e-4': {
        'class': MultistepPredictor, 'config': 'multistep_delta_ds2.json',
        'model_dir': 'multistep_delta_ds3_lr3e-4', 'model_label': 'MD D3 lr3e-4'},
    'md_d3_lr3e-5': {
        'class': MultistepPredictor, 'config': 'multistep_delta_ds2.json',
        'model_dir': 'multistep_delta_ds3_lr3e-5', 'model_label': 'MD D3 lr3e-5'},
    # 'iris_vel_lr1e-4': {
    #     'class': TransformerIrisWorldModel, 'config': 'transformer_iris_concat_pos_embd_input_size_70_config.json',
    #     'model_dir': 'transformer_iris_concat_pos_embd_input_size_70_lr1e-4', 'model_label': 'Iris Vel lr1e-4'},
    # 'iris_vel_lr3e-4': {
    #     'class': TransformerIrisWorldModel, 'config': 'transformer_iris_concat_pos_embd_input_size_70_config.json',
    #     'model_dir': 'transformer_iris_concat_pos_embd_input_size_70_lr3e-4', 'model_label': 'Iris Vel lr3e-4'},
    # 'iris_vel_lr1e-5': {
    #     'class': TransformerIrisWorldModel, 'config': 'transformer_iris_concat_pos_embd_input_size_70_config.json',
    #     'model_dir': 'transformer_iris_concat_pos_embd_input_size_70_lr3e-5', 'model_label': 'Iris Vel lr3e-5'},
    # 'iris_norm_lr1e-4': {
    #     'class': TransformerIrisWorldModel, 'config': 'transformer_iris_concat_pos_embd_norm_config.json',
    #     'model_dir': 'transformer_iris_concat_pos_embd_norm_lr1e-4', 'model_label': 'Iris Norm lr1e-4'},
    # 'iris_norm_lr3e-4': {
    #     'class': TransformerIrisWorldModel, 'config': 'transformer_iris_concat_pos_embd_norm_config.json',
    #     'model_dir': 'transformer_iris_concat_pos_embd_norm_lr3e-4', 'model_label': 'Iris Norm lr3e-4'},
    # 'iris_norm_lr1e-5': {
    #     'class': TransformerIrisWorldModel, 'config': 'transformer_iris_concat_pos_embd_norm_config.json',
    #     'model_dir': 'transformer_iris_concat_pos_embd_norm_lr3e-5', 'model_label': 'Iris Norm lr3e-5'},
    'rssm_cont_h512_lr1e-4': {
        'class': DreamerV2, 'config': 'rssm_cont_ds2_dec_hidden_size_512.json',
        'model_dir': 'rssm_cont_ds3_dec_hidden_size_512_lr1e-4', 'model_label': 'RSSM Cont H512_lr1e-4'},
    'rssm_cont_h1024_lr1e-4': {
        'class': DreamerV2, 'config': 'rssm_cont_ds2.json',
        'model_dir': 'rssm_cont_ds3_dec_hidden_size_1024_lr1e-4', 'model_label': 'RSSM Cont H1024_lr1e-4'},
    'rssm_cont_h2048_lr1e-4': {
        'class': DreamerV2, 'config': 'rssm_cont_ds2_dec_hidden_size_2048.json',
        'model_dir': 'rssm_cont_ds3_dec_hidden_size_2048_lr1e-4', 'model_label': 'RSSM Cont H2048_lr1e-4'},
    'rssm_cont_h2048_lr3e-4': {
        'class': DreamerV2, 'config': 'rssm_cont_ds2_dec_hidden_size_2048.json',
        'model_dir': 'rssm_cont_ds3_dec_hidden_size_2048_lr3e-4', 'model_label': 'RSSM Cont H2048_lr3e-4'},
    'rssm_cont_h2048_lr3e-5': {
        'class': DreamerV2, 'config': 'rssm_cont_ds2_dec_hidden_size_2048.json',
        'model_dir': 'rssm_cont_ds3_dec_hidden_size_2048_lr3e-5', 'model_label': 'RSSM Cont H2048_lr3e-5'},    
    'rssm_cont_h4096_lr1e-4': {
        'class': DreamerV2, 'config': 'rssm_cont_ds2_dec_hidden_size_4096.json',
        'model_dir': 'rssm_cont_ds3_dec_hidden_size_4096_lr1e-4', 'model_label': 'RSSM Cont H4096_lr1e-4'},
    'rssm_cont_h4096_lr3e-4': {
        'class': DreamerV2, 'config': 'rssm_cont_ds2_dec_hidden_size_4096.json',
        'model_dir': 'rssm_cont_ds3_dec_hidden_size_4096_lr3e-4', 'model_label': 'RSSM Cont H4096_lr3e-4'},
    'rssm_cont_h4096_lr3e-5': {
        'class': DreamerV2, 'config': 'rssm_cont_ds2_dec_hidden_size_4096.json',
        'model_dir': 'rssm_cont_ds3_dec_hidden_size_4096_lr3e-5', 'model_label': 'RSSM Cont H4096_lr3e-5'},
}
data_dir_ccn = '/mnt/fs2/ziyxiang/swm_data_and_results/data/'
checkpoint_dir_ccn = '/mnt/fs2/ziyxiang/swm_data_and_results/checkpoint/'
data_dir_ccn2 = '/ccn2/u/ziyxiang/swm_data_and_results/data/'
checkpoint_dir_ccn2 = '/ccn2/u/ziyxiang/swm_data_and_results/checkpoint/'

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
    'eval_types': ['goal_events', 'multigoal_events', 'move_events', 'pickup_events'],
    'move_threshold': 4.0,
    'non_goal_burn_in': 50,
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
