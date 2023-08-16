import os
from models import DreamerV2, MultistepPredictor, MultistepDelta,  \
    TransformerMSPredictor, TransformerIrisWorldModel, TransformerWorldModel, \
    EventPredictor, MSPredictorEventContext, EventModel, MultistepPredictor4D, RSSM_Delta
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
    'rssm_delta': RSSM_Delta,
    'transformer_wm': TransformerWorldModel,   
    'transformer_mp': TransformerMSPredictor,
    'transformer_iris': TransformerIrisWorldModel,
    'imma': IMMA,
    'gat': GAT,
    'rfm': RFM,
    'sgnet_cvae': SGNet_CVAE,
    'agent_former': AgentFormer,
    'event_predictor': EventPredictor,
    'mp_event_context': MSPredictorEventContext,
    'event_model': EventModel,
    'multistep_predictor4d': MultistepPredictor4D,
}
"""Values for validation"""
# Only using 5-31-23 for the paper
# @TODO Clean up config file in this format 
# folder_name
# in each folder we save a json/pickle file of model_info
MODEL_DICT_VAL=  {
    'mp_mlp_2048_lr3e-4_s1': {
        'class': MultistepPredictor, 'config': 'multistep_predictor_input_size_35_mlp_hidden_size_2048.json',
        'model_dir': 'mp_mlp_2048_lr3e-4_s1', 'model_label': 'MP-S1', 'epoch': '5600'},
    'mp_mlp_2048_lr3e-4_s2': {
        'class': MultistepPredictor, 'config': 'multistep_predictor_input_size_35_mlp_hidden_size_2048.json',
        'model_dir': 'mp_mlp_2048_lr3e-4_s2', 'model_label': 'MP-S2'},
    'mp_mlp_2048_lr3e-4_s3': {
        'class': MultistepPredictor, 'config': 'multistep_predictor_input_size_35_mlp_hidden_size_2048.json',
        'model_dir': 'mp_mlp_2048_lr3e-4_s3', 'model_label': 'MP-S3'},
    'rssm_disc_h_2048_lr3e-4_s1': {
        'class': DreamerV2, 'config': 'rssm_disc_ds3_dec_hidden_size_2048.json', 
        'model_dir': 'rssm_disc_h_2048_lr3e-4_s1', 'model_label': 'RSSM-S1'},
    'rssm_disc_h_2048_lr3e-4_s2': {
        'class': DreamerV2, 'config': 'rssm_disc_ds3_dec_hidden_size_2048.json',
        'model_dir': 'rssm_disc_h_2048_lr3e-4_s2', 'model_label': 'RSSM-S2'},
    'rssm_disc_h_2048_lr3e-4_s3': {
        'class': DreamerV2, 'config': 'rssm_disc_ds3_dec_hidden_size_2048.json',
        'model_dir': 'rssm_disc_h_2048_lr3e-4_s3', 'model_label': 'RSSM-S3'},
    'rssm_cont_h_2048_lr1e-4_s1': {
        'class': DreamerV2, 'config': 'rssm_cont_ds2_dec_hidden_size_2048.json',
        'model_dir': 'rssm_cont_h_2048_lr1e-4_s1', 'model_label': 'RSSM-Cont-S1'},
    'rssm_cont_h_2048_lr1e-4_s2': {
        'class': DreamerV2, 'config': 'rssm_cont_ds2_dec_hidden_size_2048.json',
        'model_dir': 'rssm_cont_h_2048_lr1e-4_s2', 'model_label': 'RSSM-Cont-S2'},
    'rssm_cont_h_2048_lr1e-4_s3': {
        'class': DreamerV2, 'config': 'rssm_cont_ds2_dec_hidden_size_2048.json',
        'model_dir': 'rssm_cont_h_2048_lr1e-4_s3', 'model_label': 'RSSM-Cont-S3'},
    'md_mlp_2048_lr3e-4_s1' : {
        'class': MultistepPredictor, 'config': 'multistep_delta_ds2_mlp_hidden_size_2048.json',
        'model_dir': 'md_mlp_2048_lr3e-4_s1', 'model_label': 'MP-S1'},
    'md_mlp_2048_lr3e-4_s2' : {
        'class': MultistepPredictor, 'config': 'multistep_delta_ds2_mlp_hidden_size_2048.json',
        'model_dir': 'md_mlp_2048_lr3e-4_s2', 'model_label': 'MD-S2'},
    'md_mlp_2048_lr3e-4_s3' : {
        'class': MultistepPredictor, 'config': 'multistep_delta_ds2_mlp_hidden_size_2048.json',
        'model_dir': 'md_mlp_2048_lr3e-4_s3', 'model_label': 'MD-S3'},
    # 'tf_emb512_lr1e-4_s1': {
    #     'class': TransformerWorldModel, 'config': 'tf_concat_pos_embd_emb_512_config.json',
    #     'model_dir': 'tf_emb512_lr1e-4_s1', 'model_label': 'TF Emb512 S1'},
    # 'tf_emb512_lr1e-4_s2': {
    #     'class': TransformerWorldModel, 'config': 'tf_concat_pos_embd_emb_512_config.json',
    #     'model_dir': 'tf_emb512_lr1e-4_s2', 'model_label': 'TF Emb512 S2'},
    # 'tf_emb512_lr1e-4_s3': {
    #     'class': TransformerWorldModel, 'config': 'tf_concat_pos_embd_emb_512_config.json',
    #     'model_dir': 'tf_emb512_lr1e-4_s3', 'model_label': 'TF Emb512 S3'},
    # 'tf_emb1024_lr1e-4_s1': {
    #     'class': TransformerWorldModel, 'config': 'tf_concat_pos_embd_emb_1024_config.json',
    #     'model_dir': 'tf_emb1024_lr1e-4_s1', 'model_label': 'TF Emb1024 S1'},
    # 'tf_emb1024_lr1e-4_s2': {
    #     'class': TransformerWorldModel, 'config': 'tf_concat_pos_embd_emb_1024_config.json',
    #     'model_dir': 'tf_emb1024_lr1e-4_s2', 'model_label': 'TF Emb1024 S2'},
    # 'tf_emb1024_lr1e-4_s3': {
    #     'class': TransformerWorldModel, 'config': 'tf_concat_pos_embd_emb_1024_config.json',
    #     'model_dir': 'tf_emb1024_lr1e-4_s3', 'model_label': 'TF Emb1024 S3'},
    'tf_emb2048_lr1e-4_s1': {
        'class': TransformerWorldModel, 'config': 'tf_concat_pos_embd_emb_2048_config.json',
        'model_dir': 'tf_emb2048_lr1e-4_s1', 'model_label': 'TF Emb2048 S1'},
    'tf_emb2048_lr1e-4_s2': {
        'class': TransformerWorldModel, 'config': 'tf_concat_pos_embd_emb_2048_config.json',
        'model_dir': 'tf_emb2048_lr1e-4_s2', 'model_label': 'TF Emb2048 S2', "epoch": "400"},
    'tf_emb2048_lr1e-4_s3': {
        'class': TransformerWorldModel, 'config': 'tf_concat_pos_embd_emb_2048_config.json',
        'model_dir': 'tf_emb2048_lr1e-4_s3', 'model_label': 'TF Emb2048 S3'},
    'sgnet_cvae_lr1e-4': {
        'class': SGNet_CVAE, 'config': 'sgnet_cvae_default_config.json',
        'model_dir': 'sgnet_cvae_default_lr1e-4', 'model_label': 'SGNet-H512'},
    # 'sgnet_cvae_hidden64_lr1e-4': {
    #     'class': SGNet_CVAE, 'config': 'sgnet_cvae_hidden_size64_config.json',
    #     'model_dir': 'sgnet_cvae_hidden_size64_lr1e-4', 'model_label': 'SGNet-H64'},
    # 'sgnet_cvae_hidden128_lr1e-4': {
    #     'class': SGNet_CVAE, 'config': 'sgnet_cvae_hidden_size128_config.json',
    #     'model_dir': 'sgnet_cvae_hidden_size128_lr1e-4', 'model_label': 'SGNet-H128'},
    # 'sgnet_cvae_hidden256_lr1e-4': {
    #     'class': SGNet_CVAE, 'config': 'sgnet_cvae_hidden_size256_config.json',
    #     'model_dir': 'sgnet_cvae_hidden_size256_lr1e-4', 'model_label': 'SGNet-H256'},  
}
data_dir_ccn = '/mnt/fs2/ziyxiang/swm_data_and_results/data/'
checkpoint_dir_ccn = '/mnt/fs2/ziyxiang/swm_data_and_results/checkpoint/'
data_dir_ccn2 = '/ccn2/u/ziyxiang/swm_data_and_results/data/'
#checkpoint_dir_ccn2 = '/ccn2/u/ziyxiang/swm_data_and_results/checkpoint/'   # this has SGNet
checkpoint_dir_ccn2 = '/ccn2/u/ziyxiang/swm_data_and_results/submission_ckpt/'


DEFAULT_VALUES = {
    'train_seed': 911320,
    'eval_seed': 911320, #834869, #(good for SGNet & RSSM Discrete) 
    #254438,  (18%),# 911320 (16.6%),# 665218 (#2), #87172, (#1)
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
    'model_keys': ['mp_mlp_2048_lr3e-4_s1', 'mp_mlp_2048_lr3e-4_s2', 'mp_mlp_2048_lr3e-4_s3', 
                   'rssm_disc_h_2048_lr3e-4_s1', 'rssm_disc_h_2048_lr3e-4_s2', 'rssm_disc_h_2048_lr3e-4_s3', 
                    'rssm_cont_h_2048_lr1e-4_s1', 'rssm_cont_h_2048_lr1e-4_s2', 'rssm_cont_h_2048_lr1e-4_s3',
                    'md_mlp_2048_lr3e-4_s1', 'md_mlp_2048_lr3e-4_s2', 'md_mlp_2048_lr3e-4_s3',
                    'tf_emb2048_lr1e-4_s1', 'tf_emb2048_lr1e-4_s2', 'tf_emb2048_lr1e-4_s3'],
    #'model_keys': list(MODEL_DICT_VAL.keys()),
    'eval_types': ['goal_events', 'multigoal_events', 'move_events', 'pickup_events', 'displacement'],
    'move_threshold': 4.0,
    'non_goal_burn_in': 50,
    'dataset': 'swm_data.pkl'
}

DATASET_NUMS = {
    'train_test_splits_3D_dataset.pkl': 1,
    'dataset_5_25_23.pkl': 2,
    'swm_data.pkl': 3,
    'data_norm_velocity.pkl': 4,
}

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
