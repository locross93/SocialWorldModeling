from models import DreamerV2, MultistepPredictor, \
    TransformerMSPredictor, MultistepDelta, TransformerWorldModel, TransformerIrisWorldModel, \
    EventPredictor, MSPredictorEventContext, EventModel, MultistepPredictor4D
from gnn_models.imma import IMMA
from gnn_models.gat import GAT
from gnn_models.rfm import RFM
import platform
import socket


MODEL_DICT_TRAIN = {
    'dreamerv2': DreamerV2,
    'multistep_predictor': MultistepPredictor,    
    'transformer_mp': TransformerMSPredictor,
    'transformer_wm': TransformerWorldModel,
    'imma': IMMA,
    'gat': GAT,
    'rfm': RFM,
    'multistep_delta': MultistepDelta,
    'event_predictor': EventPredictor,
    'mp_event_context': MSPredictorEventContext,
    'event_model': EventModel,
    'multistep_predictor4d': MultistepPredictor4D,
}
MODEL_DICT_VAL=  {
    # 'rssm_cont': {
    #     'class': DreamerV2, 'config': 'rssm_cont_default_config.json', 
    #     'model_dir': 'rssm_cont_h1024_l2_mlp1024', 'model_label': 'RSSM Continuous'},
    'rssm_disc': {
        'class': DreamerV2, 'config': 'rssm_disc_ds1.json', 
        'model_dir': 'rssm_disc_bi50_2', 'model_label': 'RSSM Discrete DS1'},
    'rssm_cont': {
        'class': DreamerV2, 'config': 'rssm_cont_ds1.json', 
        'model_dir': 'rssm_cont_h1024_l2_mlp1024', 'model_label': 'RSSM Continuous DS1'},
    'multistep_pred': {
        'class': MultistepPredictor, 'config': 'multistep_predictor_default_config.json', 
        'model_dir': 'multistep_predictor_ds1', 'model_label': 'Multistep Predictor DS1'},
    'multistep_delta': {
        'class': MultistepDelta, 'config': 'multistep_delta_default_config.json', 
        'model_dir': 'multistep_delta_h1024_l2_mlp1024_l2', 'model_label': 'Multistep Delta DS1'},
    # 'transformer': {
    #     'class': TransformerMSPredictor, 'config': 'transformer_default_config.json', 
    #     'model_dir': 'transformer_mp', 'model_label': 'Transformer MP'},
    'transformer_wm': {
       'class': TransformerWorldModel, 'config': 'transformer_wm_default_config.json', 
       'model_dir': 'transformer_wm', 'model_label': 'Transformer'},
    'gat': {
       'class': GAT, 'config': 'gat_default_config.json', 
       'model_dir': 'gat', 'model_label': 'GAT'},
    'gat_128': {
       'class': GAT, 'config': 'gat_encoder_rnn_config.json', 
       'model_dir': 'gat_128', 'model_label': 'GAT'},
    'gat_rnn_enc': {
       'class': GAT, 'config': 'gat_encoder_rnn_config.json', 
       'model_dir': 'gat_rnn_enc', 'model_label': 'GAT RNN Encoder'},
    'gat_rnn_norm_vel': {
       'class': GAT, 'config': 'gat_encoder_rnn_config.json', 
       'model_dir': 'gat_rnn_norm_vel', 'model_label': 'GAT RNN Encoder'},
    'imma': {
       'class': IMMA, 'config': 'imma_default_config.json', 
       'model_dir': 'imma', 'model_label': 'IMMA'},
    'imma_encoder': {
       'class': IMMA, 'config': 'imma_encoder_rnn.json', 
       'model_dir': 'imma_encoder', 'model_label': 'IMMA RNN Encoder'},
    'imma_rnn_simple_ds': {
       'class': IMMA, 'config': 'imma_rnn_simple_ds_hidden_dim_128.json', 
       'model_dir': 'imma_rnn_simple_ds', 'model_label': 'IMMA RNN Encoder'},
    'imma_rnn_norm_vel': {
       'class': IMMA, 'config': 'imma_encoder_rnn.json', 
       'model_dir': 'imma_rnn_norm_vel', 'model_label': 'IMMA RNN Encoder'},
    'imma_rnn_norm_vel_lr_5e-5': {
        'class': IMMA, 'config': 'imma_encoder_hidden_dim_1024.json',
        'model_dir': 'imma_encoder', 'model_label': 'IMMA RNN Encoder'},
    'rfm': {
       'class': RFM, 'config': 'rfm_default_config.json', 
       'model_dir': 'rfm', 'model_label': 'RFM'},
    'rfm_rnn': {
       'class': RFM, 'config': 'rfm_rnn_config.json', 
       'model_dir': 'rfm_rnn', 'model_label': 'RFM RNN'},
    'mp_ds2': {
        'class': MultistepPredictor, 'config': 'multistep_predictor_input_size_35.json', 
        'model_dir': 'multistep_predictor', 'epoch': '6000', 'model_label': 'Multistep Predictor'},
    'mp_ds3': {
        'class': MultistepPredictor, 'config': 'multistep_predictor_input_size_35.json',
        'model_dir': 'mp_ds3', 'model_label': 'Multistep Predictor DS3'},
    'md_ds3': {
        'class': MultistepDelta, 'config': 'multistep_delta_ds2.json',
        'model_dir': 'multistep_delta_ds3', 'model_label': 'Multistep Delta DS3'},
    'mp_norm_vel': {
        'class': MultistepPredictor, 'config': 'multistep_predictor_input_size_70.json',
        'model_dir': 'mp_norm_vel', 'model_label': 'Multistep Predictor Norm Velocity'},
    'rssm_disc_ds2': {
        'class': DreamerV2, 'config': 'rssm_disc_ds2.json', 
        'model_dir': 'rssm_ds2', 'model_label': 'RSSM Discrete DS2'},
    'rssm_cont_ds2': {
        'class': DreamerV2, 'config': 'rssm_cont_ds2.json', 
        'model_dir': 'rssm_cont_ds2', 'model_label': 'RSSM Continuous'},
    'rssm_disc_ds3': {
        'class': DreamerV2, 'config': 'rssm_disc_ds2.json', 
        'model_dir': 'rssm_disc_ds3', 'model_label': 'RSSM Discrete DS3'},
    'mp_replay_early': {
        'class': MultistepPredictor, 'config': 'multistep_predictor_input_size_35.json',
        'model_dir': 'mp_replay_early', 'model_label': 'MP Replay Early'},
    'trans_wm_replay_early': {
       'class': TransformerWorldModel, 'config': 'transformer_wm_ds2.json', 
       'model_dir': 'transformer_wm_replay_early', 'model_label': 'Transformer WM Replay Early'},
    'rssm_cont_ds3': {
        'class': DreamerV2, 'config': 'rssm_cont_ds2.json', 
        'model_dir': 'rssm_cont_replay_early', 'model_label': 'RSSM Continuous Replay Early'},
    'transformer_iris': {
        'class': TransformerIrisWorldModel, 'config': 'transformer_iris_concat_pos_embd_default_config.json',
        'model_dir': 'transformer_iris_concat_pos_embd_lr1e-4', 'model_label': 'Transformer Iris Concat Pos Embd lr1e-4'},
    'event_context_mp': {
        'class': EventModel, 'config': 'event_context_world_model.json',
        'model_dir': 'event_context_world', 'model_label': 'Event Context World Model'},
    'em_dropout': {
        'class': EventModel, 'config': 'event_model_dropout.json',
        'model_dir': 'em_dropout', 'model_label': 'Event Model Dropout: 0.1'},
    'event_model2': {
        'class': EventModel, 'config': 'event_context_world_model.json',
        'model_dir': 'event_model2', 'model_label': 'Event Model 2'},
    'emodel_no_horizon': {
        'class': EventModel, 'config': 'event_model_no_horizon.json',
        'model_dir': 'emodel_no_horizon', 'model_label': 'Event Model No Horizon'},
    'pred_end_state': {
        'class': EventModel, 'config': 'event_context_world_model.json',
        'model_dir': 'pred_end_state', 'model_label': 'Pred End State'},
    'em_2048': {
        'class': EventModel, 'config': 'event_model_hidden_2048.json',
        'model_dir': 'em_2048', 'model_label': 'Event Model 2048'},
    'em_lr5e-5': {
        'class': EventModel, 'config': 'event_context_world_model.json',
        'model_dir': 'em_lr5e-5', 'model_label': 'Event Model lr5e-5'},
    'em_min_horizon': {
        'class': EventModel, 'config': 'event_context_world_model.json',
        'model_dir': 'em_min_horizon', 'model_label': 'Event Model Min Horizon'},
    'rssm_norm_vel': {
        'class': DreamerV2, 'config': 'rssm_disc_input_size_70.json',
        'model_dir': 'rssm_norm_vel.pkl', 'model_label': 'RSSM Norm Vel'},
    'em_gt': {
        'class': EventModel, 'config': 'event_context_world_model.json',
        'model_dir': 'em_gt', 'model_label': 'Event Model GT'},
    'rssm_1step': {
        'class': DreamerV2, 'config': 'rssm_disc_default_config.json',
        'model_dir': 'rssm_1step', 'model_label': 'RSSM 1 Step Rollout'},
    'rssm_5step': {
        'class': DreamerV2, 'config': 'rssm_disc_default_config.json',
        'model_dir': 'rssm_5step', 'model_label': 'RSSM 5 Step Rollout'},
    'rssm_10step': {
        'class': DreamerV2, 'config': 'rssm_disc_default_config.json',
        'model_dir': 'rssm_10step', 'model_label': 'RSSM 10 Step Rollout'},
    'rssm_20step': {
        'class': DreamerV2, 'config': 'rssm_disc_default_config.json',
        'model_dir': 'rssm_20step', 'model_label': 'RSSM 20 Step Rollout'},
    'mp_frame_stack': {
        'class': MultistepPredictor4D, 'config': 'multistep_predictor_frame_stack.json',
        'model_dir': 'mp_frame_stack', 'model_label': 'MP Frame Stack'},
    'mp_mlp_2048_lr3e-4_s1': {
        'class': MultistepPredictor, 'config': 'multistep_predictor_input_size_35_mlp_hidden_size_2048.json',
        'model_dir': 'mp_mlp_2048_lr3e-4_s1', 'model_label': 'MP-S1'},
    'rssm_disc_h_2048_lr3e-4_s1': {
        'class': DreamerV2, 'config': 'rssm_disc_ds3_dec_hidden_size_2048.json', 
        'model_dir': 'rssm_disc_h_2048_lr3e-4_s1', 'model_label': 'RSSM-S1'},
    'rssm_cont_h_2048_lr1e-4_s1': {
        'class': DreamerV2, 'config': 'rssm_cont_ds2_dec_hidden_size_2048.json',
        'model_dir': 'rssm_cont_h_2048_lr1e-4_s1', 'model_label': 'RSSM-Cont-S1'},
    'md_mlp_2048_lr3e-4_s1' : {
        'class': MultistepPredictor, 'config': 'multistep_delta_ds2_mlp_hidden_size_2048.json',
        'model_dir': 'md_mlp_2048_lr3e-4_s1', 'model_label': 'MP-S1'},
    'tf_emb2048_lr1e-4_s1': {
        'class': TransformerWorldModel, 'config': 'tf_concat_pos_embd_emb_2048_config.json',
        'model_dir': 'tf_emb2048_lr1e-4_s1', 'model_label': 'TF Emb2048 S1'},
    'gt_end_state_s1': {
        'class': EventModel, 'config': 'event_context_world_model.json',
        'model_dir': 'gt_end_state_s1', 'model_label': 'GT End State S1'},
    'gt_end_state_s2': {
        'class': EventModel, 'config': 'event_context_world_model.json',
        'model_dir': 'gt_end_state_s2', 'model_label': 'GT End State S2'},
    'gt_end_state_s3': {
        'class': EventModel, 'config': 'event_context_world_model.json',
        'model_dir': 'gt_end_state_s3', 'model_label': 'GT End State S3'},
}
# Get the hostname of the machine
hostname = socket.gethostname()
if 'node05-ccncluster' in hostname:
    analysis_dir = '/home/locross/SocialWorldModeling/'
    data_path = '/data2/locross/data/swm_data.pkl'
    data_dir = '/data2/locross/data/'
    checkpoint_dir = '/data2/locross/analysis/'
    results_dir = '/home/locross/SocialWorldModeling/results'
    model_config_dir = '/home/locross/SocialWorldModeling/model_configs/'
elif 'DESKTOP-LU5SR6R' in hostname:
    analysis_dir = '/Users/locro/Documents/Stanford/SocialWorldModeling/'
    data_path = '/Users/locro/Documents/Stanford/analysis/data/swm_data.pkl'
    data_dir = '/Users/locro/Documents/Stanford/analysis/data/'
    checkpoint_dir = '/Users/locro/Documents/Stanford/SocialWorldModeling/'
    results_dir = '/Users/locro/Documents/Stanford/SocialWorldModeling/results/'
    model_config_dir = '/Users/locro/Documents/Stanford/SocialWorldModeling/model_configs/'
elif 'node4-ccn2cluster.stanford.edu' in hostname or 'node5-ccn2cluster.stanford.edu' in hostname:
    analysis_dir = '/home/locross/SocialWorldModeling/'
    data_path = '/ccn2/u/ziyxiang/swm_data_and_results/data/swm_data.pkl'
    data_dir = '/ccn2/u/ziyxiang/swm_data_and_results/data/'
    checkpoint_dir = '/ccn2/u/ziyxiang/swm_data_and_results/checkpoint/'
    results_dir = '/home/locross/SocialWorldModeling/results'
    model_config_dir = '/home/locross/SocialWorldModeling/model_configs/'
elif 'node02-ccncluster' in hostname or 'node07-ccncluster' in hostname:
    analysis_dir = '/home/locross/SocialWorldModeling/'
    data_path = '/mnt/fs2/locross/data/swm_data.pkl'
    data_dir = '/mnt/fs2/locross/data/'
    checkpoint_dir = '/mnt/fs2/locross/analysis/'
    results_dir = '/home/locross/SocialWorldModeling/results'
    model_config_dir = '/home/locross/SocialWorldModeling/model_configs/'

DEFAULT_VALUES = {
	'train_seed': 911320,
    'eval_seed': 911320, #834869, #(good for SGNet & RSSM Discrete) 
    'analysis_dir': analysis_dir,
    'data_path': data_path,
    'data_dir': data_dir,
    'checkpoint_dir': checkpoint_dir,
    'results_dir': results_dir,
    'model_config_dir': model_config_dir,
    # general training parameters for all models
    'batch_size': 8 if platform.system() == 'Windows' else 2048,
    'lr': 1e-5,
    'epochs': int(3e4),
    'save_every': 200,
    #'model_keys': list(MODEL_DICT_VAL.keys()),
    'model_keys': ['mp_mlp_2048_lr3e-4_s1','rssm_disc_h_2048_lr3e-4_s1','rssm_cont_h_2048_lr1e-4_s1','md_mlp_2048_lr3e-4_s1'],
    #'model_keys': ['imma','gat','rfm','rfm_rnn'],
    #'model_keys': ['rssm_disc_ds2', 'rssm_disc_ds3', 'rssm_cont_ds3', 'mp_ds3', 'md_ds3', 'transformer_iris'],
    #'model_keys': ['rssm_disc_ds3', 'mp_ds2', 'event_context_mp'],
    #'model_keys': ['rssm_1step', 'rssm_5step', 'rssm_10step', 'rssm_20step', 'rssm_disc_ds3'],
    #'model_keys': ['event_context_mp', 'em_dropout', 'event_model2', 'emodel_no_horizon', 'pred_end_state', 'em_2048', 'em_lr5e-5', 'em_min_horizon', 'em_gt'],    
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
