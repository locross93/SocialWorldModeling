from models import DreamerV2, MultistepPredictor, \
    TransformerMSPredictor, MultistepDelta, TransformerWorldModel
from gnn_models.imma import IMMA
from gnn_models.gat import GAT
from gnn_models.rfm import RFM
import platform


MODEL_DICT_TRAIN = {
    'dreamerv2': DreamerV2,
    'multistep_predictor': MultistepPredictor,    
    'transformer_mp': TransformerMSPredictor,
    'transformer_wm': TransformerWorldModel,
    'imma': IMMA,
    'gat': GAT,
    'rfm': RFM
}
MODEL_DICT_VAL=  {
    # 'rssm_cont': {
    #     'class': DreamerV2, 'config': 'rssm_cont_default_config.json', 
    #     'model_dir': 'rssm_cont_h1024_l2_mlp1024', 'model_label': 'RSSM Continuous'},
    'rssm_disc': {
        'class': DreamerV2, 'config': 'rssm_disc_default_config.json', 
        'model_dir': 'rssm_disc', 'model_label': 'RSSM Discrete'},
    'multistep_pred': {
        'class': MultistepPredictor, 'config': 'multistep_predictor_default_config.json', 
        'model_dir': 'multistep_predictor_ds1', 'model_label': 'Multistep Predictor'},
    'multistep_delta': {
        'class': MultistepDelta, 'config': 'multistep_delta_default_config.json', 
        'model_dir': 'multistep_delta_h1024_l2_mlp1024_l2', 'model_label': 'Multistep Delta'},
    # 'transformer': {
    #     'class': TransformerMSPredictor, 'config': 'transformer_default_config.json', 
    #     'model_dir': 'transformer_mp', 'model_label': 'Transformer MP'},
    'transformer_wm': {
       'class': TransformerWorldModel, 'config': 'transformer_wm_default_config.json', 
       'model_dir': 'transformer_wm', 'model_label': 'Transformer'},
    'gat': {
       'class': GAT, 'config': 'gat_default_config.json', 
       'model_dir': 'gat', 'model_label': 'GAT'},
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
    'rfm': {
       'class': RFM, 'config': 'rfm_default_config.json', 
       'model_dir': 'rfm', 'model_label': 'RFM'},
    'rfm_rnn': {
       'class': RFM, 'config': 'rfm_rnn_config.json', 
       'model_dir': 'rfm_rnn', 'model_label': 'RFM RNN'},
    'mp_ds2': {
        'class': MultistepPredictor, 'config': 'multistep_predictor_input_size_35.json', 
        'model_dir': 'multistep_predictor', 'model_label': 'Multistep Predictor'},
    'rssm_disc_ds2': {
        'class': DreamerV2, 'config': 'rssm_disc_ds2.json', 
        'model_dir': 'rssm_ds2', 'model_label': 'RSSM Discrete'},
    'rssm_cont_ds2': {
        'class': DreamerV2, 'config': 'rssm_cont_ds2.json', 
        'model_dir': 'rssm_cont_ds2', 'model_label': 'RSSM Continuous'},
}
DEFAULT_VALUES = {
    'analysis_dir': '/Users/locro/Documents/Stanford/SocialWorldModeling/' if platform.system() == 'Windows' else  '/home/locross/SocialWorldModeling/',
    'data_path': '/Users/locro/Documents/Stanford/analysis/data/dataset_5_25_23.pkl' if platform.system() == 'Windows' else  '/mnt/fs2/locross/analysis/data/dataset_5_25_23.pkl',
    'data_dir': '/Users/locro/Documents/Stanford/analysis/data/' if platform.system() == 'Windows' else  '/mnt/fs2/locross/analysis/data/',
    'checkpoint_dir': '/Users/locro/Documents/Stanford/SocialWorldModeling/' if platform.system() == 'Windows' else  '/mnt/fs2/locross/analysis/',
    'results_dir': '/Users/locro/Documents/Stanford/SocialWorldModeling/results/' if platform.system() == 'Windows' else  '/home/locross/SocialWorldModeling/results',
    'model_config_dir': './model_configs',
    # general training parameters for all models
    'batch_size': 8 if platform.system() == 'Windows' else 2048,
    'lr': 1e-5,
    'epochs': int(3e4),
    'save_every': 200,
    #'model_keys': list(MODEL_DICT_VAL.keys()),
    #'model_keys': ['multistep_pred'],
    #'model_keys': ['imma','gat','rfm','rfm_rnn'],
    'model_keys': ['mp_ds2', 'rssm_disc_ds2', 'rssm_cont_ds2'],
    'eval_types': ['goal_events', 'multigoal_events', 'move_events', 'pickup_events'],
    'move_threshold': 4.0,
    'non_goal_burn_in': 50,
}
DATASET_NUMS = {
    'train_test_splits_3D_dataset.pkl': 1,
    'dataset_5_25_23.pkl': 2,
}