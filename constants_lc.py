from models import DreamerV2, MultistepPredictor, \
    TransformerMSPredictor, MultistepDelta, TransformerWorldModel
from gnn_models.imma import IMMA
from gnn_models.gat import GAT
import platform


MODEL_DICT_TRAIN = {
    'dreamerv2': DreamerV2,
    'multistep_predictor': MultistepPredictor,    
    'transformer_mp': TransformerMSPredictor,
    'transformer_wm': TransformerWorldModel,
    'imma': IMMA,
    'gat': GAT
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
        'model_dir': 'mp_input_embed_h1024_l2_mlp1024_l2', 'model_label': 'Multistep Predictor'},
    'multistep_delta': {
        'class': MultistepDelta, 'config': 'multistep_delta_default_config.json', 
        'model_dir': 'multistep_delta_h1024_l2_mlp1024_l2', 'model_label': 'Multistep Delta'},
    # 'transformer': {
    #     'class': TransformerMSPredictor, 'config': 'transformer_default_config.json', 
    #     'model_dir': 'transformer_mp', 'model_label': 'Transformer MP'},
    'transformer_wm': {
       'class': TransformerWorldModel, 'config': 'transformer_wm_default_config.json', 
       'model_dir': 'transformer_wm', 'model_label': 'Transformer'},
}
DEFAULT_VALUES = {
    'analysis_dir': '/Users/locro/Documents/Stanford/SocialWorldModeling/' if platform.system() == 'Windows' else  '/home/locross/SocialWorldModeling/',
    'data_dir': '/Users/locro/Documents/Stanford/analysis/data/' if platform.system() == 'Windows' else  '/mnt/fs2/locross/analysis/data/',
    'checkpoint_dir': '/Users/locro/Documents/Stanford/SocialWorldModeling/' if platform.system() == 'Windows' else  '/mnt/fs2/locross/analysis/',
    'model_config_dir': './model_configs',
    'model_keys': list(MODEL_DICT_VAL.keys()),
    'eval_types': ['goal_events', 'multigoal_events', 'move_events'],
    'move_threshold': 4.0,
    'non_goal_burn_in': 50,
}
DATASET_NUMS = {
    'train_test_splits_3D_dataset.pkl': 1,
    'dataset_5_25_23.pkl': 2,
}