from models import DreamerV2, MultistepPredictor, \
    TransformerMSPredictor, MultistepDelta


MODEL_DICT_TRAIN = {
    'dreamerv2': DreamerV2,
    'multistep_predictor': MultistepPredictor,    
    'transformer_mp': TransformerMSPredictor,
    #'transformer_wm': TransformerWorldModel
}
MODEL_DICT_VAL=  {
    'rssm_disc': {
        'class': DreamerV2, 'config': 'rssm_disc_default_config.json', 
        'model_dir': 'rssm_disc', 'epoch': '1000', 'model_label': 'RSSM Discrete'},
    # 'multistep_pred': {
    #     'class': MultistepPredictor, 'config': 'multistep_predictor_default_config.json', 
    #     'model_dir': 'multistep_predictor', 'model_label': 'Multistep Predictor'},
    # 'transformer': {
    #     'class': TransformerMSPredictor, 'config': 'transformer_default_config.json', 
    #     'model_dir': 'transformer_mp', 'model_label': 'Transformer MP'},
    #'transformer_wm': {
    #    'class': TransformerWorldModel, 'config': 'transformer_wm_default_config.json', 
    #    'model_dir': 'transformer_wm', 'model_label': 'Transformer'},
}
DEFAULT_VALUES = {
    'analysis_dir': './',
    'data_dir': '/data2/ziyxiang/social_world_model/data',
    'checkpoint_dir': '/data2/ziyxiang/social_world_model/checkpoint',
    'model_config_dir': './model_configs',
    'model_keys': list(MODEL_DICT_VAL.keys()),
    'eval_types': ['goal_events', 'multigoal_events', 'move_events'],
    'move_threshold': 4.0,
    'non_goal_burn_in': 50,
}