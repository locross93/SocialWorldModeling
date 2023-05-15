from models import DreamerV2, MultistepPredictor, \
    TransformerMSPredictor, MultistepDelta

DEFAULT_VALUES = {
    'analysis_dir': './',
    'data_dir': '/data2/ziyxiang/social_world_model/data',
    'checkpoint_dir': '/data2/ziyxiang/social_world_model/checkpoint',
    'eval_types': ['move_events', 'pickup_events', 'goal_events'],
}
MODEL_DICT_TRAIN = {
    'dreamerv2': DreamerV2,
    'multistep_predictor': MultistepPredictor,
    'transformer_multistep_predictor': TransformerMSPredictor
}
MODEL_DICT_VAL=  {
    'rssm_disc': {
        'class': DreamerV2, 'config': 'rssm_disc_default_config.json', 
        'model_dir': 'rssm_disc', 'epoch': '1000', 'model_label': 'RSSM Discrete'},
}