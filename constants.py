from models import DreamerV2, MultistepPredictor, \
    TransformerMSPredictor, TransformerIrisWorldModel, TransformerWorldModel


MODEL_DICT_TRAIN = {
    'dreamerv2': DreamerV2,
    'multistep_predictor': MultistepPredictor, 
    'transformer_wm': TransformerWorldModel,   
    'transformer_mp': TransformerMSPredictor,
    'transformer_iris': TransformerIrisWorldModel
}

MODEL_DICT_VAL=  {
    'rssm_disc': {
        'class': DreamerV2, 'config': 'rssm_disc_default_config.json', 
        'model_dir': 'rssm_disc', 'epoch': '1000', 'model_label': 'RSSM Discrete'},
    'transformer_iris': {
        'class': TransformerIrisWorldModel, 'config': 'transformer_iris_config.json',
        'model_dir': 'transformer_iris', 'epoch': '1000', 'model_label': 'Transformer Iris'},
    'transformer_iris_dropout_0.1': {
        'class': TransformerIrisWorldModel, 'config': 'transformer_iris_dropout_0.1_config.json',
        'model_dir': 'transformer_iris_dropout_0.1', 'epoch': '1000', 'model_label': 'Transformer Iris Dropout 0.1'},
}


DEFAULT_VALUES = {
    # general pipeline parameters
    'analysis_dir': './',
    'data_dir': '/data2/ziyxiang/social_world_model/data',
    'checkpoint_dir': '/data2/ziyxiang/social_world_model/checkpoint',
    'model_config_dir': './model_configs',
    # general training parameters for all models
    'batch_size': 2048,
    'lr': 1e-4,
    'epochs': int(1e5),
    'save_every': 500,
    # eval parameters
    'model_keys': list(MODEL_DICT_VAL.keys()),
    'eval_types': ['goal_events', 'multigoal_events', 'move_events'],
    'move_threshold': 4.0,
    'non_goal_burn_in': 50,
}