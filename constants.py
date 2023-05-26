from models import DreamerV2, MultistepPredictor, \
    TransformerMSPredictor, TransformerIrisWorldModel, TransformerWorldModel


MODEL_DICT_TRAIN = {
    'rssm_disc': DreamerV2,
    'multistep_predictor': MultistepPredictor, 
    'transformer_wm': TransformerWorldModel,   
    'transformer_mp': TransformerMSPredictor,
    'transformer_iris': TransformerIrisWorldModel
}

MODEL_DICT_VAL=  {
    # 'rssm_disc': {
    #     'class': DreamerV2, 'config': 'rssm_disc_default_config.json', 
    #     'model_dir': 'rssm_disc', 'epoch': '1000', 'model_label': 'RSSM Discrete'},
    # 'transformer_iris': {
    #     'class': TransformerIrisWorldModel, 'config': 'transformer_iris_config.json',
    #     'model_dir': 'transformer_iris', 'epoch': '5000', 'model_label': 'Transformer Iris'},
    'transformer_iris_dropout_0.1': {
         'class': TransformerIrisWorldModel, 'config': 'transformer_iris_dropout_0.1_config.json',
         'model_dir': 'transformer_iris_dropout', 'epoch': '3000', 'model_label': 'Transformer Iris Dropout 0.1'},
    # 'transformer_wm': {
    #     'class': TransformerWorldModel, 'config': 'transformer_wm_default_config.json',
    #     'model_dir': 'transformer_wm_default', 'epoch': '5000', 'model_label': 'Transformer WM Default'},
    # 'transformer_mp': {
    #     'class': TransformerMSPredictor, 'config': 'transformer_mp_default_config.json',
    #     'model_dir': 'transformer_mp_default', 'epoch': '5000', 'model_label': 'Transformer MP Default'},
    'transformer_iris_mp_window_10': {
        'class': TransformerIrisWorldModel, 'config': 'transformer_iris_mp_window_10_config.json',
        'model_dir': 'transformer_iris_mp_window_10', 'epoch': '3000', 'model_label': 'Transformer Iris MP Window 10'},
    # 'transformer_iris_mp_window_20': {
    #     'class': TransformerIrisWorldModel, 'config': 'transformer_iris_mp_window_20_config.json',
    #     'model_dir': 'transformer_iris_mp_window_20', 'epoch': '3000', 'model_label': 'Transformer Iris MP Window 20'},
    'transformer_iris_mp_window_30': {
        'class': TransformerIrisWorldModel, 'config': 'transformer_iris_mp_window_30_config.json',
        'model_dir': 'transformer_iris_mp_window_30', 'epoch': '3000', 'model_label': 'Transformer Iris MP Window 30'},
    # 'transformer_iris_mp_window_40': {
    #     'class': TransformerIrisWorldModel, 'config': 'transformer_iris_mp_window_40_config.json',
    #     'model_dir': 'transformer_iris_mp_window_40', 'epoch': '3000', 'model_label': 'Transformer Iris MP Window 40'},
    'transformer_iris_mp_window_50': {
        'class': TransformerIrisWorldModel, 'config': 'transformer_iris_mp_window_50_config.json',
        'model_dir': 'transformer_iris_mp_window_50', 'epoch': '3000', 'model_label': 'Transformer Iris MP Window 50'},
    # 'transformer_iris_mp_window_60': {
    #     'class': TransformerIrisWorldModel, 'config': 'transformer_iris_mp_window_60_config.json',
    #     'model_dir': 'transformer_iris_mp_window_60', 'epoch': '3000', 'model_label': 'Transformer Iris MP Window 60'},
    'transformer_iris_mp_window_70': {
        'class': TransformerIrisWorldModel, 'config': 'transformer_iris_mp_window_70_config.json',
        'model_dir': 'transformer_iris_mp_window_70', 'epoch': '3000', 'model_label': 'Transformer Iris MP Window 70'},
    'transformer_iris_mp_square_40': {
        'class': TransformerIrisWorldModel, 'config': 'transformer_iris_mp_square_40_config.json',
        'model_dir': 'transformer_iris_mp_square_40', 'epoch': '3000', 'model_label': 'Transformer Iris MP Square 40'},
    'transformer_iris_mp_square_50': {
        'class': TransformerIrisWorldModel, 'config': 'transformer_iris_mp_square_50_config.json',
        'model_dir': 'transformer_iris_mp_square_50', 'epoch': '3000', 'model_label': 'Transformer Iris MP Square 50'},
    'transformer_iris_mp_square_60': {
        'class': TransformerIrisWorldModel, 'config': 'transformer_iris_mp_square_60_config.json',
        'model_dir': 'transformer_iris_mp_square_60', 'epoch': '3000', 'model_label': 'Transformer Iris MP Square 60'},
    'transformer_iris_mp_square_70': {
        'class': TransformerIrisWorldModel, 'config': 'transformer_iris_mp_square_70_config.json',
        'model_dir': 'transformer_iris_mp_square_70', 'epoch': '3000', 'model_label': 'Transformer Iris MP Square 70'},
}


DEFAULT_VALUES = {
    # general pipeline parameters
    'analysis_dir': './',
    #'data_path': '/data2/ziyxiang/social_world_model/data/train_test_splits_3D_dataset.pkl',
    #'checkpoint_dir': '/data2/ziyxiang/social_world_model/checkpoint',
    # as of 5/26, the new models in nfs checkpoint path use the new data path
    'data_path': '/ccn2/u/ziyxiang/swm_data_and_results/data/dataset_5_25_23.pkl',
    'checkpoint_dir': '/ccn2/u/ziyxiang/swm_data_and_results/checkpoint',
    'model_config_dir': './model_configs',
    # general training parameters for all models
    'batch_size': 2048,
    'lr': 1e-3,
    'epochs': int(3e4),
    'save_every': 500,
    # eval parameters
    'model_keys': list(MODEL_DICT_VAL.keys()),
    'eval_types': ['goal_events', 'multigoal_events', 'move_events'],
    'move_threshold': 4.0,
    'non_goal_burn_in': 50,
}