
model_parameters = {	
        'model_type':'3D Convolutional Neural Network',
        'output_type':'regression',
        'batch_size': 10,
        'epocs': 20,
        'split_ratio': 0.3,
        'optimizer':'adam',
        'loss_func':'mse',
        'regularizer_coeff': 0.01,
        'activate_tensorboard':0
        }

data_study_params = {
	'batch_size':32,
	'epocs':50,
	'min_train_samples':100,
        'train_increment':100,
        'max_train_samples':3000,
	'split_ratio':0.2
}

kmc_params={
        'tree_based_model':'xgb',
        'importance_creteria':'gini',
        'split_ratio':0.2,
        'save_model':0,
        'plot_kmc':1,
}

bm_params={
        'max_models':10,
        'runs':3
}

transfer_learning={
        'tl_type':'full_fine_tune',
        'tl_base':'model_pointnet_64.h5',
        'tl_app':'regression' 
}