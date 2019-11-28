
model_parameters = {	
        'model_type':'3D Convolutional Neural Network',
        'output_type':'regression',
        'batch_size': 32,
        'epocs': 150,
        'split_ratio': 0.3,
        'optimizer':'adam',
        'loss_func':'mse',
        'regularizer_coeff': 0.01,
        'activate_tensorboard':0
        }

data_study_params = {
	'batch_size':32,
	'epocs':5,
	'no_of_splits':10,
	'min_train_samples':50,
	'split_ratio':0.2
}