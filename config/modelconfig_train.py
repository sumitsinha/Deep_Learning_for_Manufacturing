
model_parameters = {	
        'model_type':'cnn_3d_vanilla',
        'output_type':'regression',
        'batch_size': 32,
        'epocs': 150,
        'split_ratio': 0.3,
        'optimizer':'adam',
        'regularizer_coeff': 0.1
        }

data_study_params = {
	'batch_size':32,
	'epocs':50,
	'no_of_splits':14,
	'min_train_samples':5000,
	'split_ratio':0.2
}