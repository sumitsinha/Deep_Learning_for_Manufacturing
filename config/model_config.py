"""The Model Configuration file contains configuration for training the model, conducting data study, KMC Generation, Benchmarking and Transfer Learning
        
        Model Training Parameters
                
        :param model_parameters['model_type']: The type of model to be used for training, currently defaults to 3D CNN  
        :type model_parameters['model_type']: str (required)

        :param model_parameters['output_type']: (regression, classification) The output type of the model used to initialize the output layer, currently defaults to regression  
        :type model_parameters['output_type']: str (required)

        :param model_parameters['batch_size']: The batch size while training, can be tuned based on the hardware specifications, currently defaults to 32  
        :type model_parameters['batch_size']: int (required)

        :param model_parameters['epocs']: The number of epocs the model is to be trained for, currently defaults to 150  
        :type model_parameters['epocs']: int (required)

        :param model_parameters['split_ratio']: Split Ratio for train and validation  
        :type model_parameters['split_ratio']: float (required)

        :param model_parameters['optimizer']: The optimizer to be used for model training, refer https://keras.io/optimizers/ for more information, currently defaults to adam  
        :type model_parameters['optimizer']: keras.optimizer (required)

        :param model_parameters['loss_func']: The loss function to be optimized while model training, refer https://keras.io/losses/ for more information, currently defaults to Mean Squared Error (MSE)
        :type model_parameters['loss_func']: keras.losses (required)

        :param model_parameters['regularizer_coeff']: The regularizing coefficient to be used for L2 norm regularization of the fully connected layer, refer https://keras.io/regularizers/ for more information currently defaults to 0.1 
        :type model_parameters['regularizer_coeff']: float (required)

        :param model_parameters['activate_tensorboard']: Tensorboard activation flag https://www.tensorflow.org/tensorboard, currently set to 0, changes to 1 for activating tensorbiard, Warning: There can be some compatibility issues with different Tensorflow and Cuda Toolkit Versions
        :type model_parameters['loss_func']: int (required)

        Data Study Parameters

        :param data_study_params['batch_size']: The batch size while conducting data study, can be tuned based on the hardware specifications, currently defaults to 32  
        :type data_study_params['batch_size']: int (required)

        :param data_study_params['epocs']: The number of epocs the model is to be trained for, currently defaults to 50  
        :type data_study_params['epocs']: int (required)

        :param data_study_params['split_ratio']: Split Ratio for train and validation during data study
        :type data_study_params['split_ratio']: float (required)

        :param data_study_params['min_train_samples']: Minimum train Samples for data study, currently defaults to 100
        :type data_study_params['min_train_samples']: int (required)

        :param data_study_params['max_train_samples']: Maximum train samples for data study, dataset size is the maximum value
        :type data_study_params['max_train_samples']: int (required)

        :param data_study_params['train_increment']: Increment in the train size with each iteration, currently defaults to 100
        :type data_study_params['train_increment']: int (required)
        
        Key Measurment Characteristics Generation Parameters

        :param kmc_params['tree_based_model']: The model to be used while generating feature importance, refer: https://xgboost.readthedocs.io/en/latest/R-package/discoverYourData.html#measure-feature-importance for more details, currently defaults to xgb, random forests can also be used
        :type kmc_params['tree_based_model']: str (required)
        
        :param kmc_params['tree_based_model']: The importance criteria to be used, currently defaults to gini index
        :type kmc_params['tree_based_model']: str (required)

        :param kmc_params['split_ratio']: Split Ratio for train and validation during data study
        :type kmc_params['split_ratio']: float (required)

        :param kmc_params['save_model']: Flag to save the model, Currently defaults to 0, change to 1 if model needs to be saved
        :type kmc_params['save_model']: int (required)

        :param kmc_params['plot_kmc']: Flag to plot the KMC, Currently defaults to 1, change to 0 if no plotting is required
        :type kmc_params['plot_kmc']: int (required)

        Benchmarking Parameters

        :param bm_params['max_models']: The maximum number of models to be used for benchmarking, currently defaults to 10
        :type bm_params['max_models']: int (required)

        :param bm_params['runs']: Number of benchmarking runs to be conducted
        :type bm_params['runs']: int (required)

        Transfer Learning Parameters

        :param transfer_learning['tl_type']: The type of transfer learning to be used (full_fine_tune,  variable_lr, feature_extractor) currently defaults to full_fine_tune
        :type transfer_learning['tl_type']: str (required)

        :param transfer_learning['tl_base']: The type of base model (3D CNN Architecture) to be used (pointdevnet, voxnet, 3d-UNet), currently defaults to PointdevNet
        :type transfer_learning['tl_base']: str (required)

        :param transfer_learning['tl_app']: The application of the transfer learning model (classification, regression), currently defaults to regression
        :type transfer_learning['tl_app']: str (required)

        :param transfer_learning['conv_layer_m']: The learning rate multiplier for convolution layers (only needed when tl_type is variable_lr), defaults to 0.1 (10% of the network Learning Rate)
        :type transfer_learning['conv_layer_m']: float (required)

        :param transfer_learning['dense_layer_m']: The learning rate multiplier for dense layers (only needed when tl_type is variable_lr), defaults to 1 (100% of the network Learning Rate)
        :type transfer_learning['dense_layer_m']: float (required)

        
"""


model_parameters = {	
        'model_type':'Bayesian 3D Convolution Neural Network', #other option: 'Bayesian 3D Convolution Neural Network'  
        'learning_type':'Basic', # use 'Transfer Learning' if transfer learning is to be leveraged
        'output_type':'regression',
        'batch_size': 32,
        'epocs':500,
        'split_ratio': 0.2,
        'optimizer':'adam',
        'loss_func':'mse',
        'regularizer_coeff': 0.01,
        'activate_tensorboard':0
        }
cae_sim_params = {
        'simulation_platform':'MatLab',
        'simulation_engine':'VRM',
        'max_run_length':15,
        'cae_input_path': 'check',
        'cae_output_path':'check',
        #case_study parameter imported from assembly_configration
}

encode_decode_params ={
        'model_depth':4,
        'inital_filter_dim':16,
        'kcc_sublist':0,#[0,1,2,3,4,5,6,7,8,9,10,11] use a list in case only a selected sublist of KCCs have to be used: 0 means all KCCs
        'output_heads':2
}
data_study_params = {
	'batch_size':32,
	'epocs':500,
	'min_train_samples':400,
        'train_increment':200,
        'max_train_samples':5000,
	'split_ratio':0.2,
        'tl_flag':0
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
        'runs':15,
        'split_ratio': 0.2
}

transfer_learning={
        'tl_type':'full_fine_tune', #options 'full_fine_tune', variable_lr', 'feature_extractor'
        'tl_base':'model_pointnet_64_halo.h5',
        'tl_app':'halo_deploy',
        'conv_layer_m':0.1,
        'dense_layer_m':1, 
}