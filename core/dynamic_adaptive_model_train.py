""" The model train file trains the model on the download dataset and other parameters specified in the assemblyconfig file
The main function runs the training and populates the created file structure with the trained model, logs and plots
"""

import os
import sys
current_path=os.path.dirname(__file__)
parentdir = os.path.dirname(current_path)

#choose GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0" # Nvidia Quadro GV100
#os.environ["CUDA_VISIBLE_DEVICES"]="1" # Nvidia Quadro M2000

#Adding Path to various Modules
sys.path.append("../core")
sys.path.append("../visualization")
sys.path.append("../utilities")
sys.path.append("../datasets")
sys.path.append("../trained_models")
sys.path.append("../config")
sys.path.append("../cae_simulations")
sys.path.append("../active_learning")
sys.path.append("../transfer_learning")
#path_var=os.path.join(os.path.dirname(__file__),"../utilities")
#sys.path.append(path_var)
#sys.path.insert(0,parentdir) 

#Importing Required Modules
import pathlib
import numpy as np
import pandas as pd
import plotly as py
import plotly.graph_objects as go
import cufflinks as cf
from tqdm import tqdm

import tensorflow as tf
import tensorflow_probability as tfp
A = tf.constant([[3, 2], [5, 2]])

print('Dummy TensorFlow initialization to load Cudnn Library: ', tf.eye(2,2))

import matlab.engine
from pyDOE import lhs
from scipy.stats import uniform,norm

#Importing Config files
import assembly_config as config
import model_config as cftrain

#Importing required modules from the package
from measurement_system import HexagonWlsScanner
from assembly_system import VRMSimulationModel
from wls400a_system import GetInferenceData
from data_import import GetTrainData

from sampling_system import AdaptiveSampling
import kcc_config as kcc_config
import sampling_config as sampling_config

from metrics_eval import MetricsEval
from uncertainity_sampling import UncertainitySampling
#from tl_core import TransferLearning




if __name__ == '__main__':

	print('Parsing from Assembly Config File....')

	data_type=config.assembly_system['data_type']
	application=config.assembly_system['application']
	part_type=config.assembly_system['part_type']
	part_name=config.assembly_system['part_name']
	data_format=config.assembly_system['data_format']
	assembly_type=config.assembly_system['assembly_type']
	assembly_kccs=config.assembly_system['assembly_kccs']	
	assembly_kpis=config.assembly_system['assembly_kpis']
	voxel_dim=config.assembly_system['voxel_dim']
	point_dim=config.assembly_system['point_dim']
	voxel_channels=config.assembly_system['voxel_channels']
	noise_type=config.assembly_system['noise_type']
	mapping_index=config.assembly_system['mapping_index']
	aritifical_noise=config.assembly_system['aritifical_noise']
	data_folder=config.assembly_system['data_folder']
	kcc_folder=config.assembly_system['kcc_folder']

	print('Parsing from Training Config File')

	model_type=cftrain.model_parameters['model_type']
	learning_type=cftrain.model_parameters['learning_type'] 
	output_type=cftrain.model_parameters['output_type']
	optimizer=cftrain.model_parameters['optimizer']
	loss_func=cftrain.model_parameters['loss_func']
	regularizer_coeff=cftrain.model_parameters['regularizer_coeff']
	activate_tensorboard=cftrain.model_parameters['activate_tensorboard']

	batch_size=cftrain.data_study_params['batch_size']
	epocs=cftrain.data_study_params['epocs']
	split_ratio=cftrain.data_study_params['split_ratio']
	min_train_samples=cftrain.data_study_params['min_train_samples']
	max_train_samples=cftrain.data_study_params['max_train_samples']
	train_increment=cftrain.data_study_params['train_increment']
	tl_flag=cftrain.data_study_params['tl_flag']

	tl_type=cftrain.transfer_learning['tl_type']
	tl_base=cftrain.transfer_learning['tl_base']
	tl_app=cftrain.transfer_learning['tl_app']
	conv_layer_m=cftrain.transfer_learning['conv_layer_m']
	dense_layer_m=cftrain.transfer_learning['dense_layer_m']

	simulation_platform=cftrain.cae_sim_params['simulation_platform']
	simulation_engine=cftrain.cae_sim_params['simulation_engine']
	max_run_length=cftrain.cae_sim_params['max_run_length']
	case_study=part_type

	print('Creating file Structure....')
	folder_name=part_type
	train_path='../trained_models/'+part_type+'/adaptive'
	pathlib.Path(train_path).mkdir(parents=True, exist_ok=True) 

	model_path=train_path+'/model'
	pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
	
	logs_path=train_path+'/logs'
	pathlib.Path(logs_path).mkdir(parents=True, exist_ok=True)

	plots_path=train_path+'/plots'
	pathlib.Path(plots_path).mkdir(parents=True, exist_ok=True)

	deployment_path=train_path+'/deploy'
	pathlib.Path(deployment_path).mkdir(parents=True, exist_ok=True)

	print('Initializing....')
	measurement_system=HexagonWlsScanner(data_type,application,system_noise,part_type,data_format)
	print('Measurement system initialized')
	vrm_system=VRMSimulationModel(assembly_type,assembly_kccs,assembly_kpis,part_name,part_type,voxel_dim,voxel_channels,point_dim,aritifical_noise)

	print('Assembly and simulation system initialized')
	get_data=GetTrainData();

	metrics_eval=MetricsEval();
	
	point_index=get_data.load_mapping_index(mapping_index)

	print('Support systems initialized')
	
	kcc_struct=kcc_config.kcc_struct
	sampling_config=sampling_config.sampling_config
	adaptive_sampling=AdaptiveSampling(sampling_config['sample_dim'],sampling_config['sample_type'],sampling_config['adaptive_sample_dim'],sampling_config['adaptive_runs'])
	
	output_dimension=assembly_kccs
	eval_metrics_type= ["Mean Absolute Error","Mean Squared Error","Root Mean Squared Error","R Squared"]

	kcc_id=[]
	for kcc in kcc_struct:
		kcc_name=kcc['kcc_name']
		kcc_id.append(kcc_name)

	print('Running Dynamic Training...')

	unsap=UncertainitySampling(sampling_config['adaptive_sample_dim'],sampling_config['num_mix'])

	combined_conv_data_list=[]
	combined_kcc_data_list=[]

	eval_metrics_type= ["Mean Absolute Error","Mean Squared Error","Root Mean Squared Error","R Squared"]

	datastudy_output_test=np.zeros((max_run_length,(assembly_kccs+1)*len(eval_metrics_type)+1))

	test_flag=1
	sampling_validation_flag=1

	if(test_flag==1):
		print('Generating Testing Data...')
		print('LHS Sampling for test samples')
				
		#get prediction errors
		#get uncertainty estimates
		from cae_simulations import CAESimulations
		cae_simulations=CAESimulations(simulation_platform,simulation_engine,max_run_length,case_study)
		test_samples=adaptive_sampling.inital_sampling_uniform_random(kcc_struct,sampling_config['test_sample_dim'])

		file_name=sampling_config['output_file_name_test']+".csv"
		file_path=kcc_folder+'/'+file_name
		file_names_x=sampling_config['datagen_filename_x']+'test'+'_'+str(0)+'.csv'
		file_names_y=sampling_config['datagen_filename_y']+'test'+'_'+str(0)+'.csv'
		file_names_z=sampling_config['datagen_filename_z']+'test'+'_'+str(0)+'.csv'	
		
		np.savetxt(file_path, test_samples, delimiter=",")
		print('Sampling Completed...')

		cae_status=cae_simulations.run_simulations(run_id=0,type_flag='test')

		print("Pre-processing simulated test data")
		dataset_test=[]
		dataset_test.append(get_data.data_import([file_names_x],data_folder))
		dataset_test.append(get_data.data_import([file_names_y],data_folder))
		dataset_test.append(get_data.data_import([file_names_z],data_folder))
				
		input_conv_data_test, kcc_subset_dump_test,kpi_subset_dump_test=get_data.data_convert_voxel_mc(vrm_system,dataset_test,point_index,test_samples)

	if(sampling_validation_flag==1):
		print('Generating Adaptive Sampling Data...')
		print('LHS Sampling for validation samples')
				
		#get prediction errors
		#get uncertainty estimates
		from cae_simulations import CAESimulations
		cae_simulations=CAESimulations(simulation_platform,simulation_engine,max_run_length,case_study)
		validate_samples=adaptive_sampling.inital_sampling_uniform_random(kcc_struct,sampling_config['sample_validation_dim'])

		file_name=sampling_config['output_file_name_validate']+".csv"
		file_path=kcc_folder+'/'+file_name
		file_names_x=sampling_config['datagen_filename_x']+'validate'+'_'+str(0)+'.csv'
		file_names_y=sampling_config['datagen_filename_y']+'validate'+'_'+str(0)+'.csv'
		file_names_z=sampling_config['datagen_filename_z']+'validate'+'_'+str(0)+'.csv'
			
		np.savetxt(file_path, initial_samples, delimiter=",")
		print('Sampling Completed...')
		cae_status=cae_simulations.run_simulations(run_id=0,type_flag='validate')

		print("Pre-processing simulated test data")
		dataset_validate=[]
		dataset_validate.append(get_data.data_import([file_names_x],data_folder))
		dataset_validate.append(get_data.data_import([file_names_y],data_folder))
		dataset_validate.append(get_data.data_import([file_names_z],data_folder))
				
		input_conv_data_validate, kcc_subset_dump_validate,kpi_subset_dump_validate=get_data.data_convert_voxel_mc(vrm_system,dataset_validate,point_index,validate_samples)

	for i in tqdm(range(max_run_length)):
		
		run_id=i
		print('Training Run ID: ',i)
		
		file_name=sampling_config['output_file_name_train']+'_'+str(i)+'.csv'
		
		file_names_x=sampling_config['datagen_filename_x']+'train'+'_'+str(i)+'.csv'
		file_names_y=sampling_config['datagen_filename_y']+'train'+'_'+str(i)+'.csv'
		file_names_z=sampling_config['datagen_filename_z']+'train'+'_'+str(i)+'.csv'
		file_names_x=[file_names_x]
		file_names_y=[file_names_y]
		file_names_z=[file_names_z]
		
		if(i==0):
			print('Generating initial samples...')
		
			train_dim=sampling_config['sample_dim']
			initial_samples=adaptive_sampling.inital_sampling_uniform_random(kcc_struct,sampling_config['sample_dim'])
			file_path=kcc_folder+'/'+file_name
			np.savetxt(file_path, initial_samples, delimiter=",")
			train_samples=initial_samples
			print('Sampling Completed...')
			 
			cae_status=cae_simulations.run_simulations(i,'train')
			
			dataset=[]
			dataset.append(get_data.data_import(file_names_x,data_folder))
			dataset.append(get_data.data_import(file_names_y,data_folder))
			dataset.append(get_data.data_import(file_names_z,data_folder))
			
			input_conv_data, kcc_subset_dump,kpi_subset_dump=get_data.data_convert_voxel_mc(vrm_system,dataset,point_index,train_samples)


		if(i>0):
			
			print('Adaptive Sampling..')

			#Currently using random sampling
			file_name=sampling_config['output_file_name_train']+'_'+str(i)+'.csv'
			train_dim=train_dim+sampling_config['adaptive_sample_dim']
			#initial_samples=adaptive_sampling.inital_sampling_uniform_random(kcc_struct,sampling_config['adaptive_sample_dim'])
			adaptive_gen_samples,gmm_model_params=unsap.get_distribution_samples(kcc_subset_dump_validate,y_pred_validate,y_std_validate)
			
			np.savetxt(logs_path+'/gmm_model_params_run_'+str(run_id)+'.csv', gmm_model_params, delimiter=",")
			file_path=kcc_folder+'/'+file_name
			np.savetxt(file_path,adaptive_gen_samples, delimiter=",")
			print('Sampling Completed...')

			cae_status=cae_simulations.run_simulations(i,'train')
			
			dataset=[]
			dataset.append(get_data.data_import(file_names_x,data_folder))
			dataset.append(get_data.data_import(file_names_y,data_folder))
			dataset.append(get_data.data_import(file_names_z,data_folder))
			
			input_conv_data, kcc_subset_dump,kpi_subset_dump=get_data.data_convert_voxel_mc(vrm_system,dataset,point_index,adaptive_gen_samples)


		combined_conv_data_list.append(input_conv_data)
		combined_kcc_data_list.append(kcc_subset_dump)

		print('Concatenating dataset')
		print(len(combined_conv_data_list))

		combined_conv_data=np.concatenate(combined_conv_data_list,axis=0)
		combined_kcc_data=np.concatenate(combined_kcc_data_list,axis=0)
		print(combined_conv_data.shape,combined_kcc_data.shape)
		
		if(model_type=='Bayesian 3D Convolution Neural Network'):
			
			from core_model_bayes import Bayes_DLModel
			from model_train_bayes import BayesTrainModel

			dl_model=Bayes_DLModel(model_type,output_dimension,optimizer,loss_func,regularizer_coeff,output_type)
			model=dl_model.bayes_cnn_model_3d(voxel_dim,voxel_channels)

			print('Model summary used for training')
			print(model.summary())

			train_model=BayesTrainModel(batch_size,epocs,split_ratio)
			trained_model=train_model.run_train_model(model,combined_conv_data,combined_kcc_data,model_path,logs_path,plots_path,activate_tensorboard,run_id)
			print('Training Complete')


		if(model_type=='3D Convolution Neural Network'):
			from core_model import DLModel
			from model_deployment import DeployModel

			from model_train import TrainModel
			deploy_model=DeployModel();
			
			if(learning_type=='Basic'):
				dl_model=DLModel(model_type,output_dimension,optimizer,loss_func,regularizer_coeff,output_type)
				model=dl_model.cnn_model_3d(voxel_dim,voxel_channels)
			
			if(learning_type=='Transfer Learning'):
				transfer_learning=TransferLearning(tl_type,tl_base,tl_app,model_type,assembly_kccs,optimizer,loss_func,regularizer_coeff,output_type)
				base_model=transfer_learning.get_trained_model()
				
				print('Base Model used for Transfer Learning...')
				print(base_model.summary())
				
				#plot_model(base_model, to_file='model.png')

				transfer_model=transfer_learning.build_transfer_model(base_model)

				if(tl_type=='full_fine_tune'):
					model=transfer_learning.full_fine_tune(transfer_model)

				if(tl_type=='variable_lr'):
					model=transfer_learning.set_variable_learning_rates(transfer_model,conv_layer_m,dense_layer_m)

				if(tl_type=='feature_extractor'):
					model=transfer_learning.set_fixed_train_params(transfer_model)
		
			print('Model summary used for training')
			print(model.summary())
			
			train_model=TrainModel(batch_size,epocs,split_ratio)
			trained_model,eval_metrics,accuracy_metrics_df=train_model.run_train_model(model,combined_conv_data,combined_kcc_data,model_path,logs_path,plots_path,activate_tensorboard,run_id)

		print('Training complete for run: ',i)

		print('Sampling using trained model...')

		if(model_type=='Bayesian 3D Convolution Neural Network'):
			
			from bayes_model_deployment import BayesDeployModel
			from tensorflow.keras import backend as K
			
			deploy_model=BayesDeployModel()
			dl_model=Bayes_DLModel(model_type,output_dimension,optimizer,loss_func,regularizer_coeff,output_type)
			
			model=dl_model.bayes_cnn_model_3d(voxel_dim,voxel_channels)
			model_bayes_path=model_path+'/Bayes_trained_model_'+str(run_id)
			inference_model=deploy_model.get_model(model,model_bayes_path,voxel_dim,voxel_channels)

			y_pred=np.zeros_like(kcc_subset_dump_validate)
			
			plots_path_validate=plots_path+'/validation_sampling'
			pathlib.Path(plots_path_validate).mkdir(parents=True, exist_ok=True)

			y_pred_validate,y_std_validate=deploy_model.model_inference(input_conv_data_validate,inference_model,y_pred,kcc_subset_dump_validate,plots_path_validate,run_id)
			#eval_metrics_test,accuracy_metrics_df_test=metrics_eval.metrics_eval_base(y_pred,kcc_subset_dump_test,logs_path,run_id)

			std_file_path=logs_path+'/'+'uncertainty_validate_'+str(run_id)+'_.csv'
			np.savetxt(std_file_path, y_std_validate, delimiter=",")

			pred_file_path=logs_path+'/'+'predictions_validate_'+str(run_id)+'_.csv'
			np.savetxt(pred_file_path, y_pred_validate, delimiter=",")


		print('Inferencing from trained model...')
		
		if(model_type=='Bayesian 3D Convolution Neural Network'):
			
			from bayes_model_deployment import BayesDeployModel
			from tensorflow.keras import backend as K
			
			deploy_model=BayesDeployModel()
			dl_model=Bayes_DLModel(model_type,output_dimension,optimizer,loss_func,regularizer_coeff,output_type)
			
			model=dl_model.bayes_cnn_model_3d(voxel_dim,voxel_channels)
			model_bayes_path=model_path+'/Bayes_trained_model_'+str(run_id)
			inference_model=deploy_model.get_model(model,model_bayes_path,voxel_dim,voxel_channels)

			plots_path_test=plots_path+'/test'
			pathlib.Path(plots_path_test).mkdir(parents=True, exist_ok=True)
			y_pred=np.zeros_like(kcc_subset_dump_test)
			y_pred,y_std=deploy_model.model_inference(input_conv_data_test,inference_model,y_pred,kcc_subset_dump_test,plots_path_test,run_id)
			eval_metrics_test,accuracy_metrics_df_test=metrics_eval.metrics_eval_base(y_pred,kcc_subset_dump_test,logs_path,run_id)

			std_file_path=logs_path+'/'+'uncertainty_test_'+str(run_id)+'_.csv'
			np.savetxt(std_file_path, y_std, delimiter=",")

			pred_file_path=logs_path+'/'+'predictions_test_'+str(run_id)+'_.csv'
			np.savetxt(pred_file_path, y_pred, delimiter=",")

		if(model_type=='3D Convolution Neural Network'):

			from keras import backend as K
			model_test_path=train_path+'/model'+'/trained_model_'+str(run_id)+'.h5'
			
			inference_model=deploy_model.get_model(model_test_path)
			y_pred=deploy_model.model_inference(input_conv_data_test,inference_model);
			eval_metrics_test,accuracy_metrics_df_test=metrics_eval.metrics_eval_base(y_pred,kcc_subset_dump_test,logs_path,run_id)

		datastudy_output_test[i,0]=train_dim
		datastudy_output_test[i,1:assembly_kccs+1]=eval_metrics_test["Mean Absolute Error"]
		datastudy_output_test[i,assembly_kccs+1:(2*assembly_kccs)+1]=eval_metrics_test["Mean Squared Error"]
		datastudy_output_test[i,(2*assembly_kccs)+1:(3*assembly_kccs)+1]=eval_metrics_test["Root Mean Squared Error"]
		datastudy_output_test[i,(3*assembly_kccs)+1:(4*assembly_kccs)+1]=eval_metrics_test["R Squared"]

		file_name='test_metrics_dynamic_train_'+str(run_id)+'_.csv'
		accuracy_metrics_df_test.to_csv(logs_path+'/'+file_name)
		print("Model Testing Complete on samples :",train_dim)
		print("The Model Test Metrics are ")
		print(eval_metrics_test)
		K.clear_session()


	for i in range(len(eval_metrics_type)):
		datastudy_output_test[:,(4*assembly_kccs)+i+1]=np.mean(datastudy_output_test[:,(i*assembly_kccs)+1:((i+1)*assembly_kccs)+1],axis=1)
	
	col_names=['Training_Samples']
	for metric in eval_metrics_type:
		for kcc in kcc_id:
			col_names.append(str(metric)+'_'+str(kcc))

	for metric in eval_metrics_type:
	    col_names.append(str(metric)+"_avg")

	ds_output_df_test=pd.DataFrame(data=datastudy_output_test,columns=col_names)
	ds_output_df_test.to_csv(logs_path+'/'+'dynamic_training_output_test.csv')
	
	print('Dynamic Training complete')

	print('Plotting Data Study Validation Results: ')
	fig_test = ds_output_df_test.iplot(x='Training_Samples',asFigure=True)
	py.offline.plot(fig_test,filename=logs_path+'/'+"dynamic_training_plot_test.html")

