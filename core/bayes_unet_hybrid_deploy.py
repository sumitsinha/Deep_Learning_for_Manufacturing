""" The model train file trains the model on the download dataset and other parameters specified in the assemblyconfig file
The main function runs the training and populates the created file structure with the trained model, logs and plots
"""

import os
import sys
current_path=os.path.dirname(__file__)
parentdir = os.path.dirname(current_path)

os.environ["CUDA_VISIBLE_DEVICES"]="0" # Nvidia Quadro GV100
#os.environ["CUDA_VISIBLE_DEVICES"]="1" # Nvidia Quadro M2000

#Adding Path to various Modules
sys.path.append("../core")
sys.path.append("../visualization")
sys.path.append("../utilities")
sys.path.append("../datasets")
sys.path.append("../trained_models")
sys.path.append("../config")
#path_var=os.path.join(os.path.dirname(__file__),"../utilities")
#sys.path.append(path_var)
#sys.path.insert(0,parentdir) 

#Importing Required Modules
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
K.clear_session()

#Importing Config files
import assembly_config as config
import model_config as cftrain
import hybrid_utils as hy_util

#Importing required modules from the package
from measurement_system import HexagonWlsScanner
from assembly_system import VRMSimulationModel
from wls400a_system import GetInferenceData
from data_import import GetTrainData
from core_model_bayes import Bayes_DLModel
from training_viz import TrainViz
from metrics_eval import MetricsEval
from keras_lr_multiplier import LRMultiplier

class Unet_DeployModel:
	"""Train Model Class, the initialization parameters are parsed from modelconfig_train.py file
		
		:param batch_size: mini batch size while training the model 
		:type batch_size: int (required)

		:param epochs: no of epochs to conduct training
		:type epochs: int (required)

		:param split_ratio: train and validation split for the model
		:type assembly_system: float (required)

		The class contains run_train_model method
	"""	
	def get_model(self,model,model_path):
		"""get_model method is is used to retrieve the trained model from a given path
				
				:param model_path: Path to the trained model, ideally it should be same as the train model path output
				:type model_path: str (required)
		"""
		tfd = tfp.distributions

		model.load_weights(model_path)
		print('U-Net Deep Learning Model found and loaded')

		#print(error)
		#print('Model not found at this path ',model_path, ' Update path in config file if required')

		return model

	def bayes_unet_run_model(self,inference_data,inference_model,y_test_list,plots_path,epistemic_samples=20,run_id=0):
		"""run_train_model function trains the model on the dataset and saves the trained model,logs and plots within the file structure, the function prints the training evaluation metrics
			
			:param model: 3D CNN model compiled within the Deep Learning Class, refer https://keras.io/models/model/ for more information 
			:type model: keras.models (required)

			:param X_in: Train dataset input (predictor variables), 3D Voxel representation of the cloud of point and node deviation data obtained from the VRM software based on the sampling input
			:type X_in: numpy.array [samples*voxel_dim*voxel_dim*voxel_dim*deviation_channels] (required)
			
			:param Y_out: Train dataset output (variables to predict), Process Parameters/KCCs obtained from sampling
			:type Y_out: numpy.array [samples*assembly_kccs] (required)

			:param model_path: model path at which the trained model is saved
			:type model_path: str (required)
			
			:param logs_path: logs path where the training metrics file is saved
			:type logs_path: str (required)

			:param plots_path: plots path where model training loss convergence plot is saved
			:type plots_path: str (required)

			:param activate_tensorboard: flag to indicate if tensorboard should be added in model callbacks for better visualization, 0 by default, set to 1 to activate tensorboard
			:type activate_tensorboard: int

			:param run_id: Run id index used in data study to conduct multiple training runs with different dataset sizes, defaults to 0
			:type run_id: int			
		"""			
		import tensorflow as tf
		from tensorflow.keras.models import load_model
		import tensorflow.keras.backend as K 
		
		from scipy.stats import iqr

		y_preds_reg=np.zeros_like(y_test_list[0])
		y_preds_cla=np.zeros_like(y_test_list[1])
		y_preds_shape_error=np.zeros_like(y_test_list[2])
		
		y_reg_std=np.zeros_like(y_preds_reg)
		y_cla_std=np.zeros_like(y_preds_cla)
		y_shape_error_std=np.zeros_like(y_preds_shape_error)

		y_reg_iqr=np.zeros_like(y_preds_reg)
		y_cla_iqr=np.zeros_like(y_preds_cla)
		y_shape_error_iqr=np.zeros_like(y_preds_shape_error)
		
		#Aleatoric Uncertainty
		y_reg_aleatoric_std=np.zeros_like(y_preds_reg)
		
		y_actual_reg=y_test_list[0]
		y_actual_cla=y_test_list[1]
		y_actual_shape_error=y_test_list[2]

		plots_path_run_id=plots_path+'/plots_run_id_'+str(run_id)
		pathlib.Path(plots_path_run_id).mkdir(parents=True, exist_ok=True)

		for i in range(len(inference_data)):
			
			from scipy.stats import norm
			inference_sample=inference_data[i,:,:,:,:]
			input_sample=np.array([inference_sample]*epistemic_samples)
			print(input_sample.shape)
			model_outputs=inference_model(input_sample)
			
			output_reg=model_outputs[0]
			output_cla=model_outputs[1]
			output_shape_error=model_outputs[2]

			output_mean=output_reg.mean()
			aleatoric_std=output_reg.stddev()

			pred_mean=np.array(output_mean).mean(axis=0)
			aleatoric_mean=np.array(aleatoric_std).mean(axis=0)
			#Sample standard deviation
			pred_std=np.array(output_mean).std(axis=0,ddof=1)
			reg_iqr=iqr(output_mean,axis=0)

			output_mean=np.array(output_mean)
			
			pred_mean_cla=np.array(output_cla).mean(axis=0)
			pred_std_cla=np.array(output_cla).std(axis=0,ddof=1)
			cla_iqr=iqr(output_cla,axis=0)
			
			#Shape Error Metrics
			pred_mean_shape_error=np.array(output_shape_error).mean(axis=0)
			pred_std_shape_error=np.array(output_shape_error).std(axis=0,ddof=1)
			shape_error_iqr=iqr(output_shape_error,axis=0)
			
			print("Estimated Mean: ",pred_mean,pred_mean_cla)
			print("Estimated STD: ",pred_std,pred_std_cla)
			print("Estimated IQR: ",reg_iqr,cla_iqr)
			print("Estimated Aleatoric Mean: ",aleatoric_mean)
			
			print(output_mean.shape,aleatoric_std.shape,output_cla.shape)
			
			pred_plots=1
			
			if(pred_plots==1):
				for j in range(output_mean.shape[1]):
					plot_data=output_mean[:,j]
					actual_obv=y_actual_reg[i,j]
					plt.hist(plot_data, range=(actual_obv-0.5,actual_obv+0.5),bins=40)
					plt.axvline(x=actual_obv,label="Actual Value = "+str(actual_obv),c='r')
					plt.axvline(x=pred_mean[j],label="Prediction Mean = "+str(pred_mean[j]),c='c')
					plt.axvline(x=pred_mean[j]+norm.ppf(0.95)*pred_std[j], label="95 CI = "+str(pred_mean[j]+norm.ppf(0.95)*pred_std[j]),c='b')
					plt.axvline(x=pred_mean[j]-norm.ppf(0.95)*pred_std[j], label="95 CI = "+str(pred_mean[j]-norm.ppf(0.95)*pred_std[j]),c='b')
					plt.title("Prediction Distribution for KCC " + str(j) + " sample "+ str(i))
					plt.savefig(plots_path_run_id+"/"+ "reg_sample_"+ str(i)+"_KCC_" + str(j) +'.png')
					plt.clf()
				
				for j in range(output_cla.shape[1]):
					plot_data=output_cla[:,j]
					actual_obv=y_actual_cla[i,j]
					plt.hist(plot_data, range=(0,1),bins=40)
					plt.axvline(x=actual_obv,label="Actual Value = "+str(actual_obv),c='r')
					plt.axvline(x=pred_mean_cla[j],label="Prediction Mean = "+str(pred_mean_cla[j]),c='c')
					plt.axvline(x=pred_mean_cla[j]+norm.ppf(0.95)*pred_std_cla[j], label="95 CI = "+str(pred_mean_cla[j]+norm.ppf(0.95)*pred_std_cla[j]),c='b')
					plt.axvline(x=pred_mean_cla[j]-norm.ppf(0.95)*pred_std_cla[j], label="95 CI = "+str(pred_mean_cla[j]-norm.ppf(0.95)*pred_std_cla[j]),c='b')
					plt.title("Prediction Distribution for KCC " + str(j) + " sample "+ str(i))
					plt.savefig(plots_path_run_id+"/"+ "cla_sample_"+ str(i)+"_KCC_" + str(j) +'.png')
					plt.clf()
			
			y_preds_reg[i,:]=pred_mean
			y_reg_std[i,:]=pred_std
			y_reg_aleatoric_std[i,:]=aleatoric_mean
			
			y_preds_cla[i,]=pred_mean_cla
			y_cla_std[i,]=pred_std_cla

			y_reg_iqr[i,:]=reg_iqr
			y_cla_iqr[i,:]=cla_iqr

			#Shape Error Tensors
			y_preds_shape_error[i,:,:,:,:]=pred_mean_shape_error
			y_shape_error_std[i,:,:,:,:]=pred_std_shape_error
			y_shape_error_iqr[i,:,:,:,:]=shape_error_iqr
			

		pred_vector=[]
		pred_vector.append(y_preds_reg)
		pred_vector.append(y_preds_cla)
		pred_vector.append(y_preds_shape_error)

		epistemic_vector=[]
		epistemic_vector.append(y_reg_std)
		epistemic_vector.append(y_cla_std)
		epistemic_vector.append(y_shape_error_std)

		epistemic_vector_iqr=[]
		epistemic_vector_iqr.append(y_reg_iqr)
		epistemic_vector_iqr.append(y_cla_iqr)
		epistemic_vector_iqr.append(y_shape_error_iqr)
		
		aleatoric_vector=[]
		aleatoric_vector.append(y_reg_aleatoric_std)

		return pred_vector,epistemic_vector,epistemic_vector_iqr,aleatoric_vector

		
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

	system_noise=config.assembly_system['system_noise']
	aritifical_noise=config.assembly_system['aritifical_noise']
	data_folder=config.assembly_system['data_folder']
	kcc_folder=config.assembly_system['kcc_folder']
	kcc_files=config.assembly_system['kcc_files']
	test_kcc_files=config.assembly_system['test_kcc_files']

	#added for hybrid model
	categorical_kccs=config.assembly_system['categorical_kccs']
	
	print('Parsing from Training Config File')

	model_type=cftrain.model_parameters['model_type']
	output_type=cftrain.model_parameters['output_type']
	batch_size=cftrain.model_parameters['batch_size']
	epocs=cftrain.model_parameters['epocs']
	split_ratio=cftrain.model_parameters['split_ratio']
	optimizer=cftrain.model_parameters['optimizer']
	loss_func=cftrain.model_parameters['loss_func']
	regularizer_coeff=cftrain.model_parameters['regularizer_coeff']
	activate_tensorboard=cftrain.model_parameters['activate_tensorboard']
	
	print('Creating file Structure....')
	
	folder_name=part_type
	train_path='../trained_models/'+part_type
	pathlib.Path(train_path).mkdir(parents=True, exist_ok=True) 

	train_path=train_path+'/unet_model'
	pathlib.Path(train_path).mkdir(parents=True, exist_ok=True) 

	model_path=train_path+'/model'
	pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
	
	logs_path=train_path+'/logs'
	pathlib.Path(logs_path).mkdir(parents=True, exist_ok=True)

	plots_path=train_path+'/plots'
	pathlib.Path(plots_path).mkdir(parents=True, exist_ok=True)

	deployment_path=train_path+'/deploy'
	pathlib.Path(deployment_path).mkdir(parents=True, exist_ok=True)

	#Objects of Measurement System, Assembly System, Get Inference Data
	print('Initializing the Assembly System and Measurement System....')

	measurement_system=HexagonWlsScanner(data_type,application,system_noise,part_type,data_format)
	vrm_system=VRMSimulationModel(assembly_type,assembly_kccs,assembly_kpis,part_name,part_type,voxel_dim,voxel_channels,point_dim,aritifical_noise)
	get_data=GetTrainData()

	#print(input_conv_data.shape,kcc_subset_dump.shape)
	print('Building Unet Model')

	kcc_sublist=cftrain.encode_decode_params['kcc_sublist']
	output_heads=cftrain.encode_decode_params['output_heads']
	encode_decode_multi_output_construct=config.encode_decode_multi_output_construct
	
	if(output_heads==len(encode_decode_multi_output_construct)):
		print("Valid Output Stages and heads")
	else:
		print("Inconsistent model setting")

	print("KCC sub-listing: ",kcc_sublist)
	
	#Check for KCC sub-listing
	if(kcc_sublist!=0):
		output_dimension=len(kcc_sublist)
	else:
		output_dimension=assembly_kccs
	
	print("Process Parameter Dimension: ",output_dimension)

	input_size=(voxel_dim,voxel_dim,voxel_dim,voxel_channels)

	model_depth=cftrain.encode_decode_params['model_depth']
	inital_filter_dim=cftrain.encode_decode_params['inital_filter_dim']

	dl_model=Bayes_DLModel(model_type,output_dimension,optimizer,loss_func,regularizer_coeff,output_type)
	
	#changed to attention model
	model=dl_model.bayes_unet_model_3d_hybrid(inital_filter_dim,model_depth,categorical_kccs,voxel_dim,voxel_channels,output_heads)

	model_path=train_path+'/model'+'/unet_oser_0'
	#print(model.summary())
	#sys.exit()

	test_input_file_names_x=config.encode_decode_construct['input_test_data_files_x']
	test_input_file_names_y=config.encode_decode_construct['input_test_data_files_y']
	test_input_file_names_z=config.encode_decode_construct['input_test_data_files_z']

	print('Importing and Preprocessing Cloud-of-Point Data')
	
	point_index=get_data.load_mapping_index(mapping_index)
	
	test_input_dataset=[]
	test_input_dataset.append(get_data.data_import(test_input_file_names_x,data_folder))
	test_input_dataset.append(get_data.data_import(test_input_file_names_y,data_folder))
	test_input_dataset.append(get_data.data_import(test_input_file_names_z,data_folder))

	test_kcc_dataset=get_data.data_import(test_kcc_files,kcc_folder)
	
	if(kcc_sublist!=0):
		print("Sub-setting Process Parameters: ",kcc_sublist)
		test_kcc_dataset=test_kcc_dataset[:,kcc_sublist]
	else:
		print("Using all Process Parameters")
	
	#Pre-processing to point cloud data
	test_input_conv_data, test_kcc_subset_dump,test_kpi_subset_dump=get_data.data_convert_voxel_mc(vrm_system,test_input_dataset,point_index,test_kcc_dataset)

	kcc_regression_test,kcc_classification_test=hy_util.split_kcc(test_kcc_subset_dump)

	Y_out_test_list=[]
	Y_out_test_list.append(kcc_regression_test)
	Y_out_test_list.append(kcc_classification_test)
	
	y_shape_error_test_list=[]

	for encode_decode_construct in encode_decode_multi_output_construct:
		#importing file names for model output
		print("Importing output data for stage: ",encode_decode_construct)
		

		test_output_file_names_x=encode_decode_construct['output_test_data_files_x']
		test_output_file_names_y=encode_decode_construct['output_test_data_files_y']
		test_output_file_names_z=encode_decode_construct['output_test_data_files_z']
	
		test_output_dataset=[]
		test_output_dataset.append(get_data.data_import(test_output_file_names_x,data_folder))
		test_output_dataset.append(get_data.data_import(test_output_file_names_y,data_folder))
		test_output_dataset.append(get_data.data_import(test_output_file_names_z,data_folder))
		
		test_output_conv_data, test_kcc_subset_dump,test_kpi_subset_dump=get_data.data_convert_voxel_mc(vrm_system,test_output_dataset,point_index,test_kcc_dataset)
		
		y_shape_error_test_list.append(test_output_conv_data)

	shape_error_test=np.concatenate(y_shape_error_test_list, axis=4)

	Y_out_test_list.append(shape_error_test)

	unet_deploy_model=Unet_DeployModel()
	
	pred_vector,epistemic_vector,epistemic_vector_iqr,aleatoric_vector=unet_deploy_model.bayes_unet_run_model(test_input_conv_data,model,Y_out_test_list,plots_path)
	
	print("Computing Metrics..")
	
	metrics_eval=MetricsEval();
	
	eval_metrics_reg,accuracy_metrics_df_reg=metrics_eval.metrics_eval_base(pred_vector[0],Y_out_test_list[0],logs_path)
	eval_metrics_cla,accuracy_metrics_df_cla=metrics_eval.metrics_eval_classification(pred_vector[1],Y_out_test_list[1],logs_path)
		
			
	#y_cop_pred_flat=y_cop_pred.flatten()
	#y_cop_test_flat=y_cop_test.flatten()

	#combined_array=np.stack([y_cop_test_flat,y_cop_pred_flat],axis=1)
	#filtered_array=combined_array[np.where(combined_array[:,0] >= 0.05)]
	#y_cop_test_vector=filtered_array[:,0:1]
	#y_cop_pred_vector=filtered_array[:,1:2]

	eval_metrics_cop_list=[]
	accuracy_metrics_df_cop_list=[]
	
	t=0		
	
	index=0

	for i in range(output_heads):
		y_cop_pred=model_outputs[2][:,:,:,:,t:t+3]
		y_cop_test=Y_out_test_list[2]
		y_cop_pred_vector=np.reshape(y_cop_pred,(y_cop_pred.shape[0],-1))
		y_cop_test_vector=np.reshape(y_cop_test,(y_cop_test.shape[0],-1))
		y_cop_pred_vector=y_cop_pred_vector.T
		y_cop_test_vector=y_cop_test_vector.T
		
		print(y_cop_pred_vector.shape)
		#y_cop_test_flat=y_cop_test.flatten()
				
		eval_metrics_cop,accuracy_metrics_df_cop=metrics_eval.metrics_eval_cop(y_cop_pred_vector,y_cop_test_vector,logs_path)
		eval_metrics_cop_list.append(eval_metrics_cop)
		accuracy_metrics_df_cop_list.append(accuracy_metrics_df_cop)

		accuracy_metrics_df_cop.to_csv(logs_path+'/metrics_test_cop_'+str(index)+'.csv')
		
		print("The Model Segmentation Validation Metrics are ")
		print(accuracy_metrics_df_cop.mean())
		
		accuracy_metrics_df_cop.mean().to_csv(logs_path+'/metrics_test_cop_summary_'+str(index)+'.csv')
		
		t=t+3
		index=index+1

	#Saving Log files	
	accuracy_metrics_df_reg.to_csv(logs_path+'/metrics_test_regression.csv')
	accuracy_metrics_df_cla.to_csv(logs_path+'/metrics_test_classification.csv')
	
	print("The Model Validation Metrics for Regression based KCCs")	
	print(accuracy_metrics_df_reg)
	
	accuracy_metrics_df_reg.mean().to_csv(logs_path+'/metrics_train_regression_summary.csv')
	print("The Model Validation Metrics Regression Summary")
	print(accuracy_metrics_df_reg.mean())

	print("The Model Validation Metrics for Classification based KCCs")	
	print(accuracy_metrics_df_cla)
	
	accuracy_metrics_df_cla.mean().to_csv(logs_path+'/metrics_train_classification_summary.csv')
	print("The Model Validation Metrics Classification Summary")
	print(accuracy_metrics_df_cla.mean())
	