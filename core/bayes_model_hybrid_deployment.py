""" The model deploy file is used to leverage a trained model to perform inference on unknown set of node deviations.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ["CUDA_VISIBLE_DEVICES"]="0" # Nvidia Quadro GV100
#os.environ["CUDA_VISIBLE_DEVICES"]="1" # Nvidia Quadro M2000

import sys
current_path=os.path.dirname(__file__)
parentdir = os.path.dirname(current_path)

#Adding Path to various Modules
sys.path.append("../core")
sys.path.append("../visualization")
sys.path.append("../utilities")
sys.path.append("../datasets")
sys.path.append("../trained_models")
sys.path.append("../config")

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import csv
import logging
tf.get_logger().setLevel(logging.ERROR)


#Importing Config files
import assembly_config as config
import model_config as cftrain
import measurement_config as mscofig
import hybrid_utils as hy_util

#Importing required modules from the package
from measurement_system import HexagonWlsScanner
from assembly_system import VRMSimulationModel
from assembly_system import PartType
from wls400a_system import GetInferenceData
from metrics_eval import MetricsEval
from data_import import GetTrainData
from core_model_bayes import Bayes_DLModel
#from cam_viz import CamViz

class BayesDeployModel:
	"""The Deploy Model class is used to import a trained model and use it to infer on unknown data

	"""
	def get_model(self,model,model_path,voxel_dim,deviation_channels):
		"""get_model method is is used to retrieve the trained model from a given path
				
				:param model_path: Path to the trained model, ideally it should be same as the train model path output
				:type model_path: str (required)
		"""
		tfd = tfp.distributions

		model.load_weights(model_path)
		print('Deep Learning Model found and loaded')

		#print(error)
		#print('Model not found at this path ',model_path, ' Update path in config file if required')

		return model

	def model_inference(self,inference_data,inference_model,y_test_list,plots_path,epistemic_samples=20,run_id=0):
		"""model_inference method is used to infer from unknown sample(s) using the trained model 
				
				:param inference_data: Unknown dataset having same structure as the train dataset
				:type inference_data: numpy.array [samples*voxel_dim*voxel_dim*voxel_dim*deviation_channels] (required) (required)

				:param inference_model: Trained model
				:type inference_model: keras.model (required)
				
				:param print_result: Flag to indicate if the result needs to be printed, 0 by default, change to 1 in case the results need to be printed on the console
				:type print_result: int

		"""		
		#result=inference_model.(inference_data)
		from scipy.stats import iqr

		y_preds_reg=np.zeros_like(y_test_list[0])
		y_preds_cla=np.zeros_like(y_test_list[1])
		
		y_reg_std=np.zeros_like(y_preds_reg)
		y_cla_std=np.zeros_like(y_preds_cla)

		y_reg_iqr=np.zeros_like(y_preds_reg)
		y_cla_iqr=np.zeros_like(y_preds_cla)
		
		#Aleatoric Uncertainty
		y_reg_aleatoric_std=np.zeros_like(y_preds_reg)
		
		y_actual_reg=y_test_list[0]
		y_actual_cla=y_test_list[1]

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

		pred_vector=[]
		pred_vector.append(y_preds_reg)
		pred_vector.append(y_preds_cla)

		epistemic_vector=[]
		epistemic_vector.append(y_reg_std)
		epistemic_vector.append(y_cla_std)

		epistemic_vector_iqr=[]
		epistemic_vector_iqr.append(y_reg_iqr)
		epistemic_vector_iqr.append(y_cla_iqr)

		aleatoric_vector=[]
		aleatoric_vector.append(y_reg_aleatoric_std)

		return pred_vector,epistemic_vector,epistemic_vector_iqr,aleatoric_vector
	
	def model_mean_eval(self,inference_data,inference_model):

		def take_mean(f, *args, **kwargs):
		  """Tracer which sets each random variable's value to its mean."""
		  rv = f(*args, **kwargs)
		  rv._value = rv.distribution.mean()
		  return rv

		#import tensorflow_probability.edward2 as ed
		from edward2 import trace
		with trace(take_mean):
			model_outputs = inference_model(inference_data)

		#Edward Trace returns a Distribution object from the model: Output Lambda for regression part
		#Classification object is numpy array
		y_reg_output=model_outputs[0].mean()
		y_cla_output=model_outputs[1]
		
		return y_reg_output,y_cla_output

if __name__ == '__main__':
	
	print("Welcome to Deep Learning for Manufacturing (dlmfg)...")
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
	file_names_x=config.assembly_system['test_data_files_x']
	file_names_y=config.assembly_system['test_data_files_y']
	file_names_z=config.assembly_system['test_data_files_z']
	system_noise=config.assembly_system['system_noise']
	aritifical_noise=config.assembly_system['aritifical_noise']
	data_folder=config.assembly_system['data_folder']
	kcc_folder=config.assembly_system['kcc_folder']
	kcc_files=config.assembly_system['test_kcc_files']
	
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

	print('Initializing the Assembly System and Measurement System....')
	measurement_system=HexagonWlsScanner(data_type,application,system_noise,part_type,data_format)
	vrm_system=VRMSimulationModel(assembly_type,assembly_kccs,assembly_kpis,part_name,part_type,voxel_dim,voxel_channels,point_dim,aritifical_noise)
	deploy_model=BayesDeployModel()
	
	#Generate Paths
	train_path='../trained_models/'+part_type
	model_path=train_path+'/model'+'/Bayes_MH_0'
	logs_path=train_path+'/logs'
	deploy_path=train_path+'/deploy/'
	plots_path=train_path+'/plots/'

	#Voxel Mapping File

	get_data=GetTrainData();
	
	print('Importing and Preprocessing Cloud-of-Point Data')
	dataset=[]
	dataset.append(get_data.data_import(file_names_x,data_folder))
	dataset.append(get_data.data_import(file_names_y,data_folder))
	dataset.append(get_data.data_import(file_names_z,data_folder))
	point_index=get_data.load_mapping_index(mapping_index)

	#Make an Object of the Measurement System Class
	measurement_system=HexagonWlsScanner(data_type,application, system_noise,part_type,data_format)
	#Make an Object of the Assembly System Class
	assembly_system=PartType(assembly_type,assembly_kccs,assembly_kpis,part_name,part_type,voxel_dim,voxel_channels,point_dim)

	#Import model architecture
	output_dimension=assembly_kccs
	
	dl_model=Bayes_DLModel(model_type,output_dimension,optimizer,loss_func,regularizer_coeff,output_type)
	
	model=dl_model.bayes_cnn_model_3d_hybrid(categorical_kccs,voxel_dim,voxel_channels)

	#Inference from simulated data
	inference_model=deploy_model.get_model(model,model_path,voxel_dim,voxel_channels)

	kcc_dataset=get_data.data_import(kcc_files,kcc_folder)

	input_conv_data, kcc_subset_dump,kpi_subset_dump=get_data.data_convert_voxel_mc(vrm_system,dataset,point_index,kcc_dataset)
	
	kcc_regression,kcc_classification=hy_util.split_kcc(kcc_subset_dump)
	y_out_test=[kcc_regression,kcc_classification]

	mean_eval=0
	
	#Predict by setting all model param distributions to mean
	#Question asked on tensorflow, waiting for solution....
	#Evaluate mean vector
	if(mean_eval==1):
		
		y_preds_reg,y_preds_cla=deploy_model.model_mean_eval(input_conv_data,inference_model)
		metrics_eval=MetricsEval();
		
		eval_metrics_reg,accuracy_metrics_df_reg=metrics_eval.metrics_eval_base(y_preds_reg,y_out_test[0],logs_path)
		eval_metrics_cla,accuracy_metrics_df_cla=metrics_eval.metrics_eval_classification(y_preds_reg,y_out_test[1],logs_path)

		accuracy_metrics_df_reg.to_csv(logs_path+'/metrics_test_regression.csv')
		accuracy_metrics_df_cla.to_csv(logs_path+'/metrics_test_classification.csv')
		
		print("The Model Validation Metrics for Regression based KCCs")	
		print(accuracy_metrics_df_reg)
		accuracy_metrics_df_reg.mean().to_csv(logs_path+'/metrics_test_regression_summary.csv')
		print("The Model Validation Metrics Regression Summary")
		print(accuracy_metrics_df_reg.mean())

		print("The Model Validation Metrics for Classification based KCCs")	
		print(accuracy_metrics_df_cla)
		accuracy_metrics_df_cla.mean().to_csv(logs_path+'/metrics_test_classification_summary.csv')
		print("The Model Validation Metrics Classification Summary")
		print(accuracy_metrics_df_cla.mean())
		
		np.savetxt((deploy_path+"predicted_reg_edward.csv"), y_preds_reg, delimiter=",")
		np.savetxt((deploy_path+"predicted_reg_edward.csv"), y_preds_cla, delimiter=",")

		#print('Predicted Values saved to disk...')
		sys.exit()
	
	pred_vector,epistemic_vector,epistemic_vector_iqr,aleatoric_vector=deploy_model.model_inference(input_conv_data,inference_model,y_out_test,plots_path)

	epistemic_std_avg_reg=np.array(epistemic_vector[0]).mean(axis=0)
	epistemic_std_avg_cla=np.array(epistemic_vector[1]).mean(axis=0)

	avg_aleatoric_std=np.array(aleatoric_vector[0]).mean(axis=0)

	print("Average Epistemic Uncertainty of each KCC Regression: ",epistemic_std_avg_reg)
	print("Average Epistemic Uncertainty of each KCC Classification: ",epistemic_std_avg_cla)
	print("Average Aleatoric Uncertainty of each KCC: ",avg_aleatoric_std)
	
	evalerror=1

	if(evalerror==1):
		metrics_eval=MetricsEval();
		
		eval_metrics_reg,accuracy_metrics_df_reg=metrics_eval.metrics_eval_base(pred_vector[0],y_out_test[0],logs_path)
		eval_metrics_cla,accuracy_metrics_df_cla=metrics_eval.metrics_eval_classification(pred_vector[1],y_out_test[1],logs_path)

		accuracy_metrics_df_reg.to_csv(logs_path+'/metrics_test_regression.csv')
		accuracy_metrics_df_cla.to_csv(logs_path+'/metrics_test_classification.csv')
		
		print("The Model Validation Metrics for Regression based KCCs")	
		print(accuracy_metrics_df_reg)
		accuracy_metrics_df_reg.mean().to_csv(logs_path+'/metrics_test_regression_summary.csv')
		print("The Model Validation Metrics Regression Summary")
		print(accuracy_metrics_df_reg.mean())

		print("The Model Validation Metrics for Classification based KCCs")	
		print(accuracy_metrics_df_cla)
		accuracy_metrics_df_cla.mean().to_csv(logs_path+'/metrics_test_classification_summary.csv')
		print("The Model Validation Metrics Classification Summary")
		print(accuracy_metrics_df_cla.mean())
		
		np.savetxt((deploy_path+"predicted_reg.csv"), pred_vector[0], delimiter=",")
		np.savetxt((deploy_path+"predicted_cla.csv"), pred_vector[1], delimiter=",")
		#print('Predicted Values saved to disk...')

		np.savetxt((deploy_path+"pred_std_reg.csv"), epistemic_vector[0], delimiter=",")
		np.savetxt((deploy_path+"pred_std_cla.csv"), epistemic_vector[1], delimiter=",")
		#print('Predicted Standard Deviation Values saved to disk...')

		np.savetxt((deploy_path+"pred_iqr_reg.csv"), epistemic_vector_iqr[0], delimiter=",")
		np.savetxt((deploy_path+"pred_iqr_cla.csv"), epistemic_vector_iqr[1], delimiter=",")
		
		np.savetxt((deploy_path+"pred_aleatoric_std_reg.csv"), aleatoric_vector[0], delimiter=",")
		#print('Predicted Values saved to disk...')

		np.savetxt((deploy_path+"epistemic_std_avg_reg.csv"), epistemic_std_avg_reg, delimiter=",")
		np.savetxt((deploy_path+"epistemic_std_avg_cla.csv"), epistemic_std_avg_cla, delimiter=",")
		
		np.savetxt((deploy_path+"aleatoric_std_avg_reg.csv"), avg_aleatoric_std, delimiter=",")

		print('Model Logs saved to disk...')

