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

	def model_inference(self,inference_data,inference_model,y_pred,y_actual,plots_path,epistemic_samples=1000,run_id=0):
		"""model_inference method is used to infer from unknown sample(s) using the trained model 
				
				:param inference_data: Unknown dataset having same structure as the train dataset
				:type inference_data: numpy.array [samples*voxel_dim*voxel_dim*voxel_dim*deviation_channels] (required) (required)

				:param inference_model: Trained model
				:type inference_model: keras.model (required)
				
				:param print_result: Flag to indicate if the result needs to be printed, 0 by default, change to 1 in case the results need to be printed on the console
				:type print_result: int

		"""		
		#result=inference_model.(inference_data)
		y_std=np.zeros_like(y_pred)
		y_aleatoric_std=np.zeros_like(y_pred)
		plots_path_run_id=plots_path+'/plots_run_id_'+str(run_id)
		pathlib.Path(plots_path_run_id).mkdir(parents=True, exist_ok=True)

		for i in range(len(inference_data)):
			
			from scipy.stats import norm
			inference_sample=inference_data[i,:,:,:,:]
			print(inference_sample.shape)
			input_sample=np.array([inference_sample]*epistemic_samples)
			print(input_sample.shape)
			# input_sample=tf.cast(input_sample, tf.float32)
			# init = tf.global_variables_initializer()
			# with tf.Session() as sess:
			output=inference_model(input_sample)
			output_mean=output.mean()
			aleatoric_std=output.stddev()
			#print(output_mean)
			#print(aleatoric_std)
			# 	sess.run(init)
			# 	mean=sess.run(output_mean)
			# 	print(mean)
			#print((input_sample[0,:,:,:,:]==input_sample[50,:,:,:,:]).all())
			#output=inference_model(input_sample)
			#output_mean=output.mean()
			pred_mean=np.array(output_mean).mean(axis=0)
			aleatoric_mean=np.array(aleatoric_std).mean(axis=0)
			pred_std=np.array(output_mean).std(axis=0,ddof=1)
			output_mean=np.array(output_mean)
			
			print(pred_mean)
			print(pred_std)
			print(aleatoric_mean)
			print(output_mean.shape,aleatoric_std.shape)
			
			pred_plots=1
			
			if(pred_plots==1):
				for j in range(y_pred.shape[1]):
					plot_data=output_mean[:,j]
					actual_obv=y_actual[i,j]
					plt.hist(plot_data, range=(actual_obv-0.5,actual_obv+0.5),bins=40)
					plt.axvline(x=actual_obv,label="Actual Value = "+str(actual_obv),c='r')
					plt.axvline(x=pred_mean[j],label="Prediction Mean = "+str(actual_obv),c='c')
					plt.axvline(x=pred_mean[j]+norm.ppf(0.95)*pred_std[j], label="95 CI = "+str(pred_mean[j]+norm.ppf(0.95)*pred_std[j]),c='b')
					plt.axvline(x=pred_mean[j]-norm.ppf(0.95)*pred_std[j], label="95 CI = "+str(pred_mean[j]-norm.ppf(0.95)*pred_std[j]),c='b')
					plt.title("Prediction Distribution for KCC " + str(j) + " sample "+ str(i))
					#plt.show()
					plt.savefig(plots_path_run_id+"/"+ "sample_"+ str(i)+"_KCC_" + str(j) +'.png')
					plt.clf()
			y_pred[i,:]=pred_mean
			y_std[i,:]=pred_std
			y_aleatoric_std[i,:]=aleatoric_mean
			#print(pred_mean)
			#print(pred_std)

		return y_pred,y_std,y_aleatoric_std

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
	model_path=train_path+'/model'+'/Bayes_trained_model_0'
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
	model=dl_model.bayes_cnn_model_3d(voxel_dim,voxel_channels)

	#Inference from simulated data
	inference_model=deploy_model.get_model(model,model_path,voxel_dim,voxel_channels)

	kcc_dataset=get_data.data_import(kcc_files,kcc_folder)

	input_conv_data, kcc_subset_dump,kpi_subset_dump=get_data.data_convert_voxel_mc(vrm_system,dataset,point_index,kcc_dataset)
	y_pred=np.zeros_like(kcc_dataset)

	y_pred,y_std,y_aleatoric_std=deploy_model.model_inference(input_conv_data,inference_model,y_pred,kcc_dataset.values,plots_path);

	avg_std=np.array(y_std).mean(axis=0)
	avg_aleatoric_std=np.array(y_aleatoric_std).mean(axis=0)
	print("Average Epistemic Uncertainty of each KCC: ",avg_std)
	print("Average Aleatoric Uncertainty of each KCC: ",avg_aleatoric_std)
	evalerror=1

	if(evalerror==1):
		metrics_eval=MetricsEval();
		eval_metrics,accuracy_metrics_df=metrics_eval.metrics_eval_base(y_pred,kcc_dataset,logs_path)
		
		print('Evaluation Metrics: ',eval_metrics)
		accuracy_metrics_df.to_csv(logs_path+'/metrics_test.csv')
		
		np.savetxt((deploy_path+"predicted.csv"), y_pred, delimiter=",")
		#print('Predicted Values saved to disk...')

		np.savetxt((deploy_path+"pred_std.csv"), y_std, delimiter=",")
		#print('Predicted Standard Deviation Values saved to disk...')
		
		np.savetxt((deploy_path+"aleatoric_std.csv"), y_aleatoric_std, delimiter=",")
		#print('Predicted Values saved to disk...')

		np.savetxt((deploy_path+"aleatoric_std_avg.csv"), avg_aleatoric_std, delimiter=",")
		np.savetxt((deploy_path+"epistemic_std_avg.csv"), avg_std, delimiter=",")

		print('Model Logs saved to disk...')