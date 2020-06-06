""" The model deploy file is used to leverage a trained model to perform inference on unknown set of node deviations.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

import numpy as np
import pandas as pd
import tensorflow as tf
import csv
import logging
tf.get_logger().setLevel(logging.ERROR)

from tensorflow.keras.models import load_model


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
from encode_decode_model import Encode_Decode_Model
#from cam_viz import CamViz
#from cop_viz import CopViz

class DeployModel:
	"""The Deploy Model class is used to import a trained model and use it to infer on unknown data

	"""
	def get_model(self,model,model_path):
		"""get_model method is is used to retrieve the trained model from a given path
				
				:param model_path: Path to the trained model, ideally it should be same as the train model path output
				:type model_path: str (required)
		"""
		from tensorflow.keras.models import load_model
		model.load_weights(model_path)
		print("Trained Model Weights loaded successfully")

		return model

	def model_inference(self,inference_data,inference_model,deploy_path,print_result=0,plot_result=0,get_cam_data=0,append_result=0):
		"""model_inference method is used to infer from unknown sample(s) using the trained model 
				
				:param inference_data: Unknown dataset having same structure as the train dataset
				:type inference_data: numpy.array [samples*voxel_dim*voxel_dim*voxel_dim*deviation_channels] (required) (required)

				:param inference_model: Trained model
				:type inference_model: keras.model (required)
				
				:param print_result: Flag to indicate if the result needs to be printed, 0 by default, change to 1 in case the results need to be printed on the console
				:type print_result: int

		"""		
		print("Predicting using model...")
		result=inference_model.predict(inference_data)
		description="The Process Parameters variations are inferred from the obtained measurement data and the trained CNN based model"
		
		#rounded_result=np.round(result,2)
		
		if(print_result==1):
			print('The model estimates are: ')
			print(result)
		
		return result

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
	

	print('Initializing the Assembly System and Measurement System....')
	measurement_system=HexagonWlsScanner(data_type,application,system_noise,part_type,data_format)
	vrm_system=VRMSimulationModel(assembly_type,assembly_kccs,assembly_kpis,part_name,part_type,voxel_dim,voxel_channels,point_dim,aritifical_noise)
	deploy_model=DeployModel()
	
	#Generate Paths
	train_path='../trained_models/'+part_type
	model_path=train_path+'/model'+'/trained_model_resnet_hybrid_0.h5'
	logs_path=train_path+'/logs'
	deploy_path=train_path+'/deploy/'

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
	
	kcc_dataset=get_data.data_import(kcc_files,kcc_folder)
	input_conv_data, kcc_subset_dump,kpi_subset_dump=get_data.data_convert_voxel_mc(vrm_system,dataset,point_index,kcc_dataset)
	
	kcc_regression,kcc_classification=hy_util.split_kcc(kcc_subset_dump)

	print('Building 3D CNN model')

	output_dimension=assembly_kccs
	
	dl_model_unet=Encode_Decode_Model(output_dimension)
	model=dl_model_unet.resnet_3d_cnn_hybrid(voxel_dim,voxel_channels,kcc_classification.shape[1])
	
	#sys.exit()
	y_test=[kcc_regression,kcc_classification]
	#Inference from simulated data
	
	inference_model=deploy_model.get_model(model,model_path)
	print(inference_model.summary())

	y_pred=deploy_model.model_inference(input_conv_data,inference_model,deploy_path,print_result=0,plot_result=0);

	evalerror=1

	if(evalerror==1):
		metrics_eval=MetricsEval();
		
		eval_metrics_reg,accuracy_metrics_df_reg=metrics_eval.metrics_eval_base(y_pred[0],y_test[0],logs_path)
		eval_metrics_cla,accuracy_metrics_df_cla=metrics_eval.metrics_eval_classification(y_pred[1],y_test[1],logs_path)
		
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
