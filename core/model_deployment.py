import os
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
from keras.models import load_model

#Importing Config files
import assemblyconfig_halostamping as config
import modelconfig_train as cftrain

#Importing required modules from the package
from measurement_system import HexagonWlsScanner
from assembly_system import VRMSimulationModel
from assembly_system import PartType
from wls400a_system import GetInferenceData

def get_model(model_path):

	try:
		inference_model=load_model(model_path)
	except AssertionError as error:
		print(error)
		print('Model not found at this path ',model_path, ' Update path in config file if required')

	return inference_model

def model_inference(inference_data,inference_model):
	
	result=inference_model.predict(inference_data)
	description="The Process Parameters variations are inferred from the obtained meeasurement data and the trained CNN based model"
	print('The model estimates are: ')
	print(result)

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
	file_names=config.assembly_system['data_files']
	system_noise=config.assembly_system['system_noise']
	aritifical_noise=config.assembly_system['aritifical_noise']
	
	print('Intilizing the Assembly System and Measurement System....')
	measurement_system=HexagonWlsScanner(data_type,application,system_noise,part_type,data_format)
	vrm_system=VRMSimulationModel(assembly_type,assembly_kccs,assembly_kpis,part_name,part_type,voxel_dim,voxel_channels,point_dim,aritifical_noise)
	
	#Generate Paths
	train_path='../trained_models/'+part_type
	model_path=train_path+'/model'+'/trained_model.h5'

	deploy_path=train_path+'/deploy/'


	#Voxel Mapping File
	index_file='Halo_cov_index_data_64.dat'
	voxel_path='../utilities/'+index_file
	#Make an Object of the Measurment System Class
	measurement_system=HexagonWlsScanner(data_type,application, system_noise,part_type,data_format)
	#Make an Object of the Assembly System Class
	assembly_system=PartType(assembly_type,assembly_kccs,assembly_kpis,part_name,part_type,voxel_dim,voxel_channels,point_dim)

	#Load required files
	
	measurement_file='MC1.txt'
	measurement_path=deploy_path+measurement_file

	#Make an object of Get Data Class
	get_data=GetInferenceData();

	#Call functions of the get Data Class
	measurement_data=get_data.load_measurement_file(measurement_path)
	voxel_point_index=get_data.load_mapping_index(voxel_path)
	y_dev_data_filtered=get_data.data_pre_processing(measurement_data,voxel_channels)
	voxel_dev_data=get_data.voxel_mapping(y_dev_data_filtered,voxel_point_index,point_dim,voxel_dim,voxel_channels)
	inference_model=get_model(model_path)

	model_inference(voxel_dev_data,inference_model);