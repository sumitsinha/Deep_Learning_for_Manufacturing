import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
import argparse

def get_model(model_path='./trained_model/model_final.h5'):

	try:
		inference_model=load_model(model_path)
	except AssertionError as error:
		print(error)
		print('Model not found at this path ',model_path, ' Update path in command line if required')

	return inference_model

def model_inference(inference_data,inference_model):
	result=inference_model.predict(inference_data)

	description="The Process Parameters are inferred from the obtained meeasurement data and the trained CNN based model"
	print('The model estimates are: ')
	print(result)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Arguments to initiate Measurement System Class and Assembly System Class")
    parser.add_argument("-D", "--data_type", help = "Example: 3D Point Cloud Data", required = False, default = "3D Point Cloud Data")
    parser.add_argument("-A", "--application", help = "Example: Inline Root Cause Analysis", required = False, default = "Inline Root Cause Analysis")
    parser.add_argument("-P", "--part_type", help = "Example: Door Inner and Hinge Assembly", required = False, default = "Door Inner and Hinge Assembly")
    parser.add_argument("-F", "--data_format", help = "Example: Complete vs Partial Data", required = False, default = "Complete")
	
	parser.add_argument("-S", "--assembly_type", help = "Example: Multi-Stage vs Single-Stage", required = False, default = "Single-Stage")
    parser.add_argument("-C", "--assembly_kccs", help = "Number of KCCs for the Assembly", required = False, default =15,type=int )
    parser.add_argument("-I", "--assembly_kpis	", help = "Number of KPIs for the Assembly", required = False, default = 6,type=int)
    parser.add_argument("-V", "--voxel_dim", help = "The Granularity of Voxels - 32 64 128", required = False, default = 64,type=int)
    parser.add_argument("-V", "--voxel_channels", help = "Number of Channels - 1 or 3", required = False, default = 1,type=int)
	argument = parser.parse_args()
	
	data_type=argument.data_type
	application=argument.application
	part_type=argument.part_type
	data_format=argument.data_format
	assembly_type=argument.assembly_type	
	assembly_kccs=argument.assembly_kccs	
	assembly_kpis=argument.assembly_kpis
	voxel_dim=argument.voxel_dim
	voxel_channels=argument.voxel_channels		
	
	#Make an Object of the Measurment System Class
	measurement_system=HexagonWlsScanner(data_type,application, system_noise,part_type,data_format)
	#Make an Object of the Assembly System Class
	assembly_system=PartType(assembly_type,assembly_kccs,assembly_kpis,voxel_dim,voxel_channels,part_type)

	#Load required files
	index_file='Halo_cov_index_data.dat'
	measurement_file=''
	model_path=
	#Make an object of Get Data Class
	get_data=GetInferenceData();

	#Call functions of the get Data Class
	measurement_data=load_measurement_file(measurement_file_name)
	voxel_point_index=get_data.load_mapping_index(index_file)
	y_dev_data_filtered=get_data.data_pre_processing(measurement_data,voxel_channels)
	voxel_dev_data=get_data.voxel_mapping(y_dev_data_filtered,voxel_point_index,voxel_dim)
	inference_model=get_model(model_path,measurement_system,assembly_system)

	model_inference(voxel_dev_data,inference_model);