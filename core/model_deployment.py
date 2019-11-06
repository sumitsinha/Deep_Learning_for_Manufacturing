import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
import argparse

def infer_from_data():
	model_path='./trained_model/model_final.h5'

	try:
		inference_model=load_model(model_path)
	except AssertionError as error:
		print(error)
		print('Model not found at this path ',model_path, ' Update path in command line if required')


	inference_data=get_inference_data()

	result=inference_model(inference_data)

	description="The Process Parameters are inferred from the obtained meeasurement data and the trained CNN based model"
	print('The model estimates are: ')


def get_inference_data(data_source='WLS_3D_scan_data')

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Arguments to initiate Measurement System Class and Assembly System Class")
    parser.add_argument("-D", "--data_type", help = "Example: 3D Point Cloud Data", required = False, default = "3D Point Cloud Data")
    parser.add_argument("-A", "--application", help = "Example: Inline Root Cause Analysis", required = False, default = "Inline Root Cause Analysis")
    parser.add_argument("-P", "--part_type", help = "Example: Door Inner and Hinge Assembly", required = False, default = "Door Inner and Hinge Assembly")
    parser.add_argument("-o", "--data_format", help = "Example: Complete vs Partial Data", required = False, default = "Complete")
	argument = parser.parse_args()
	
	data_type=argument.data_type
	application=argument.application
	part_type=argument.part_type
	data_format=argument.data_format
	
	#Make an Object of the Measurment System Class
	measurment_system=HexagonWlsScanner(data_type,application, system_noise,part_type,data_format)

	#Make an Object of the Assembly System Class
	assembly_system=PartType()