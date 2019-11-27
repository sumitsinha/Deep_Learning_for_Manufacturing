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
#path_var=os.path.join(os.path.dirname(__file__),"../utilities")
#sys.path.append(path_var)
#sys.path.insert(0,parentdir) 

#Importing Required Modules
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
K.clear_session()

#Importing Config files
import assemblyconfig_halostamping as config
import modelconfig_train as cftrain

#Importing required modules from the package
from measurement_system import HexagonWlsScanner
from assembly_system import VRMSimulationModel
from wls400a_system import GetInferenceData
from data_import import GetTrainData
from core_model import DLModel
from training_viz import TrainViz
from metrics_eval import MetricsEval

class TrainModel:
	
	def __init__(self, batch_size=32,epochs=150):
			self.batch_size=batch_size
			self.epochs=epochs

	def run_train_model(self,model,X_in,Y_out,model_path,logs_path,plots_path,split_ratio=0.2):
		
		from sklearn.model_selection import train_test_split
		from keras.models import load_model
		from keras.callbacks import ModelCheckpoint
		from keras.callbacks import TensorBoard

		model_file_path=model_path+'/trained_model.h5'
		X_train, X_test, y_train, y_test = train_test_split(X_in, Y_out, test_size = split_ratio)
		print("Data Split Completed")
		#Checkpointer to save the best model
		checkpointer = ModelCheckpoint(model_file_path, verbose=1, save_best_only='mae')

		#Activating Tensorboard for Vizvalization
		tensorboard = TensorBoard(log_dir=logs_path,histogram_freq=1, write_graph=True, write_images=True)
		history=model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=self.epochs, batch_size=self.batch_size,callbacks=[checkpointer])
		
		trainviz=TrainViz()
		trainviz.training_plot(history,plots_path)
		
		inference_model=load_model(model_file_path)
		y_pred=inference_model.predict(X_test)

		metrics_eval=MetricsEval();
		eval_metrics=metrics_eval.metrics_eval_base(y_pred,y_test,logs_path)
		return model,eval_metrics

	def run_train_model_dynamic():
		pass

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
	file_names_x=config.assembly_system['data_files_x']
	file_names_y=config.assembly_system['data_files_y']
	file_names_z=config.assembly_system['data_files_z']
	system_noise=config.assembly_system['system_noise']
	aritifical_noise=config.assembly_system['aritifical_noise']
	data_folder=config.assembly_system['data_folder']
	kcc_folder=config.assembly_system['kcc_folder']
	kcc_files=config.assembly_system['kcc_files']

	print('Creating file Structure....')
	folder_name=part_type
	train_path='../trained_models/'+part_type
	pathlib.Path(train_path).mkdir(parents=True, exist_ok=True) 

	model_path=train_path+'/model'
	pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
	
	logs_path=train_path+'/logs'
	pathlib.Path(logs_path).mkdir(parents=True, exist_ok=True)

	plots_path=train_path+'/plots'
	pathlib.Path(plots_path).mkdir(parents=True, exist_ok=True)

	deployment_path=train_path+'/deploy'
	pathlib.Path(deployment_path).mkdir(parents=True, exist_ok=True)

	#Objects of Measurement System, Assembly System, Get Infrence Data
	print('Intilizing the Assembly System and Measurement System....')
	measurement_system=HexagonWlsScanner(data_type,application,system_noise,part_type,data_format)
	vrm_system=VRMSimulationModel(assembly_type,assembly_kccs,assembly_kpis,part_name,part_type,voxel_dim,voxel_channels,point_dim,aritifical_noise)
	get_data=GetTrainData();

	print('Importing and Preprocessing Cloud-of-Point Data')
	dataset=[]
	dataset.append(get_data.data_import(file_names_x,data_folder))
	dataset.append(get_data.data_import(file_names_y,data_folder))
	dataset.append(get_data.data_import(file_names_z,data_folder))
	point_index=get_data.load_mapping_index(mapping_index)

	kcc_dataset=get_data.data_import(kcc_files,kcc_folder)
	
	input_conv_data, kcc_subset_dump,kpi_subset_dump=get_data.data_convert_voxel_mc(vrm_system,dataset,point_index,kcc_dataset)

	print(input_conv_data.shape,kcc_subset_dump.shape)
	print('Building 3D CNN model')

	output_dimension=assembly_kccs
	dl_model=DLModel(output_dimension)
	model=dl_model.cnn_model_3d(voxel_dim,voxel_channels)

	print('Training 3D CNN model')
	tensorboard_str='tensorboard' + '--logdir '+logs_path
	print('Vizavlize at Tensorboard using ', tensorboard_str)
	train_model=TrainModel()
	trained_model,eval_metrics=train_model.run_train_model(model,input_conv_data,kcc_subset_dump,model_path,logs_path,plots_path)

	print("Model Training Complete..")
	print("The Model Validation Metrics are ")
	print(eval_metrics)

	print('Training Completed Succssesfully')



