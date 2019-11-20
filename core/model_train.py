import os
import sys
current_path=os.path.dirname(__file__)
parentdir = os.path.dirname(current_path)
sys.path.append("../visualization")
sys.path.append("../utilities")
sys.path.append("../datasets")
sys.path.append("../trained_models")
path_var=os.path.join(os.path.dirname(__file__),"../utilities")
sys.path.append(path_var)
sys.path.insert(0,parentdir) 

#Importing Required Modules
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import tqdm

#Importing required modules from the package
import assemblyconfig_halostamping as config
from measurement_system import HexagonWlsScanner
from assembly_system import VRMSimulationModel
from wls400a_system import GetInferenceData
from data_import import GetTrainData
from core_model import DLModel
from training_viz import TrainViz
from metrics_eval import MetricsEval

class TrainModel:
	
	def __init__(self, batch_size=32,epochs=10):
			self.batch_size=batch_size
			self.epochs=epochs

	def run_train_model(self,model,X_in,Y_out,split_ratio=0.3):
		
		from sklearn.model_selection import train_test_split
		from keras.models import load_model
		from keras.callbacks import ModelCheckpoint
		from keras.callbacks import TensorBoard
		print(X_in.shape)
		print(Y_out.shape)
		model_path='../trained_models/CNN_model_3D.h5'
		X_train, X_test, y_train, y_test = train_test_split(X_in, Y_out, test_size = split_ratio)

		#Checkpointer to save the best model
		checkpointer = ModelCheckpoint(model_path, verbose=1, save_best_only='mae')

		#Activating Tensorboard for Vizvalization
		tensorboard = TensorBoard(log_dir='./logs',histogram_freq=1, write_graph=True, write_images=True)
		history=model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=self.epochs, batch_size=self.batch_size,callbacks=[tensorboard,checkpointer])
		
		trainviz=TrainViz()
		trainviz.training_plot(history)
		
		inference_model=load_model(model_path)
		y_pred=inference_model.predict(X_test)

		metrics_eval=MetricsEval();
		eval_metrics=metrics_eval.metrics_eval(y_pred,y_test)
		return model,eval_metrics

if __name__ == '__main__':

	#Parsing from Config File
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

	#Objects of Measurement System, Assembly System, Get Infrence Data
	measurement_system=HexagonWlsScanner(data_type,application,system_noise,part_type,data_format)
	vrm_system=VRMSimulationModel(assembly_type,assembly_kccs,assembly_kpis,part_name,part_type,voxel_dim,voxel_channels,point_dim,aritifical_noise)
	get_data=GetTrainData();

	print('Importing and preprocessing Cloud-of-Point Data')
	dataset=get_data.data_import(file_names)
	point_index=get_data.load_mapping_index(mapping_index)
	
	input_conv_data, kcc_subset_dump,kpi_subset_dump=get_data.data_convert_voxel(vrm_system,dataset,point_index)

	output_dimension=assembly_kccs
	dl_model=DLModel(output_dimension)
	model=dl_model.cnn_model_3d(voxel_dim,voxel_channels)

	train_model=TrainModel()
	trained_model,eval_metrics=train_model.run_train_model(model,input_conv_data,kcc_subset_dump)

	print("Model Training Complete..")
	print("The Model Validation Metrics are ")
	print(eval_metrics)



