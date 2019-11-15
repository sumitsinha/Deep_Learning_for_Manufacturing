import os
import sys
current_path=os.path.dirname(__file__)
parentdir = os.path.dirname(current_path)
sys.path.append("../Vizvalization")
sys.path.append("../utilities")
sys.path.append("../datasets")
sys.path.append("../trained_models")
path_var=os.path.join(os.path.dirname(__file__),"../utilities")
sys.path.append(path_var)
sys.path.insert(0,parentdir) 

import numpy as np
import pandas as pd
import tensorflow as tf
import dlmfg
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from training_viz import training_plot
from assemblyconfig import assembly_system

class TrainModel:
	
	def __init__(self, batch_size=32,epochs=150):
			self.batch_size=batch_size
			self.epochs=epochs

	def run_train_model(self,model,X_in,Y_out,split_ratio=0.3):
		
		model_path='./model/CNN_model_3D.h5'
		X_train, X_test, y_train, y_test = train_test_split(X_in, Y_out, test_size = split_ratio)

		#Checkpointer to save the best model
		checkpointer = ModelCheckpoint(,model_path verbose=1, save_best_only='mae')

		#Activating Tensorboard for Vizvalization
		tensorboard = TensorBoard(log_dir='./logs',histogram_freq=1, write_graph=True, write_images=True)
		history=model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=self.epochs, batch_size=self.batch_size,callbacks=[tensorboard,checkpointer])

		training_plot(history)
		inference_model=load_model(model_path)
		y_pred=inference_model.predict(y_test)

		eval_metrics=MetricsEval.metrics_eval(y_pred,y_test)
		return model,eval_metrics

if __name__ == '__main__':

	#Parsing from Config File
	data_type=assembly_system['data_type']
	application=assembly_system['application']
	part_type=assembly_system['part_type']
	data_format=assembly_system['data_format']
	assembly_type=assembly_system['assembly_type']
	assembly_kccs=assembly_system['assembly_kccs']	
	assembly_kpis=assembly_system['assembly_kpis']
	voxel_dim=assembly_system['voxel_dim']
	point_dim=assembly_system['point_dim']
	voxel_channels=assembly_system['voxel_channels']
	noise_levels=assembly_system['noise_levels']
	noise_type=assembly_system['noise_type']
	mapping_index=assembly_system['index_conv']
	file_names=assembly_system['data_files']
	
	#Objects of Measurement System, Assembly System, Get Infrence Data
	measurement_system=HexagonWlsScanner(data_type,application, system_noise,part_type,data_format)
	vrm_system=VRMSimulationModel(assembly_type,assembly_kccs,assembly_kpis,voxel_dim,point_dim,voxel_channels,noise_levels,noise_type)
	get_data=GetInferenceData();

	print('Importing and preprocessing Cloud-of-Point Data')
	
	get_train_data=GetTrainData(vrm_system)
	dataset=get_train_data.data_import(file_names)
	point_index=load_mapping_index(mapping_index)
	input_conv_data, kcc_subset_dump=get_train_data.data_convert_voxel(dataset,point_index)

	output_dimension=assembly_kccs
	dl_model=DLModel(output_dimension)
	model=dl_model.CNN_model_3D()

	train_model=TrainModel()
	trained_model,eval_metrics=train_model.run_train_model(model,input_conv_data,kcc_subset_dump)

	print("Model Training Complete..")
	print("The Model Validation Metrics are ")
	print(eval_metrics)



