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

	parser = argparse.ArgumentParser(description="Arguments to initiate Measurement System Class and Assembly System Class")
    parser.add_argument("-D", "--data_type", help = "Example: 3D Point Cloud Data", required = False, default = "3D Point Cloud Data")
    parser.add_argument("-A", "--application", help = "Example: Inline Root Cause Analysis", required = False, default = "Inline Root Cause Analysis")
    parser.add_argument("-P", "--part_type", help = "Example: Door Inner and Hinge Assembly", required = False, default = "Door Inner and Hinge Assembly")
    parser.add_argument("-F", "--data_format", help = "Example: Complete vs Partial Data", required = False, default = "Complete")
	parser.add_argument("-S", "--assembly_type", help = "Example: Multi-Stage vs Single-Stage", required = False, default = "Single-Stage")
    parser.add_argument("-C", "--assembly_kccs", help = "Number of KCCs for the Assembly", required = False, default =15,type=int )
    parser.add_argument("-I", "--assembly_kpis	", help = "Number of KPIs for the Assembly", required = False, default = 6,type=int)
    parser.add_argument("-V", "--voxel_dim", help = "The Granularity of Voxels - 32 64 128", required = False, default = 64,type=int)
    parser.add_argument("-P", "--point_dim", help = "Number of key Nodes", required = True, type=int)
    parser.add_argument("-C", "--voxel_channels", help = "Number of Channels - 1 or 3", required = False, default = 1,type=int)
    parser.add_argument("-N", "--noise_levels", help = "Amount of Artificial Noise to add while training", required = False, default = 0.1,type=float)
    parser.add_argument("-T", "--noise_type", help = "Type of noise to be added uniform/Gaussian default uniform", required = False, default = "uniform")
	argument = parser.parse_args()
	
	data_type=argument.data_type
	application=argument.application
	part_type=argument.part_type
	data_format=argument.data_format
	assembly_type=argument.assembly_type	
	assembly_kccs=argument.assembly_kccs	
	assembly_kpis=argument.assembly_kpis
	voxel_dim=argument.voxel_dim
	point_dim=argument.point_dim
	voxel_channels=argument.voxel_channels
	noise_levels=argument.noise_levels
	noise_type=argument.noise_type

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

	#Objects of Measurement System and Assembly System
	measurement_system=HexagonWlsScanner(data_type,application, system_noise,part_type,data_format)
	vrm_system=VRMSimulationModel(assembly_type,assembly_kccs,assembly_kpis,voxel_dim,point_dim,voxel_channels,noise_levels,noise_type)
	get_data=GetInferenceData();
	filename='Point_Mapping_Index'
	print('Importing and preprocessing Cloud-of-Point Data')
	
	file_names=['car_halo_run1_ydev.csv','car_halo_run2_ydev.csv','car_halo_run3_ydev.csv','car_halo_run4_ydev.csv','car_halo_run5_ydev.csv']
	get_train_data=GetTrainData(vrm_system)
	dataset=get_train_data.data_import(file_names)
	point_index=load_mapping_index(filename)
	input_conv_data, kcc_subset_dump=get_train_data.data_convert_voxel(dataset,point_index)

	#%%
	output_dimension=assembly_kccs
	dl_model=DLModel(output_dimension)
	model=dl_model.CNN_model_3D()

	train_model=TrainModel()

	trained_model,eval_metrics=train_model.run_train_model(model,input_conv_data,kcc_subset_dump)



	print("Model Training Complete..")
	print("The Model Validation Metrics are ")
	print(eval_metrics)



