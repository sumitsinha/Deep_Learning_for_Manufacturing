import numpy as np
import pandas as pd
import tensorflow as tf
import dfml
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

class TrainModel():
	
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

	#Objects of Measurement System and Assembly System
	measurement_system=HexagonWlsScanner(data_type,application, system_noise,part_type,data_format)
	vrm_system=VRMSimulationModel(assembly_type,assembly_kccs,assembly_kpis,voxel_dim,point_dim,voxel_channels,noise_levels,noise_type)

	print('Importing and preprocessing Cloud-of-Point Data')
	dataset_0 = pd.read_csv('./Data/car_halo_run1_ydev.csv',header=None)
	dataset_1 = pd.read_csv('./Data/car_halo_run2_ydev.csv',header=None)
	dataset_2 = pd.read_csv('./Data/car_halo_run3_ydev.csv',header=None)
	dataset_3 = pd.read_csv('./Data/car_halo_run4_ydev.csv',header=None)
	dataset_4 = pd.read_csv('./Data/car_halo_run5_ydev.csv',header=None)
	dataset_5 = pd.read_csv('./Data/car_halo_run6_ydev.csv',header=None)
	dataset_6 = pd.read_csv('./Data/car_halo_run7_ydev.csv',header=None)
	dataset_7 = pd.read_csv('./Data/car_halo_run8_ydev.csv',header=None)

	#Importing Dataset and calling data import function
	dataset = pd.concat([dataset_0, dataset_1,dataset_2,dataset_3,dataset_4,dataset_5,dataset_6,dataset_7], ignore_index=True)
	get_train_data=GetTrainData(vrm_system)
	input_conv_data, kcc_subset_dump=data_convert_voxel(dataset)

	#%%
	output_dimension=assembly_kccs
	dl_model=DLModel(output_dimension)
	model=dl_model.CNN_model_3D()

	train_model=TrainModel()

	trained_model,eval_metrics=train_model.run_train_model(model,input_conv_data,kcc_subset_dump)

	print("Model Training Complete..")
	print("The Model Validation Metrics are ")
	print(eval_metrics)



