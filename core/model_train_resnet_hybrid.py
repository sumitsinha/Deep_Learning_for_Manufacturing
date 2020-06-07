""" The model train file trains the model on the download dataset and other parameters specified in the assemblyconfig file
The main function runs the training and populates the created file structure with the trained model, logs and plots
"""

import os
import sys
current_path=os.path.dirname(__file__)
parentdir = os.path.dirname(current_path)

os.environ["CUDA_VISIBLE_DEVICES"]="0" # Nvidia Quadro GV100
#os.environ["CUDA_VISIBLE_DEVICES"]="1" # Nvidia Quadro M2000

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
#from keras import backend as K
#K.clear_session()

#Importing Config files
import assembly_config as config
import model_config as cftrain
import hybrid_utils as hy_util
#Importing required modules from the package
from measurement_system import HexagonWlsScanner
from assembly_system import VRMSimulationModel
from wls400a_system import GetInferenceData
from data_import import GetTrainData
from core_model import DLModel
from training_viz import TrainViz
from metrics_eval import MetricsEval
from encode_decode_model import Encode_Decode_Model
#from keras_lr_multiplier import LRMultiplier

class TrainModel:
	"""Train Model Class, the initialization parameters are parsed from modelconfig_train.py file
		
		:param batch_size: mini batch size while training the model 
		:type batch_size: int (required)

		:param epochs: no of epochs to conduct training
		:type epochs: int (required)

		:param split_ratio: train and validation split for the model
		:type assembly_system: float (required)

		The class contains run_train_model method
	"""	
	def __init__(self,batch_size,epochs,split_ratio):
			self.batch_size=batch_size
			self.epochs=epochs
			self.split_ratio=split_ratio
			

	def run_train_model(self,model,X_in,Y_out,model_path,logs_path,plots_path,activate_tensorboard=0,run_id=0,tl_type='full_fine_tune'):
		"""run_train_model function trains the model on the dataset and saves the trained model,logs and plots within the file structure, the function prints the training evaluation metrics
			
			:param model: 3D CNN model compiled within the Deep Learning Class, refer https://keras.io/models/model/ for more information 
			:type model: keras.models (required)

			:param X_in: Train dataset input (predictor variables), 3D Voxel representation of the cloud of point and node deviation data obtained from the VRM software based on the sampling input
			:type X_in: numpy.array [samples*voxel_dim*voxel_dim*voxel_dim*deviation_channels] (required)
			
			:param Y_out: Train dataset output (variables to predict), Process Parameters/KCCs obtained from sampling
			:type Y_out: numpy.array [samples*assembly_kccs] (required)

			:param model_path: model path at which the trained model is saved
			:type model_path: str (required)
			
			:param logs_path: logs path where the training metrics file is saved
			:type logs_path: str (required)

			:param plots_path: plots path where model training loss convergence plot is saved
			:type plots_path: str (required)

			:param activate_tensorboard: flag to indicate if tensorboard should be added in model callbacks for better visualization, 0 by default, set to 1 to activate tensorboard
			:type activate_tensorboard: int

			:param run_id: Run id index used in data study to conduct multiple training runs with different dataset sizes, defaults to 0
			:type run_id: int			
		"""			
		import tensorflow as tf
		from sklearn.model_selection import train_test_split
		from tensorflow.keras.models import load_model
		from tensorflow.keras.callbacks import ModelCheckpoint
		from tensorflow.keras.callbacks import TensorBoard

		model_file_path=model_path+'/trained_model_resnet_hybrid_'+str(run_id)+'.h5'
		
		X_train, X_test, y_train_reg, y_test_reg,y_train_cla, y_test_cla = train_test_split(X_in, Y_out[0],Y_out[1], test_size = self.split_ratio)
		
		y_train=[y_train_reg,y_train_cla]
		y_test=[y_test_reg,y_test_cla]

		print("Data Split Completed")
		
		#Checkpointer to save the best model
		checkpointer = ModelCheckpoint(model_file_path, verbose=1, save_best_only='val_loss',save_weights_only=True)
		
		callbacks=[checkpointer]
		
		if(activate_tensorboard==1):
			#Activating Tensorboard for Visualization
			tensorboard = TensorBoard(log_dir=logs_path,histogram_freq=1, write_graph=True, write_images=True)
			callbacks=[checkpointer,tensorboard]
		
		tensorboard = TensorBoard(log_dir=logs_path,histogram_freq=1, write_graph=True, write_images=True)
		history=model.fit(x=X_train, y=y_train, validation_data=(X_test,y_test), epochs=self.epochs, batch_size=self.batch_size,callbacks=callbacks)
		
		#trainviz=TrainViz()
		#trainviz.training_plot(history,plots_path,run_id)
	
		model.load_weights(model_file_path)
			
		y_pred=model.predict(X_test)

		metrics_eval=MetricsEval();
		eval_metrics_reg,accuracy_metrics_df_reg=metrics_eval.metrics_eval_base(y_pred[0],y_test[0],logs_path)
		eval_metrics_cla,accuracy_metrics_df_cla=metrics_eval.metrics_eval_classification(y_pred[1],y_test[1],logs_path)
		
		return model,accuracy_metrics_df_reg,accuracy_metrics_df_cla

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

	#added for hybrid model
	categorical_kccs=config.assembly_system['categorical_kccs']
	
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

	#Objects of Measurement System, Assembly System, Get Inference Data
	print('Initializing the Assembly System and Measurement System....')
	
	measurement_system=HexagonWlsScanner(data_type,application,system_noise,part_type,data_format)
	vrm_system=VRMSimulationModel(assembly_type,assembly_kccs,assembly_kpis,part_name,part_type,voxel_dim,voxel_channels,point_dim,aritifical_noise)
	get_data=GetTrainData();

	#print(input_conv_data.shape,kcc_subset_dump.shape)

	if(activate_tensorboard==1):
		tensorboard_str='tensorboard' + '--logdir '+logs_path
		print('Visualize at Tensorboard using ', tensorboard_str)
	print('Importing and Preprocessing Cloud-of-Point Data')
	
	dataset=[]
	dataset.append(get_data.data_import(file_names_x,data_folder))
	dataset.append(get_data.data_import(file_names_y,data_folder))
	dataset.append(get_data.data_import(file_names_z,data_folder))
	point_index=get_data.load_mapping_index(mapping_index)

	kcc_dataset=get_data.data_import(kcc_files,kcc_folder)
	input_conv_data, kcc_subset_dump,kpi_subset_dump=get_data.data_convert_voxel_mc(vrm_system,dataset,point_index,kcc_dataset)
	
	#Added Function to split KCCs to regression and classification 
	kcc_regression,kcc_classification=hy_util.split_kcc(kcc_subset_dump)

	print('Building 3D CNN model')

	output_dimension=assembly_kccs
	
	dl_model_unet=Encode_Decode_Model(output_dimension)
	model=dl_model_unet.resnet_3d_cnn_hybrid(voxel_dim,voxel_channels,kcc_classification.shape[1])
	
	print(model.summary())
	#sys.exit()
	print('Training 3D CNN model')
	model_outputs=[kcc_regression,kcc_classification]
	
	train_model=TrainModel(batch_size,epocs,split_ratio)
	
	trained_model,accuracy_metrics_df_reg,accuracy_metrics_df_cla=train_model.run_train_model(model,input_conv_data,model_outputs,model_path,logs_path,plots_path,activate_tensorboard)
	
	accuracy_metrics_df_reg.to_csv(logs_path+'/metrics_train_regression.csv')
	accuracy_metrics_df_cla.to_csv(logs_path+'/metrics_train_classification.csv')
	
	print("Model Training Complete..")
	
	print("The Model Validation Metrics for Regression based KCCs")	
	print(accuracy_metrics_df_reg)
	accuracy_metrics_df_reg.mean().to_csv(logs_path+'/metrics_train_regression_summary.csv')
	print("The Model Validation Metrics Regression Summary")
	print(accuracy_metrics_df_reg.mean())

	print("The Model Validation Metrics for Classification based KCCs")	
	print(accuracy_metrics_df_cla)
	accuracy_metrics_df_cla.mean().to_csv(logs_path+'/metrics_train_classification_summary.csv')
	print("The Model Validation Metrics Classification Summary")
	print(accuracy_metrics_df_cla.mean())

	print('Training Completed Successfully')



