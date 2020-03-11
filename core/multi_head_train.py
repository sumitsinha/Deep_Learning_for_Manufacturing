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

#Importing required modules from the package
from measurement_system import HexagonWlsScanner
from assembly_system import VRMSimulationModel
from wls400a_system import GetInferenceData
from data_import import GetTrainData
from multi_head_model import Multi_Head_DLModel
from training_viz import TrainViz
from metrics_eval import MetricsEval
#from model_train import TrainModel
from keras_lr_multiplier import LRMultiplier

class Multi_Head_TrainModel:
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
			

	def run_train_model(self,model,X_train,y_train,X_test,y_test,model_path,logs_path,plots_path,activate_tensorboard=0,run_id=0,tl_type='full_fine_tune'):
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

		model_file_path=model_path+'/trained_model_'+str(run_id)+'.h5'
		
		#X_train, X_test, y_train, y_test = train_test_split(X_in, Y_out, test_size = self.split_ratio)
		print("Data Split Completed")
		
		#Checkpointer to save the best model
		checkpointer = ModelCheckpoint(model_file_path, verbose=1, save_best_only='mae')
		callbacks=[checkpointer]
		
		if(activate_tensorboard==1):
			#Activating Tensorboard for Visualization
			tensorboard = TensorBoard(log_dir=logs_path,histogram_freq=1, write_graph=True, write_images=True)
			callbacks=[checkpointer,tensorboard]
		
		tensorboard = TensorBoard(log_dir='C:\\Users\\SINHA_S\\Desktop\\dlmfg_package\\dlmfg\\trained_models\\inner_rf_assembly\\multi_head\\logs',histogram_freq=1, write_graph=True, write_images=True)
		history=model.fit(x=X_train, y=y_train, validation_data=(X_test,y_test), epochs=self.epochs, batch_size=self.batch_size,callbacks=callbacks)
		
		trainviz=TrainViz()
		#trainviz.training_plot(history,plots_path,run_id)
		
		if(tl_type=='variable_lr'):
			inference_model=load_model(model_file_path, custom_objects={'LRMultiplier': LRMultiplier})
		else:
			inference_model=load_model(model_file_path)
		
		print('Compiling test metrics...')
		y_pred=inference_model.predict(X_test)

		metrics_eval=MetricsEval();
		eval_metrics,accuracy_metrics_df=metrics_eval.metrics_eval_base(y_pred,y_test,logs_path)
		return model,eval_metrics,accuracy_metrics_df

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
	test_kcc_files=config.assembly_system['test_kcc_files']

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
	
	print('Parsing multi-head network')

	assembly_stages=config.assembly_system['assembly_stages']
	multi_stage_params=config.multi_stage_data_construct

	print('Creating file Structure....')
	
	folder_name=part_type
	train_path='../trained_models/'+part_type+'/multi_head'
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
	print('Building 3D CNN model')

	output_dimension=assembly_kccs
	
	dl_model=Multi_Head_DLModel(model_type,assembly_stages,output_dimension)
	model=dl_model.multi_head_standard_cnn_model_3d(voxel_dim,voxel_channels)

	print('Importing data')
	
	point_index=get_data.load_mapping_index(mapping_index)
	kcc_dataset=get_data.data_import(kcc_files,kcc_folder)
	kcc_dataset_test=get_data.data_import(test_kcc_files,kcc_folder)

	x_in=[]
	x_test=[]

	for stage in multi_stage_params:
		
		print('Importing and Pre-Processing Train data for: ','station: ',stage['station_id'],'stage: ',stage['stage_id'])
		dataset=[]
		dataset.append(get_data.data_import(stage['data_files_x'],data_folder))
		dataset.append(get_data.data_import(stage['data_files_y'],data_folder))
		dataset.append(get_data.data_import(stage['data_files_z'],data_folder))
		
		input_conv_data, kcc_subset_dump,kpi_subset_dump=get_data.data_convert_voxel_mc(vrm_system,dataset,point_index,kcc_dataset)
		
		x_in.append(input_conv_data)

	y_out=kcc_subset_dump

	for stage in multi_stage_params:
		
		print('Importing and Pre-Processing test data for: ','station: ',stage['station_id'],'stage: ',stage['stage_id'])
		dataset=[]
		dataset.append(get_data.data_import(stage['test_data_files_x'],data_folder))
		dataset.append(get_data.data_import(stage['test_data_files_y'],data_folder))
		dataset.append(get_data.data_import(stage['test_data_files_z'],data_folder))
		
		input_conv_data_test, kcc_subset_dump_test,kpi_subset_dump=get_data.data_convert_voxel_mc(vrm_system,dataset,point_index,kcc_dataset_test)
		
		x_test.append(input_conv_data_test)

	y_test=kcc_subset_dump_test
	
	print('Data Import completed')
	train_model=Multi_Head_TrainModel(batch_size,epocs,split_ratio)
	
	trained_model,eval_metrics,accuracy_metrics_df=train_model.run_train_model(model,x_in,y_out,x_test,y_test,model_path,logs_path,plots_path,activate_tensorboard)
	
	accuracy_metrics_df.to_csv(logs_path+'/metrics_train.csv')

	print("Model Training Complete..")
	print("The Model Validation Metrics are ")
	
	print(eval_metrics)

	print('Training Completed Successfully')



