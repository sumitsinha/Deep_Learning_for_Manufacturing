""" Contains core classes and methods for initializing a Transfer Learning Class and running transfer learning using a pre-trained model and a small training dataset, the inputs are provided in assemblyconfig file in utilities"""

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


#Importing Required Modules
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras_lr_multiplier import LRMultiplier
from keras.utils import plot_model
K.clear_session()

#Importing Config files
import assembly_config as config
import model_config as cftrain

#Importing required modules from the package
from measurement_system import HexagonWlsScanner
from assembly_system import VRMSimulationModel
from wls400a_system import GetInferenceData
from data_import import GetTrainData
from core_model import DLModel
from training_viz import TrainViz
from metrics_eval import MetricsEval
from model_train import TrainModel

class TransferLearning:
	"""Transfer Learning Class

		:param tl_type: Type of transfer learning to be done, full fine-tune, partial fine-tune, feature extraction
		:type tl_type: str (required)

		:param tl_base: The base model to be used for model
		:type tl_base: str (required)

		:param tl_app: The application for transfer learning
		:type tl_app: str (required)

		:param model_type: The type of model, regression or classification
		:type model_type: str (required)

		:param output_dimension: The number of KCCs for the case study to which the pre-trained model is to be transfered, to be used to reinitialize the last layer
		:type output_dimension: int (required)

		:param optimizer: The optimizer to be used for model training (https://keras.io/optimizers/)
		:type optimizer: keras.optimizer (required)

		:param loss_function: The loss function to be used for model training (https://keras.io/losses/)
		:type loss_function: keras.losses (required)

		:param regularizer_coeff: The regularization coefficient for L2 norm regularization of the fully connected layer (https://keras.io/regularizers/)
		:type regularizer_coeff: float (required)

		:param output_type: The type of model, regression or classification
		:type output_type: str (required)

	"""
	def __init__(self, tl_type,tl_base,tl_app,model_type,output_dimension,optimizer,loss_function,regularizer_coeff,output_type):
		self.tl_type=tl_type
		self.tl_base=tl_base
		self.tl_app=tl_app
		self.output_dimension=output_dimension
		self.model_type=model_type
		self.optimizer=optimizer
		self.loss_function=loss_function
		self.regularizer_coeff=regularizer_coeff
		self.output_type=output_type
	
	def get_trained_model(self):
		"""Imports the pre-trained model based on the object initialization, currently supports Keras modelname.h5 format (refer https://keras.io/models/model/ for more information on keras model)

			:returns: Pre-trained model with weights
			:rtype: keras.model
		"""
		def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
		 
			return K.mean(2. * (K.sum(y_true * y_pred,
									  axis=axis) + smooth/2)/(K.sum(y_true,
																	axis=axis) + K.sum(y_pred,
																					   axis=axis) + smooth))

		def weighted_dice_coefficient_loss(y_true, y_pred):
			return -weighted_dice_coefficient(y_true, y_pred)

		model_path='../pre_trained_models/deterministic_models/'+self.tl_base
		
		if(self.tl_base=='unet_3d.h5'):
			base_model=load_model(model_path,custom_objects={'InstanceNormalization': InstanceNormalization,'weighted_dice_coefficient_loss':weighted_dice_coefficient_loss})
		else:
			base_model=load_model(model_path)
		
		return base_model

	def build_transfer_model(self,model):
		"""The build_transfer_function takes the pre-trained model removes the final layer and adds another layer based on the new case study parameters, which is trained on a small dataset obtained from the new case study

			:param model: keras model with preset parameters
			:type model: keras.model (required)

			:returns: Updated model with new final layer
			:rtype: keras.model
		"""
		from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Dropout
		from keras.models import Model
		
		if(self.output_type=="regression"):
			final_layer_avt='linear'

		if(self.output_type=="classification"):
			final_layer_avt='softmax'

		model.layers.pop()
		model.layers.pop()
		x = model.layers[-2].output
		
		x = Dense(self.output_dimension, activation=final_layer_avt, name='new_predictions')(x)

		transfer_model = Model(input=model.input,output=x)	
		
		return transfer_model
		
	def set_fixed_train_params(self,model):
		"""The set_fixed_params function is used to freeze the weights of the convolution layer, hence the initial part of the network is to be used only as a feature extractor

			:param model: keras model with preset parameters
			:type model: keras.model (required)

			:returns: Updated model with non trainable convolution layers
			:rtype: keras.model
		"""
		for layer in model.layers:
			if('conv'in layer.name):	
				layer.trainable = False

		model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['mae'])

		return model


	def full_fine_tune(self,model):

		model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['mae'])

		return model

	def set_variable_learning_rates(self,model,conv_layer_m,dense_layer_m):
		"""The set_fixed_params function is used to freeze the weights of the convolution layer if the initial part of the network is to be used only as a feature extractor

			:param model: keras model with preset parameters
			:type model: keras.model (required)

			:param conv_layer_m: Learning rate multiplier for convolution layer
			:type conv_layer_m: float(required)

			:param dense_layer_m: Learning rate multiplier for dense layer
			:type dense_layer_m: float(required)

			:returns: Updated model with variable learning rates
			:rtype: keras.model
		"""
		lr_dict={}

		for layer in model.layers:
			if('conv'in layer.name):	
				lr_dict.update({layer.name : conv_layer_m})
			if('dense' in layer.name):
				lr_dict.update({layer.name : dense_layer_m})

		key='conv3d_1_input'
		
		if key in lr_dict:
			del lr_dict[key]

		#print(lr_dict)

		variable_optimizer=LRMultiplier(self.optimizer,lr_dict)

		model.compile(loss=self.loss_function, optimizer=variable_optimizer, metrics=['mae'])
		return model


	def tl_mode3(self,model):
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

	tl_type=cftrain.transfer_learning['tl_type']
	tl_base=cftrain.transfer_learning['tl_base']
	tl_app=cftrain.transfer_learning['tl_app']
	conv_layer_m=cftrain.transfer_learning['conv_layer_m']
	dense_layer_m=cftrain.transfer_learning['dense_layer_m']

	print('Creating file Structure....')
	
	folder_name=part_type
	train_path='../trained_models/'+part_type
	pathlib.Path(train_path).mkdir(parents=True, exist_ok=True) 

	tl_path=train_path+'/transfer_learning'
	pathlib.Path(tl_path).mkdir(parents=True, exist_ok=True)
	
	model_path=tl_path+'/model'
	pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)

	logs_path=tl_path+'/logs'
	pathlib.Path(logs_path).mkdir(parents=True, exist_ok=True)

	plots_path=tl_path+'/plots'
	pathlib.Path(plots_path).mkdir(parents=True, exist_ok=True)

	deployment_path=tl_path+'/deploy'
	pathlib.Path(deployment_path).mkdir(parents=True, exist_ok=True)

	#Objects of Measurement System, Assembly System, Get Inference Data
	print('Initializing the Assembly System and Measurement System....')
	
	measurement_system=HexagonWlsScanner(data_type,application,system_noise,part_type,data_format)
	vrm_system=VRMSimulationModel(assembly_type,assembly_kccs,assembly_kpis,part_name,part_type,voxel_dim,voxel_channels,point_dim,aritifical_noise)
	
	get_data=GetTrainData();
	point_index=get_data.load_mapping_index(mapping_index)

	print('Training 3D CNN model')
	
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
	
	transfer_learning=TransferLearning(tl_type,tl_base,tl_app,model_type,assembly_kccs,optimizer,loss_func,regularizer_coeff,output_type)
	base_model=transfer_learning.get_trained_model()
	
	print(base_model.summary())
	
	#plot_model(base_model, to_file='model.png')

	transfer_model=transfer_learning.build_transfer_model(base_model)

	if(tl_type=='full_fine_tune'):
		model=transfer_model

	if(tl_type=='variable_lr'):
		model=transfer_learning.set_variable_learning_rates(transfer_model,conv_layer_m,dense_layer_m)

	if(tl_type=='feature_extractor'):
		model=transfer_learning.set_fixed_train_params(transfer_model)

	
	train_model=TrainModel(batch_size,epocs,split_ratio)
	trained_model,eval_metrics,accuracy_metrics_df=train_model.run_train_model(model,input_conv_data,kcc_subset_dump,model_path,logs_path,plots_path,activate_tensorboard,tl_type=tl_type)

	accuracy_metrics_df.to_csv(logs_path+'/tl_metrics.csv')
	print("Transfer Learning Based Model Training Complete..")
	print("The Model Validation Metrics are ")
	print(eval_metrics)

