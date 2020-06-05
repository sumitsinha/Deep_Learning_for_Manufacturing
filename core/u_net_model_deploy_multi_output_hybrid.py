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
from tensorflow.keras import backend as K
K.clear_session()

#Importing Config files
import assembly_config as config
import model_config as cftrain
import voxel_config as vc
import hybrid_utils as hy_util

#Importing required modules from the package
from measurement_system import HexagonWlsScanner
from assembly_system import VRMSimulationModel
from wls400a_system import GetInferenceData
from data_import import GetTrainData
from encode_decode_model import Encode_Decode_Model
from training_viz import TrainViz
from metrics_eval import MetricsEval
from keras_lr_multiplier import LRMultiplier
from point_cloud_construction import GetPointCloud

class Unet_DeployModel:
	"""Train Model Class, the initialization parameters are parsed from modelconfig_train.py file
		
		:param batch_size: mini batch size while training the model 
		:type batch_size: int (required)

		:param epochs: no of epochs to conduct training
		:type epochs: int (required)

		:param split_ratio: train and validation split for the model
		:type assembly_system: float (required)

		The class contains run_train_model method
	"""	
			
	def unet_run_model(self,model,X_in_test,model_path,logs_path,plots_path,test_result=0,Y_out_test_list=0,activate_tensorboard=0,run_id=0,tl_type='full_fine_tune'):
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
		from tensorflow.keras.models import load_model
		import tensorflow.keras.backend as K 
		#model_file_path=model_path+'/unet_trained_model_'+str(run_id)+'.h5'
		model_file_path=model_path+'/unet_trained_model_attention_hybrid_'+str(run_id)
		
		#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='C:\\Users\\sinha_s\\Desktop\\dlmfg_package\\dlmfg\\trained_models\\inner_rf_assembly\\logs',histogram_freq=1)
		
		#inference_model=load_model(model_file_path,custom_objects={'mse_scaled': mse_scaled} )
		model.load_weights(model_file_path)
		print("Trained Model Weights loaded successfully")
		print("Conducting Inference...")
		model_outputs=model.predict(X_in_test)
		y_pred_regression=model_outputs[0]
		y_pred_classification=model_outputs[1]
		print("Inference Completed !")
		
		if(test_result==1):
			metrics_eval=MetricsEval();
			eval_metrics_reg,accuracy_metrics_df_reg=metrics_eval.metrics_eval_base(y_pred_regression,Y_out_test_list[0],logs_path)
			eval_metrics_cla,accuracy_metrics_df_cla=metrics_eval.metrics_eval_classification(y_pred_classification,Y_out_test_list[1],logs_path)
		
			
			#y_cop_pred_flat=y_cop_pred.flatten()
			#y_cop_test_flat=y_cop_test.flatten()

			#combined_array=np.stack([y_cop_test_flat,y_cop_pred_flat],axis=1)
			#filtered_array=combined_array[np.where(combined_array[:,0] >= 0.05)]
			#y_cop_test_vector=filtered_array[:,0:1]
			#y_cop_pred_vector=filtered_array[:,1:2]

			eval_metrics_cop_list=[]
			accuracy_metrics_df_cop_list=[]
			
			for i in range(2,len(model_outputs)):
				y_cop_pred=model_outputs[i]
				y_cop_test=Y_out_test_list[i]
				y_cop_pred_vector=np.reshape(y_cop_pred,(y_cop_pred.shape[0],-1))
				y_cop_test_vector=np.reshape(y_cop_test,(y_cop_test.shape[0],-1))
				y_cop_pred_vector=y_cop_pred_vector.T
				y_cop_test_vector=y_cop_test_vector.T
				print(y_cop_pred_vector.shape)
				#y_cop_test_flat=y_cop_test.flatten()
				
				eval_metrics_cop,accuracy_metrics_df_cop=metrics_eval.metrics_eval_cop(y_cop_pred_vector,y_cop_test_vector,logs_path)
				eval_metrics_cop_list.append(eval_metrics_cop)
				accuracy_metrics_df_cop_list.append(accuracy_metrics_df_cop)
			
			return model_outputs,model,accuracy_metrics_df_reg,accuracy_metrics_df_cla,accuracy_metrics_df_cop_list
		
		return model_outputs,model


def plot_decode_cop_voxel(base_cop,plot_file_name):
	
	import plotly.graph_objects as go
	import plotly as py
	import plotly.express as px
	X, Y, Z = np.mgrid[0:len(base_cop), 0:len(base_cop), 0:len(base_cop)]
		#input_conv_data[0,:,:,:,0]=0.2
	values_cop = base_cop.flatten()

	from sklearn.preprocessing import MinMaxScaler
	scaler = MinMaxScaler()
	scaled_values=scaler.fit_transform(values_cop.reshape(-1, 1))
	trace1=go.Volume(
			x=X.flatten(),
			y=Y.flatten(),
			z=Z.flatten(),
			value=scaled_values[:,0],
			isomin=0,
			isomax=1,
			opacity=0.1, # needs to be small to see through all surfaces
			surface_count=17, # needs to be a large number for good volume rendering
			colorscale='Greens'
	)

	layout = go.Layout(
			margin=dict(
				l=0,
				r=0,
				b=0,
				t=0
			)
		)
		
	data=[trace1]

	fig = go.Figure(data=data,layout=layout)
	py.offline.plot(fig, filename=plot_file_name)

def plot_decode_cop_dev(nominal_cop,dev_vector,plot_file_name):
	
	import plotly.graph_objects as go
	import plotly as py
	import plotly.express as px

	#input_conv_data[0,:,:,:,0]=0.2
	values_cop = dev_vector.flatten()


	from sklearn.preprocessing import MinMaxScaler
	scaler = MinMaxScaler()
	scaled_values=scaler.fit_transform(values_cop.reshape(-1, 1))
	trace1=go.Scatter3d(
			x=nominal_cop[:,0],
			y=nominal_cop[:,1],
			z=nominal_cop[:,2],
			#surfacecolor=dev_vector,
			hoverinfo="text",
			hovertext=dev_vector,
			mode='markers',
			marker=dict(
				showscale=True,
				size=12,
				#color=scaled_values[:,0],   
				color=dev_vector,        # set color to an array/list of desired values
				colorscale='Viridis',   # choose a colorscale
				opacity=0.6
				)
   
	)

	layout = go.Layout(
			margin=dict(
				l=0,
				r=0,
				b=0,
				t=0
			)
		)
		
	data=[trace1]

	fig = go.Figure(data=data,layout=layout)
	#print(plot_file_name)
	py.offline.plot(fig, filename=plot_file_name)

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

	system_noise=config.assembly_system['system_noise']
	aritifical_noise=config.assembly_system['aritifical_noise']
	data_folder=config.assembly_system['data_folder']
	kcc_folder=config.assembly_system['kcc_folder']
	kcc_files=config.assembly_system['kcc_files']
	test_kcc_files=config.assembly_system['test_kcc_files']

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

	train_path=train_path+'/unet_model_multi_output'
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
	get_data=GetTrainData()

	kcc_sublist=cftrain.encode_decode_params['kcc_sublist']
	output_heads=cftrain.encode_decode_params['output_heads']
	encode_decode_multi_output_construct=config.encode_decode_multi_output_construct
	
	if(output_heads==len(encode_decode_multi_output_construct)):
		print("Valid Output Stages and heads")
	else:
		print("Inconsistent model setting")

	#Check for KCC sub-listing
	if(kcc_sublist!=0):
		output_dimension=len(kcc_sublist)
	else:
		output_dimension=assembly_kccs

	#print(input_conv_data.shape,kcc_subset_dump.shape)
	print('Building Unet Model')

	output_dimension=assembly_kccs
	input_size=(voxel_dim,voxel_dim,voxel_dim,voxel_channels)

	model_depth=cftrain.encode_decode_params['model_depth']
	inital_filter_dim=cftrain.encode_decode_params['inital_filter_dim']

	dl_model_unet=Encode_Decode_Model(output_dimension)
	model=dl_model_unet.encode_decode_3d_multi_output_attention_hybrid(inital_filter_dim,model_depth,input_size,categorical_kccs,output_heads,voxel_channels)

	print(model.summary())
	#sys.exit()
	
	test_input_file_names_x=config.encode_decode_construct['input_test_data_files_x']
	test_input_file_names_y=config.encode_decode_construct['input_test_data_files_y']
	test_input_file_names_z=config.encode_decode_construct['input_test_data_files_z']

	if(activate_tensorboard==1):
		tensorboard_str='tensorboard' + '--logdir '+logs_path
		print('Visualize at Tensorboard using ', tensorboard_str)
	
	print('Importing and Preprocessing Cloud-of-Point Data')
	
	point_index=get_data.load_mapping_index(mapping_index)
	
	get_point_cloud=GetPointCloud()

	cop_file_name=vc.voxel_parameters['nominal_cop_filename']
	cop_file_path='../resources/nominal_cop_files/'+cop_file_name
	#Read cop from csv file
	print('Importing Nominal COP')
	nominal_cop=vrm_system.get_nominal_cop(cop_file_path)

	test_input_dataset=[]
	test_input_dataset.append(get_data.data_import(test_input_file_names_x,data_folder))
	test_input_dataset.append(get_data.data_import(test_input_file_names_y,data_folder))
	test_input_dataset.append(get_data.data_import(test_input_file_names_z,data_folder))

	#kcc_dataset=get_data.data_import(kcc_files,kcc_folder)
	test_input_conv_data, test_kcc_subset_dump_dummy,test_kpi_subset_dump=get_data.data_convert_voxel_mc(vrm_system,test_input_dataset,point_index)
	
	#Test output files
	deploy_output=1
	
	if(deploy_output==1):
		
		test_kcc_dataset=get_data.data_import(test_kcc_files,kcc_folder)
		
		if(kcc_sublist!=0):
			print("Sub-setting Process Parameters: ",kcc_sublist)
			test_kcc_dataset=test_kcc_dataset[:,kcc_sublist]
		else:
			print("Using all Process Parameters")
		
		Y_out_test_list=[None,None]
		#Y_out_test_list.append(test_kcc_subset_dump)
		
		for encode_decode_construct in encode_decode_multi_output_construct:
		#importing file names for model output
			print("Importing output data for stage: ",encode_decode_construct)
			
			test_output_file_names_x=encode_decode_construct['output_test_data_files_x']
			test_output_file_names_y=encode_decode_construct['output_test_data_files_y']
			test_output_file_names_z=encode_decode_construct['output_test_data_files_z']
			test_output_dataset=[]
			test_output_dataset.append(get_data.data_import(test_output_file_names_x,data_folder))
			test_output_dataset.append(get_data.data_import(test_output_file_names_y,data_folder))
			test_output_dataset.append(get_data.data_import(test_output_file_names_z,data_folder))

			test_output_conv_data, test_kcc_subset_dump,test_kpi_subset_dump=get_data.data_convert_voxel_mc(vrm_system,test_output_dataset,point_index,test_kcc_dataset)
			kcc_regression_test,kcc_classification_test=hy_util.split_kcc(test_kcc_subset_dump)
			Y_out_test_list[0]=kcc_regression_test
			Y_out_test_list[1]=kcc_classification_test

			Y_out_test_list.append(test_output_conv_data)

	#Pre-processing to point cloud data
	
	unet_deploy_model=Unet_DeployModel()

	if(deploy_output==1):
		model_outputs,model,accuracy_metrics_df_reg,accuracy_metrics_df_claaccuracy_metrics_df_cop_list=unet_deploy_model.unet_run_model(model,test_input_conv_data,model_path,logs_path,plots_path,deploy_output,Y_out_test_list)
		
		accuracy_metrics_df.to_csv(logs_path+'/metrics_test_KCC.csv')
		
		print("Model Deployment Complete")
		
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

		index=2
		
		for accuracy_metrics_df_cop in accuracy_metrics_df_cop_list:
			
			accuracy_metrics_df_cop.to_csv(logs_path+'/metrics_test_cop_'+str(index)+'.csv')
			print("The Model Segmentation Validation Metrics are ")
			print(accuracy_metrics_df_cop.mean())
			accuracy_metrics_df_cop.mean().to_csv(logs_path+'/metrics_test_cop_summary_'+str(index)+'.csv')
			
			print("Plotting Cloud-of-Point for comparison")

			part_id=45

			y_cop_pred=model_outputs[index]
			y_cop_actual=Y_out_test_list[index]
			#y_cop_pred_plot=y_cop_pred[part_id,:,:,:,:]
			#y_cop_actual_plot=test_input_conv_data[part_id,:,:,:,:]

			dev_actual=get_point_cloud.getcopdev(y_cop_actual[part_id,:,:,:,:],point_index,nominal_cop)
			dev_pred=get_point_cloud.getcopdev(y_cop_pred[part_id,:,:,:,:],point_index,nominal_cop)
			
			filenamestr_pred=["/pred_plot_x"+str(index)+".html","/pred_plot_y"+str(index)+".html","/pred_plot_z"+str(index)+".html"]
			filenamestr_actual=["/actual_plot_x"+str(index)+".html","/actual_plot_y"+str(index)+".html","/actual_plot_z"+str(index)+".html"]
			
			print("Plotting All components for sample id: ",part_id)	
			
			for i in range(3):
				#pass
				#pred Plot
				plot_decode_cop_dev(nominal_cop,dev_pred[:,i],plot_file_name=deployment_path+filenamestr_pred[i])
				plot_decode_cop_dev(nominal_cop,dev_actual[:,i],plot_file_name=deployment_path+filenamestr_actual[i])

			index=index+1

	if(deploy_output==0):
		model_outputs,model=unet_deploy_model.unet_run_model(model,test_input_conv_data,model_path,logs_path,plots_path,deploy_output)

		print('Predicted KCCs')
		print(model_outputs[0],model_outputs[1])