""" The data study file is used to study the training data requirement for convergence, the file trains the model incrementally (training size is increased) based on the parameters specified in the modelconfig_train.py file and simultaneously
tests the trained model at each step to estimate the optimum number of training samples required. The file also generates combined html based plots (using plotly and cufflinks), trained model, accuracy metrics and training plots for each iteration
within the file structure 
"""


import os
import sys
current_path=os.path.dirname(__file__)
parentdir = os.path.dirname(current_path)

os.environ["CUDA_VISIBLE_DEVICES"]="0" # Nvidia Quadro GV100
#Adding Path to various Modules
sys.path.append("../core")
sys.path.append("../visualization")
sys.path.append("../utilities")
sys.path.append("../datasets")
sys.path.append("../trained_models")
sys.path.append("../config")
sys.path.append("../transfer_learning")
#path_var=os.path.join(os.path.dirname(__file__),"../utilities")
#sys.path.append(path_var)
#sys.path.insert(0,parentdir) 

#Importing Required Modules
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from tqdm import tqdm
import plotly as py
import plotly.graph_objects as go
import cufflinks as cf
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
from model_deployment import DeployModel
from metrics_eval import MetricsEval
#from tl_core import TransferLearning
	
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

	#Get Out of Sample data for Testing
	test_file_names_x=config.assembly_system['test_data_files_x']
	test_file_names_y=config.assembly_system['test_data_files_y']
	test_file_names_z=config.assembly_system['test_data_files_z']
	test_kcc_files=config.assembly_system['test_kcc_files']

	print('Parsing from Training Config File')

	model_type=cftrain.model_parameters['model_type']
	output_type=cftrain.model_parameters['output_type']
	optimizer=cftrain.model_parameters['optimizer']
	loss_func=cftrain.model_parameters['loss_func']
	regularizer_coeff=cftrain.model_parameters['regularizer_coeff']
	activate_tensorboard=cftrain.model_parameters['activate_tensorboard']

	batch_size=cftrain.data_study_params['batch_size']
	epocs=cftrain.data_study_params['epocs']
	split_ratio=cftrain.data_study_params['split_ratio']
	min_train_samples=cftrain.data_study_params['min_train_samples']
	max_train_samples=cftrain.data_study_params['max_train_samples']
	train_increment=cftrain.data_study_params['train_increment']
	tl_flag=cftrain.data_study_params['tl_flag']

	tl_type=cftrain.transfer_learning['tl_type']
	tl_base=cftrain.transfer_learning['tl_base']
	tl_app=cftrain.transfer_learning['tl_app']
	conv_layer_m=cftrain.transfer_learning['conv_layer_m']
	dense_layer_m=cftrain.transfer_learning['dense_layer_m']

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
	deploy_model=DeployModel();
	metrics_eval=MetricsEval();

	print('Importing and Preprocessing Cloud-of-Point Data')
	dataset=[]
	dataset.append(get_data.data_import(file_names_x,data_folder))
	dataset.append(get_data.data_import(file_names_y,data_folder))
	dataset.append(get_data.data_import(file_names_z,data_folder))

	dataset_test=[]
	dataset_test.append(get_data.data_import(test_file_names_x,data_folder))
	dataset_test.append(get_data.data_import(test_file_names_y,data_folder))
	dataset_test.append(get_data.data_import(test_file_names_z,data_folder))
	point_index=get_data.load_mapping_index(mapping_index)

	kcc_dataset=get_data.data_import(kcc_files,kcc_folder)
	kcc_dataset_test=get_data.data_import(test_kcc_files,kcc_folder)

	point_index=get_data.load_mapping_index(mapping_index)

	kcc_dataset_test=get_data.data_import(test_kcc_files,kcc_folder)
	
	input_conv_data, kcc_subset_dump,kpi_subset_dump=get_data.data_convert_voxel_mc(vrm_system,dataset,point_index,kcc_dataset)
	input_conv_data_test, kcc_subset_dump_test,kpi_subset_dump_test=get_data.data_convert_voxel_mc(vrm_system,dataset_test,point_index,kcc_dataset_test)
	#print(input_conv_data.shape,kcc_subset_dump.shape)
		
	if(activate_tensorboard==1):
		tensorboard_str='tensorboard' + '--logdir '+logs_path
		print('Visualize at Tensorboard using ', tensorboard_str)
	
	
	output_dimension=assembly_kccs
	max_dim=min(max_train_samples,len(input_conv_data))
	no_of_splits=int((max_dim-min_train_samples)/train_increment)

	eval_metrics_type= ["Mean Absolute Error","Mean Squared Error","Root Mean Squared Error","R Squared"]

	kcc_id=[]

	for i in range(assembly_kccs):  
		kcc_name="KCC_"+str(i+1)
		kcc_id.append(kcc_name)

	datastudy_output=np.zeros((no_of_splits,(assembly_kccs+1)*len(eval_metrics_type)+1))
	datastudy_output_test=np.zeros((no_of_splits,(assembly_kccs+1)*len(eval_metrics_type)+1))

	train_dim=min_train_samples



	for i in tqdm(range(no_of_splits)):
		
		run_id=i
		
		if(train_dim>max_dim):
			train_dim=max_dim
		
		print('Building 3D CNN model')

		
		if(tl_flag==0):
			dl_model=DLModel(model_type,output_dimension,optimizer,loss_func,regularizer_coeff,output_type)
			model=dl_model.resnet_3d_cnn(voxel_dim,voxel_channels)

		if(tl_flag==1):
			transfer_learning=TransferLearning(tl_type,tl_base,tl_app,model_type,assembly_kccs,optimizer,loss_func,regularizer_coeff,output_type)
			base_model=transfer_learning.get_trained_model()
			
			print(base_model.summary())
			
			#plot_model(base_model, to_file='model.png')

			transfer_model=transfer_learning.build_transfer_model(base_model)

			if(tl_type=='full_fine_tune'):
				model=transfer_learning.full_fine_tune(transfer_model)

			if(tl_type=='variable_lr'):
				model=transfer_learning.set_variable_learning_rates(transfer_model,conv_layer_m,dense_layer_m)

			if(tl_type=='feature_extractor'):
				model=transfer_learning.set_fixed_train_params(transfer_model)

		print(model.summary())

		print("Conducting data study study on :",train_dim, " samples")
		input_conv_subset=input_conv_data[0:train_dim,:,:,:,:]
		kcc_subset=kcc_subset_dump[0:train_dim,:]

		train_model=TrainModel(batch_size,epocs,split_ratio)
		trained_model,eval_metrics,accuracy_metrics_df=train_model.run_train_model(model,input_conv_subset,kcc_subset,model_path,logs_path,plots_path,activate_tensorboard,run_id)

		datastudy_output[i,0]=train_dim
		datastudy_output[i,1:assembly_kccs+1]=eval_metrics["Mean Absolute Error"]
		datastudy_output[i,assembly_kccs+1:(2*assembly_kccs)+1]=eval_metrics["Mean Squared Error"]
		datastudy_output[i,(2*assembly_kccs)+1:(3*assembly_kccs)+1]=eval_metrics["Root Mean Squared Error"]
		datastudy_output[i,(3*assembly_kccs)+1:(4*assembly_kccs)+1]=eval_metrics["R Squared"]

		file_name='metrics_data_study_'+str(train_dim)+'_.csv'
		accuracy_metrics_df.to_csv(logs_path+'/'+file_name)
		print("Model Training Complete on samples :",train_dim)
		print("The Model Validation Metrics are ")
		print(eval_metrics)

		#Inferring on test dataset
		model_test_path=train_path+'/model'+'/trained_model_'+str(run_id)+'.h5'
		inference_model=deploy_model.get_model(model_test_path)
		y_pred=deploy_model.model_inference(input_conv_data_test,inference_model);
		eval_metrics_test,accuracy_metrics_df_test=metrics_eval.metrics_eval_base(y_pred,kcc_subset_dump_test,logs_path,run_id)

		datastudy_output_test[i,0]=train_dim
		datastudy_output_test[i,1:assembly_kccs+1]=eval_metrics_test["Mean Absolute Error"]
		datastudy_output_test[i,assembly_kccs+1:(2*assembly_kccs)+1]=eval_metrics_test["Mean Squared Error"]
		datastudy_output_test[i,(2*assembly_kccs)+1:(3*assembly_kccs)+1]=eval_metrics_test["Root Mean Squared Error"]
		datastudy_output_test[i,(3*assembly_kccs)+1:(4*assembly_kccs)+1]=eval_metrics_test["R Squared"]

		file_name='test_metrics_data_study_'+str(train_dim)+'_.csv'
		accuracy_metrics_df_test.to_csv(logs_path+'/'+file_name)
		print("Model Testing Complete on samples :",train_dim)
		print("The Model Test Metrics are ")
		print(eval_metrics_test)
		K.clear_session()
		train_dim=train_dim+train_increment

	for i in range(len(eval_metrics_type)):
		datastudy_output[:,(4*assembly_kccs)+i+1]=np.mean(datastudy_output[:,(i*assembly_kccs)+1:((i+1)*assembly_kccs)+1],axis=1)
		datastudy_output_test[:,(4*assembly_kccs)+i+1]=np.mean(datastudy_output_test[:,(i*assembly_kccs)+1:((i+1)*assembly_kccs)+1],axis=1)

	#Gen Column Names

	col_names=['Training_Samples']
	for metric in eval_metrics_type:
		for kcc in kcc_id:
			col_names.append(str(metric)+'_'+str(kcc))

	for metric in eval_metrics_type:
	    col_names.append(str(metric)+"_avg")

	ds_output_df=pd.DataFrame(data=datastudy_output,columns=col_names)
	ds_output_df.to_csv(logs_path+'/'+'datastudy_output.csv')

	ds_output_df_test=pd.DataFrame(data=datastudy_output_test,columns=col_names)
	ds_output_df_test.to_csv(logs_path+'/'+'datastudy_output_test.csv')
	
	print('Data Study Completed Successfully')

	print('Plotting Data Study Validation Results: ')
	fig = ds_output_df.iplot(x='Training_Samples',asFigure=True)
	py.offline.plot(fig,filename=logs_path+'/'+"data_study_plot.html")

	print('Plotting Data Study Validation Results: ')
	fig_test = ds_output_df_test.iplot(x='Training_Samples',asFigure=True)
	py.offline.plot(fig_test,filename=logs_path+'/'+"data_study_plot_test.html")
