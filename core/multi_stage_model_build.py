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
from multi_head_train import Multi_Head_TrainModel
from training_viz import TrainViz
from metrics_eval import MetricsEval
#from model_train import TrainModel
from keras_lr_multiplier import LRMultiplier

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
	
	print('Parsing Multi-Stage System')

	max_stages=config.multi_stage_sensor_config['max_stages']
	eval_metric=config.multi_stage_sensor_config['eval_metric']
	eval_metric_threshold=config.multi_stage_sensor_config['eval_metric_threshold']
	inital_stage_list=config.multi_stage_sensor_config['inital_stage_list']

	multi_stage_sensor_params=config.multi_stage_sensor_construct

	print('Creating file Structure....')
	
	folder_name=part_type
	train_path='../trained_models/'+part_type+'/multi_station'
	multi_stage_path='../trained_models/'+part_type+'/multi_station'
	pathlib.Path(train_path).mkdir(parents=True, exist_ok=True)

	print('Initializing the Assembly System and Measurement System....')
	measurement_system=HexagonWlsScanner(data_type,application,system_noise,part_type,data_format)
	vrm_system=VRMSimulationModel(assembly_type,assembly_kccs,assembly_kpis,part_name,part_type,voxel_dim,voxel_channels,point_dim,aritifical_noise)
	get_data=GetTrainData();

	output_dimension=assembly_kccs
	
	def folder_struc(train_path):
		pathlib.Path(train_path).mkdir(parents=True, exist_ok=True) 

		model_path=train_path+'/model'
		pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
		
		logs_path=train_path+'/logs'
		pathlib.Path(logs_path).mkdir(parents=True, exist_ok=True)

		plots_path=train_path+'/plots'
		pathlib.Path(plots_path).mkdir(parents=True, exist_ok=True)

		deployment_path=train_path+'/deploy'
		pathlib.Path(deployment_path).mkdir(parents=True, exist_ok=True)

		return model_path,logs_path,plots_path

	#Objects of Measurement System, Assembly System, Get Inference Data
	print('Building Sensor Arcitecture and training models...')
	
	model_heads=0
	
	x_in=[]
	x_test=[]

	point_index=get_data.load_mapping_index(mapping_index)
	
	print('Importing output process parameter data..')

	y_out=get_data.data_import(kcc_files,kcc_folder)
	y_test=get_data.data_import(test_kcc_files,kcc_folder)

	#accuracy_metrics_df=pd.read_csv('metrics_data_study_100_.csv')
	metrics_list=['KCC_ID','MAE','MSE','RMSE','R2']
	
	metric_index=metrics_list.index(eval_metric)
	accuracy_state=np.zeros((len(multi_stage_sensor_params),3))
	
	for index,stage in enumerate(multi_stage_sensor_params):
		accuracy_state[index,0]=stage['stage_id']
		accuracy_state[index,2]=len(stage['process_param_ids'])

	print('Accuracy State Initlized...')

	print(accuracy_state)

	print('Running Deep Learning Integrated Sensor Optimization....')
	print('Limit for maximum number of sensor locations: ',max_stages)

	print('Metric used for diagnosis performance evaluation of Deep Learning Integrated Sensor Optimization: ',eval_metric)
	
	for i in range(max_stages):
		
		if(i==0):	
			
			if(len(inital_stage_list)>0):
				print('Building Model for inital given sensor system: ', inital_stage_list)
				model_heads=len(inital_stage_list)
				dl_model=Multi_Head_DLModel(model_type,model_heads,output_dimension)
				model=dl_model.multi_head_shared_standard_cnn_model_3d(voxel_dim,voxel_channels)

				print('Getting data from the data sources: ')

				for check_stage in inital_stage_list:
					for stage in multi_stage_sensor_params:
						if(stage['stage_id']==check_stage):
							
							print('Importing and Pre-Processing Train data for: ','station: ',stage['station_id'],'stage: ',stage['stage_id'])
							dataset=[]
							dataset.append(get_data.data_import(stage['data_files_x'],data_folder))
							dataset.append(get_data.data_import(stage['data_files_y'],data_folder))
							dataset.append(get_data.data_import(stage['data_files_z'],data_folder))
							
							input_conv_data, kcc_subset_dump,kpi_subset_dump=get_data.data_convert_voxel_mc(vrm_system,dataset,point_index)
							
							x_in.append(input_conv_data)

							print('Importing and Pre-Processing test data for: ','station: ',stage['station_id'],'stage: ',stage['stage_id'])
							dataset=[]
							dataset.append(get_data.data_import(stage['test_data_files_x'],data_folder))
							dataset.append(get_data.data_import(stage['test_data_files_y'],data_folder))
							dataset.append(get_data.data_import(stage['test_data_files_z'],data_folder))
							
							input_conv_data_test, kcc_subset_dump_test,kpi_subset_dump=get_data.data_convert_voxel_mc(vrm_system,dataset,point_index)
							x_test.append(input_conv_data_test)

				print('Total data sources: ',len(x_test))

				train_model=Multi_Head_TrainModel(batch_size,epocs,split_ratio)		
				model_path,logs_path,plots_path=folder_struc(train_path+'/run_'+str(i))
				trained_model,eval_metrics,accuracy_metrics_df=train_model.run_train_model(model,x_in,y_out,x_test,y_test,model_path,logs_path,plots_path,activate_tensorboard)

				sensor_list=inital_stage_list
				print('Process Parameter Accuarcy Metrics after adding intial sensor:')
				print(accuracy_metrics_df)

			if(len(inital_stage_list)==0):
				
				print('Adding Stage with maximum Process Parameters as inital sensor system (No selected initial sensor)')
				max_length=0
				
				for stage in multi_stage_sensor_params:
					
					if(len(stage['process_param_ids'])>=max_length):
						max_length=len(stage['process_param_ids'])
						process_params=stage['process_param_ids']
						stage_id=stage['stage_id']

				print('Selected Stage: ',stage_id)
				print('Stage contains params: ', process_params)
				inital_stage_list.append(stage_id)
				
				model_heads=len(inital_stage_list)

				dl_model=Multi_Head_DLModel(model_type,model_heads,output_dimension)
				model=dl_model.multi_head_shared_standard_cnn_model_3d(voxel_dim,voxel_channels)
				
				for stage in multi_stage_sensor_params:
					
					if(stage['stage_id']==inital_stage_list[0]):
							
							print('Importing and Pre-Processing Train data for: ','station: ',stage['station_id'],'stage: ',stage['stage_id'])
							dataset=[]
							dataset.append(get_data.data_import(stage['data_files_x'],data_folder))
							dataset.append(get_data.data_import(stage['data_files_y'],data_folder))
							dataset.append(get_data.data_import(stage['data_files_z'],data_folder))
							
							input_conv_data, kcc_subset_dump,kpi_subset_dump=get_data.data_convert_voxel_mc(vrm_system,dataset,point_index)
							
							x_in.append(input_conv_data)

							print('Importing and Pre-Processing test data for: ','station: ',stage['station_id'],'stage: ',stage['stage_id'])
							dataset=[]
							dataset.append(get_data.data_import(stage['test_data_files_x'],data_folder))
							dataset.append(get_data.data_import(stage['test_data_files_y'],data_folder))
							dataset.append(get_data.data_import(stage['test_data_files_z'],data_folder))
							
							input_conv_data_test, kcc_subset_dump_test,kpi_subset_dump=get_data.data_convert_voxel_mc(vrm_system,dataset,point_index)
							
							x_test.append(input_conv_data_test)
							
				print('Total data sources: ',len(x_test))
				train_model=Multi_Head_TrainModel(batch_size,epocs,split_ratio)		
				model_path,logs_path,plots_path=folder_struc(train_path+'/run_'+str(i))
				print('Resources at: ',train_path+'/run_'+str(i))
				trained_model,eval_metrics,accuracy_metrics_df=train_model.run_train_model(model,x_in,y_out,x_test,y_test,model_path,logs_path,plots_path,activate_tensorboard)
				accuracy_metrics_df.to_csv(logs_path+'/metrics_train.csv')
				
				sensor_list=inital_stage_list
				print('Process Parameter Accuarcy Metrics after adding intial sensor:')
				print(accuracy_metrics_df)
		
		if(i>0):
			
			print('Checking accuracy for all process parameters...')
			
			for index,stage in enumerate(multi_stage_sensor_params):
				process_params=stage['process_param_ids']
				stage_accuracy=(accuracy_metrics_df.iloc[process_params,metric_index]).values
				#print(stage_accuracy)
				accuracy_state[index,1]=np.mean(stage_accuracy)

			print('Current Accuracy state')
			#print(accuracy_state)
			sorted_accuracy_state=accuracy_state[accuracy_state[:,1].argsort()]
			print(sorted_accuracy_state)
			np.savetxt(logs_path+'/accuracy_state.csv', sorted_accuracy_state, delimiter=",")
			np.savetxt(logs_path+'/selected_sensor_locations.csv', sensor_list, delimiter=",")
			#stage_id=accuracy_state[:,1].argmax()
			
			append_flag=0
			
			for j in range(len(sorted_accuracy_state)):
				
				if(np.isnan(sorted_accuracy_state[j,1])):
					continue
				
				stage_id=int(sorted_accuracy_state[j,0])
				
				if(stage_id not in sensor_list):
					sensor_list.append(stage_id)
					append_flag=1
					print('Sensor Location added for additional data source: ',stage_id)

					for stage in multi_stage_sensor_params:
					
						if(stage['stage_id']==stage_id):
								
								print('Importing and Pre-Processing Train data for: ','station: ',stage['station_id'],'stage: ',stage['stage_id'])
								print('Stage contains process params: ', stage['process_param_ids'])
								dataset=[]
								dataset.append(get_data.data_import(stage['data_files_x'],data_folder))
								dataset.append(get_data.data_import(stage['data_files_y'],data_folder))
								dataset.append(get_data.data_import(stage['data_files_z'],data_folder))
								
								input_conv_data, kcc_subset_dump,kpi_subset_dump=get_data.data_convert_voxel_mc(vrm_system,dataset,point_index)
								
								x_in.append(input_conv_data)

								print('Importing and Pre-Processing test data for: ','station: ',stage['station_id'],'stage: ',stage['stage_id'])
								dataset=[]
								dataset.append(get_data.data_import(stage['test_data_files_x'],data_folder))
								dataset.append(get_data.data_import(stage['test_data_files_y'],data_folder))
								dataset.append(get_data.data_import(stage['test_data_files_z'],data_folder))
								
								input_conv_data_test, kcc_subset_dump_test,kpi_subset_dump=get_data.data_convert_voxel_mc(vrm_system,dataset,point_index)
								
								x_test.append(input_conv_data_test)
								

					print('Training Model using the addtional data source')
					break
			
			print('The Current Data Sources within the multi_stage system: ',sensor_list)


			if(append_flag==0):
				print('Data Sources Limit reached !')
				break

			if(append_flag==1):
				print('Running Model Training..')
				print('Total data sources: ',len(x_test))

				model_heads=len(sensor_list)

				dl_model=Multi_Head_DLModel(model_type,model_heads,output_dimension)
				model=dl_model.multi_head_shared_standard_cnn_model_3d(voxel_dim,voxel_channels)

				train_model=Multi_Head_TrainModel(batch_size,epocs,split_ratio)		
				
				model_path,logs_path,plots_path=folder_struc(train_path+'/run_'+str(i))
				print('Resources at: ',train_path+'/run_'+str(i))
				trained_model,eval_metrics,accuracy_metrics_df=train_model.run_train_model(model,x_in,y_out,x_test,y_test,model_path,logs_path,plots_path,activate_tensorboard)
				
				print('Process Parameter Accuarcy Metrics after adding sensor:')
				print(accuracy_metrics_df)
				accuracy_metrics_df.to_csv(logs_path+'/metrics_train.csv')

		np.savetxt(multi_stage_path+'/selected_sensor_locations.csv', sensor_list, delimiter=",")
		print('Selected Sensor List stored at: ', multi_stage_path)

		current_eval_metric_array=(accuracy_metrics_df.iloc[:,metric_index]).values
		current_eval_metric=np.mean(current_eval_metric_array)
		
		print('Current Accuracy: ', current_eval_metric)
		print('Reuired Accuracy: ',eval_metric_threshold)
		
		if(current_eval_metric>=eval_metric_threshold):
			print('Required Accuracy Reached...')
			break
