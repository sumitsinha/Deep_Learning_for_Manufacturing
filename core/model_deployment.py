""" The model deploy file is used to leverage a trained model to perform inference on unknown set of node deviations.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

import numpy as np
import pandas as pd
import tensorflow as tf
import csv
import logging
tf.get_logger().setLevel(logging.ERROR)

from tensorflow.keras.models import load_model


#Importing Config files
import assembly_config as config
import model_config as cftrain
import measurement_config as mscofig

#Importing required modules from the package
from measurement_system import HexagonWlsScanner
from assembly_system import VRMSimulationModel
from assembly_system import PartType
from wls400a_system import GetInferenceData
from metrics_eval import MetricsEval
from data_import import GetTrainData
#from cam_viz import CamViz
#from cop_viz import CopViz

class DeployModel:
	"""The Deploy Model class is used to import a trained model and use it to infer on unknown data

	"""
	def get_model(self,model_path):
		"""get_model method is is used to retrieve the trained model from a given path
				
				:param model_path: Path to the trained model, ideally it should be same as the train model path output
				:type model_path: str (required)
		"""
		from tensorflow.keras.models import load_model
		try:
			inference_model=load_model(model_path)
			print('Deep Learning Model found and loaded')
		except AssertionError as error:
			print(error)
			print('Model not found at this path ',model_path, ' Update path in config file if required')

		return inference_model

	def model_inference(self,inference_data,inference_model,deploy_path,print_result=0,plot_result=0,get_cam_data=0,append_result=0):
		"""model_inference method is used to infer from unknown sample(s) using the trained model 
				
				:param inference_data: Unknown dataset having same structure as the train dataset
				:type inference_data: numpy.array [samples*voxel_dim*voxel_dim*voxel_dim*deviation_channels] (required) (required)

				:param inference_model: Trained model
				:type inference_model: keras.model (required)
				
				:param print_result: Flag to indicate if the result needs to be printed, 0 by default, change to 1 in case the results need to be printed on the console
				:type print_result: int

		"""		
		result=inference_model.predict(inference_data)
		description="The Process Parameters variations are inferred from the obtained measurement data and the trained CNN based model"
		print('The model estimates are: ')
		rounded_result=np.round(result,2)
		
		if(print_result==1):
			print(rounded_result)
		
		if(append_result==1):
			with open ("user_preds.csv",'a',newline='') as filedata:
				#fieldnames = ['kcc1','kcc2','kcc3','kcc4','kcc5','kcc6']                            
				writer = csv.writer(filedata, delimiter=',')
				writer.writerow(rounded_result[0,:].tolist())
				#writer.writerow(dict(zip(fieldnames, rounded_result[0,:].tolist()))) 
				#filedata.write(rounded_result[0,:].tolist())
		
		if(plot_result==1):
			print("Plotting Results in HTML...")
			import plotly.graph_objects as go
			import plotly as py
			result_str = ["%.2f" % number for number in rounded_result[0,:]]

			kcc_str=[]
			for i in range(rounded_result.shape[1]):
				kcc_str.append("X("+str(i)+"): ")
			#kcc_str=["X(1): ","X(2): ", "X(3): ", "X(4): ", "X(5): ", "X(6): "]	
			
			display_str=np.core.defchararray.add(kcc_str, result_str)	
			print(display_str)
			fig = go.Figure(data=go.Scatter(y=rounded_result[0,:], marker=dict(
			size=30,color=100), mode='markers+text',text=display_str,x=kcc_str))
			fig.update_traces( textfont_size=20,textposition='top center')
			fig.update_layout(title_text='Deep Learning for Manufacturing - Model Estimates')
			py.offline.plot(fig, filename=deploy_path+"results.html")

		if(get_cam_data==1):
			#print(inference_model.summary())
			from cam_viz import CamViz
			from cop_viz import CopViz
			input_conv_data=inference_data
			base_cop=input_conv_data[0,:,:,:,0]+input_conv_data[0,:,:,:,1]+input_conv_data[0,:,:,:,2]
			base_cop[base_cop!=0]=0.6

			process_parameter_id=np.argmax(abs(result[0,:]))
			print("Plotting Gradient based Class Activation Map for Process Parameter: ",process_parameter_id)
			camviz=CamViz(inference_model,'conv_block_9')
			#For explicit plotting change ID here
			#process_parameter_id=0
			cop_input=input_conv_data[0:1,:,:,:,:]
			fmap_eval, grad_wrt_fmap_eval=camviz.grad_cam_3d(cop_input,process_parameter_id)
			alpha_k_c= grad_wrt_fmap_eval.mean(axis=(0,1,2,3)).reshape((1,1,1,-1))
			Lc_Grad_CAM = np.maximum(np.sum(fmap_eval*alpha_k_c,axis=-1),0).squeeze()
			scale_factor = np.array(cop_input.shape[1:4])/np.array(Lc_Grad_CAM.shape)

			from scipy.ndimage.interpolation import zoom
			import tensorflow.keras.backend as K
			
			_grad_CAM = zoom(Lc_Grad_CAM,scale_factor)
			arr_min, arr_max = np.min(_grad_CAM), np.max(_grad_CAM)
			grad_CAM = (_grad_CAM - arr_min) / (arr_max - arr_min + K.epsilon())

		#Code for Grad CAM Plotting
		plotly_viz=1	
		
		if(plotly_viz==1):
			import plotly.graph_objects as go
			import plotly as py
			import plotly.express as px
			X, Y, Z = np.mgrid[0:len(base_cop), 0:len(base_cop), 0:len(base_cop)]
			#input_conv_data[0,:,:,:,0]=0.2
			values_cop = base_cop
			values_grad_cam=grad_CAM

			trace1=go.Volume(
				x=X.flatten(),
				y=Y.flatten(),
				z=Z.flatten(),
				value=values_cop.flatten(),
				isomin=0,
				isomax=1,
				opacity=0.1, # needs to be small to see through all surfaces
				surface_count=17, # needs to be a large number for good volume rendering
				colorscale='Greens'
				)

			trace2=go.Volume(
				x=X.flatten(),
				y=Y.flatten(),
				z=Z.flatten(),
				value=values_grad_cam.flatten(),
				isomin=0,
				isomax=1,
				opacity=0.1, # needs to be small to see through all surfaces
				surface_count=17,
				colorscale='orrd' # needs to be a large number for good volume rendering
				)
			data = [trace1,trace2]
			
			layout = go.Layout(
				margin=dict(
					l=0,
					r=0,
					b=0,
					t=0
				)
			)
			
			fig = go.Figure(data=data,layout=layout)
			plot_file_name=deploy_path+'voxel_grad_cam.html'
			py.offline.plot(fig, filename=plot_file_name)


		return result

if __name__ == '__main__':
	
	print("Welcome to Deep Learning for Manufacturing (dlmfg)...")
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
	file_names_x=config.assembly_system['test_data_files_x']
	file_names_y=config.assembly_system['test_data_files_y']
	file_names_z=config.assembly_system['test_data_files_z']
	system_noise=config.assembly_system['system_noise']
	aritifical_noise=config.assembly_system['aritifical_noise']
	data_folder=config.assembly_system['data_folder']
	kcc_folder=config.assembly_system['kcc_folder']
	kcc_files=config.assembly_system['test_kcc_files']
	

	print('Initializing the Assembly System and Measurement System....')
	measurement_system=HexagonWlsScanner(data_type,application,system_noise,part_type,data_format)
	vrm_system=VRMSimulationModel(assembly_type,assembly_kccs,assembly_kpis,part_name,part_type,voxel_dim,voxel_channels,point_dim,aritifical_noise)
	deploy_model=DeployModel()
	
	#Generate Paths
	train_path='../trained_models/'+part_type
	model_path=train_path+'/model'+'/trained_model_0.h5'
	logs_path=train_path+'/logs'
	deploy_path=train_path+'/deploy/'

	#Voxel Mapping File

	get_data=GetTrainData();
	
	print('Importing and Preprocessing Cloud-of-Point Data')
	dataset=[]
	dataset.append(get_data.data_import(file_names_x,data_folder))
	dataset.append(get_data.data_import(file_names_y,data_folder))
	dataset.append(get_data.data_import(file_names_z,data_folder))
	point_index=get_data.load_mapping_index(mapping_index)


	#Make an Object of the Measurement System Class
	measurement_system=HexagonWlsScanner(data_type,application, system_noise,part_type,data_format)
	#Make an Object of the Assembly System Class
	assembly_system=PartType(assembly_type,assembly_kccs,assembly_kpis,part_name,part_type,voxel_dim,voxel_channels,point_dim)

	
	#Inference from simulated data
	inference_model=deploy_model.get_model(model_path)
	print(inference_model.summary())
	
	input_conv_data, kcc_subset_dump,kpi_subset_dump=get_data.data_convert_voxel_mc(vrm_system,dataset,point_index)

	y_pred=deploy_model.model_inference(input_conv_data,inference_model,deploy_path,print_result=1,plot_result=1);

	evalerror=1

	if(evalerror==1):
		kcc_dataset=get_data.data_import(kcc_files,kcc_folder)
		metrics_eval=MetricsEval();
		eval_metrics,accuracy_metrics_df=metrics_eval.metrics_eval_base(y_pred,kcc_dataset,logs_path)
		
		print('Evaluation Metrics: ',eval_metrics)
		accuracy_metrics_df.to_csv(logs_path+'/metrics_test.csv')
		
		np.savetxt((deploy_path+"predicted.csv"), y_pred, delimiter=",")
		print('Predicted Values saved to disk...')

	#Inference from Measurement Data

	#measurement_files=mscofig.ms_parameters['measurement_files']
	
	#Make an object of Get Data Class
	#get_data=GetInferenceData();
	
	#Call functions of the get Data Class
	#for measurement_file in measurement_files:	
		#measurement_path=deploy_path+measurement_file
		#measurement_data=get_data.load_measurement_file(measurement_path)
		#voxel_point_index=get_data.load_mapping_index(voxel_path)
		#y_dev_data_filtered=get_data.data_pre_processing(measurement_data,voxel_channels)
		#input_conv_data=get_data.voxel_mapping(y_dev_data_filtered,voxel_point_index,point_dim,voxel_dim,voxel_channels)
		#y_pred=deploy_model.model_inference(input_conv_data,inference_model);
		#print('KCCs for: ',measurement_file)
		#print(y_pred)

	#Code for Voxel Vizvalization

	#Code for CAM Visualization
	viz=0
	if(viz==1):
		print(inference_model.summary())
		camviz=CamViz(inference_model,'conv3d_3')

		grads=camviz.grad_cam_3d(input_conv_data[1:2,:,:,:,:],1)