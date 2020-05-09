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
from tkinter import *
import logging
tf.get_logger().setLevel(logging.ERROR)

from keras.models import load_model


#Importing Config files
import assembly_config as config
import model_config as cftrain
import measurement_config as mscofig
import voxel_config as vc

#Importing required modules from the package
from measurement_system import HexagonWlsScanner
from assembly_system import VRMSimulationModel
from assembly_system import PartType
from wls400a_system import GetInferenceData
from metrics_eval import MetricsEval
from data_import import GetTrainData
from model_deployment import DeployModel
from cop_viz import CopViz
from tkinter import filedialog



def import_data_demo(get_data,point_index,file_names_x,file_names_y,file_names_z,data_folder):
	
	global input_conv_data
	global dataset
	global folder_path
	filename = filedialog.askdirectory()
	folder_path.set(filename)
	
	print("Selected Data Folder: "+ filename)
 

	print('Importing and Preprocessing Cloud-of-Point Data')
	dataset=[]
	dataset.append(get_data.data_import(file_names_x,filename))
	dataset.append(get_data.data_import(file_names_y,filename))
	dataset.append(get_data.data_import(file_names_z,filename))
	input_conv_data, kcc_subset_dump,kpi_subset_dump=get_data.data_convert_voxel_mc(vrm_system,dataset,point_index)
	

	#return input_conv_data,dataset

def plot_data(dataset,nominal_cop,deploy_path):
	
	print('Visualizing Cloud-of-Point data')
	x_dev=dataset[0].values
	y_dev=dataset[1].values
	z_dev=dataset[2].values
	print(x_dev.shape)
	dev=np.zeros_like(nominal_cop)
	dev[:,0]=x_dev[0,0:(x_dev.shape[1]-1)]
	dev[:,1]=y_dev[0,0:(y_dev.shape[1]-1)]
	dev[:,2]=z_dev[0,0:(z_dev.shape[1]-1)]

	sample_cop=nominal_cop+dev
	plot_file_name=deploy_path+'_sample_cop.html'
	copviz=CopViz(sample_cop)
	copviz.plot_cop(plot_file_name)

def browse_button(get_data,point_index,file_names_x,file_names_y,file_names_z,data_folder):
	# Allow user to select a directory and store it in global var
	# called folder_path
	global folder_path
	filename = filedialog.askdirectory()
	folder_path.set(filename)
	print("Selected Data Folder: "+ filename)

def show_leaderboard():
	os.system('python -W ignore ../leaderboard/leaderboard_gen.py')

def deploy_model_visual():
	os.system('python -W ignore model_deploy_visual.py')

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
	get_data=GetTrainData();
	
	#Generate Paths
	train_path='../trained_models/'+part_type
	model_path=train_path+'/model'+'/trained_model_0.h5'
	logs_path=train_path+'/logs'
	deploy_path=train_path+'/deploy/'

	#Import all static resources
	#import Model
	inference_model=deploy_model.get_model(model_path)
	point_index=get_data.load_mapping_index(mapping_index)
	cop_file_name=vc.voxel_parameters['nominal_cop_filename']
	file_path='../resources/nominal_cop_files/'+cop_file_name
	#Read cop from csv file
	nominal_cop=vrm_system.get_nominal_cop(file_path)

	#Calling Functions

	window=Tk()

	window.title("Welcome to Deep Learning for Manufacturing (dlmfg)")
	window.geometry('550x200')
	#Get data
	

	#Plot Cloud-of-Point Data
	#plot_data(dataset,nominal_cop)

	# Demo
	#y_pred=deploy_model.model_inference(input_conv_data,inference_model)
	folder_path = StringVar()
	lbl1 = Label(master=window,textvariable=folder_path)
	lbl1.grid(row=2, column=1)

	F=Button(window,text="Load Data",command= lambda: import_data_demo(get_data,point_index,file_names_x,file_names_y,file_names_z,data_folder))
	F.grid(row=0, column=2)

	A=Button(window,text="Plot Data",command= lambda:plot_data(dataset,nominal_cop,deploy_path))
	A.grid(row=0, column=3)

	B=Button(window,text="Run Model",command= lambda:deploy_model.model_inference(input_conv_data,inference_model,deploy_path))
	B.grid(row=0, column=4)

	#V=Button(window,text="Model Insight: CAM",command= deploy_model_visual)
	#V.grid(row=0, column=5)

	#L=Button(window,text="Show Leader Board",command= show_leaderboard)
	#L.grid(row=3, column=3)

	window.mainloop()


	