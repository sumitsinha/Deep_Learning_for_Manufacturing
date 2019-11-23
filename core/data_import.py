import pandas as pd
import numpy as np
from tqdm import tqdm

class GetTrainData():
	
	def data_import(self,file_names):
		
		"""
		dataset_0 = pd.read_csv('./Data/car_halo_run1_ydev.csv',header=None)
		dataset_1 = pd.read_csv('./Data/car_halo_run2_ydev.csv',header=None)
		dataset_2 = pd.read_csv('./Data/car_halo_run3_ydev.csv',header=None)
		dataset_3 = pd.read_csv('./Data/car_halo_run4_ydev.csv',header=None)
		dataset_4 = pd.read_csv('./Data/car_halo_run5_ydev.csv',header=None)
		dataset_5 = pd.read_csv('./Data/car_halo_run6_ydev.csv',header=None)
		dataset_6 = pd.read_csv('./Data/car_halo_run7_ydev.csv',header=None)
		dataset_7 = pd.read_csv('./Data/car_halo_run8_ydev.csv',header=None)
		dataset = pd.concat([dataset_0, dataset_1,dataset_2,dataset_3,dataset_4,dataset_5,dataset_6,dataset_7], ignore_index=True)
		
		"""

		data_files=[]
		for file in file_names:
			file_path='../datasets/'+file
			data_files.append(pd.read_csv(file_path,header=None))
		dataset = pd.concat(data_files, ignore_index=True)
		return dataset

	def load_mapping_index(self,index_file):
		
		file_path='../utilities/'+index_file
		try:
			voxel_point_index = np.load(file_path,allow_pickle=True)
		except AssertionError as error:
			print(error)
			print('Voxel Mapping File not found !')

		return voxel_point_index

	

	def data_convert_voxel(self,vrm_system,dataset,point_index):
		
		def get_dev_data(y1,y2):   
		    if(abs(y1)>abs(y2)):
		        y_dev=y1
		    else:
		        y_dev=y2
		    retval=y_dev
		    return retval

		point_dim=vrm_system.point_dim
		voxel_dim=vrm_system.voxel_dim
		dev_channel=vrm_system.voxel_channels
		noise_level=vrm_system.noise_level
		noise_type=vrm_system.noise_type
		kcc_dim=vrm_system.assembly_kccs
		kpi_dim=vrm_system.assembly_kpis

		#Declaring the variables for intilizing input data structure intilization  
		start_index=0
		end_index=len(dataset)
		
		#end_index=50000
		run_length=end_index-start_index
		input_conv_data=np.zeros((run_length,voxel_dim,voxel_dim,voxel_dim,dev_channel))
		kcc_dump=dataset.iloc[start_index:end_index, point_dim:point_dim+kcc_dim]
		kpi_dump=dataset.iloc[start_index:end_index, point_dim+kcc_dim:point_dim+kcc_dim+kpi_dim]

		for index in tqdm(range(run_length)):
			y_point_data=dataset.iloc[index, 0:point_dim]
			dev_data=y_point_data.values
			if(noise_type=='uniform'):
				measurement_noise= np.random.uniform(low=-noise_level, high=noise_level, size=(point_dim))
			else:
				measurement_noise=np.random.gauss(0,noise_level, size=(point_dim))
			dev_data=dev_data+measurement_noise
			cop_dev_data=np.zeros((voxel_dim,voxel_dim,voxel_dim,dev_channel))    
		
			for p in range(point_dim):
				x_index=int(point_index[p,0])
				y_index=int(point_index[p,1])
				z_index=int(point_index[p,2])
				cop_dev_data[x_index,y_index,z_index,0]=get_dev_data(cop_dev_data[x_index,y_index,z_index,0],dev_data[p])
			
			input_conv_data[index,:,:,:]=cop_dev_data

		return input_conv_data, kcc_dump,kpi_dump

if (__name__=="__main__"):
	#Importing Datafiles
	print('Importing and preprocessing Cloud-of-Point Data')
	file_names=['car_halo_run1_ydev.csv','car_halo_run2_ydev.csv','car_halo_run3_ydev.csv','car_halo_run4_ydev.csv','car_halo_run5_ydev.csv']
	datset=data_import(file_names)
	#Importing Dataset and calling data import function
	dataset = pd.concat([dataset_0, dataset_1,dataset_2,dataset_3,dataset_4,dataset_5,dataset_6,dataset_7], ignore_index=True)
	input_conv_data,kcc_subset_dump=data_convert_voxel(dataset)
	print('Data import and processing completed')
