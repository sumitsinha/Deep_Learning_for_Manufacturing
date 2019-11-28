import pandas as pd
import numpy as np
from tqdm import tqdm

class GetTrainData():
	
	def data_import(self,file_names,data_folder):
		data_files=[]
		for file in file_names:
			file_path=data_folder+file
			data_files.append(pd.read_csv(file_path,header=None))
		dataset = pd.concat(data_files, ignore_index=True)
		return dataset

	def load_mapping_index(self,index_file):
		
		file_path='../resources/mapping_files/'+index_file
		try:
			voxel_point_index = np.load(file_path,allow_pickle=True)
		except AssertionError as error:
			print(error)
			print('Voxel Mapping File not found !')

		return voxel_point_index

	def data_convert_voxel_mc(self,vrm_system,dataset,point_index,kcc_data):
		
		def get_dev_data(x1,x2,y1,y2,z1,z2):   
			
			if(abs(x1)>abs(x2)):
				x_dev=x1
			else:
				x_dev=x2
			
			if(abs(y1)>abs(y2)):
				y_dev=y1
			else:
				y_dev=y2

			if(abs(z1)>abs(z2)):
				z_dev=z1
			else:
				z_dev=z2
			
			retval=np.array([x_dev,y_dev,z_dev])
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
		end_index=len(dataset[0])
		
		#end_index=50000
		run_length=end_index-start_index
		input_conv_data=np.zeros((run_length,voxel_dim,voxel_dim,voxel_dim,dev_channel))
		kcc_dump=kcc_data.values
		#kcc_dump=dataset.iloc[start_index:end_index, point_dim:point_dim+kcc_dim]
		kpi_dump=dataset[0].iloc[start_index:end_index, point_dim:point_dim+kpi_dim]
		kpi_dump=kpi_dump.values
		not_convergent=0
		for index in tqdm(range(run_length)):
			x_point_data=dataset[0].iloc[index, 0:point_dim]
			y_point_data=dataset[1].iloc[index, 0:point_dim]
			z_point_data=dataset[2].iloc[index, 0:point_dim]
			
			if(dataset[0].iloc[index, point_dim]==0):
				not_convergent=not_convergent+1

			dev_data_x=x_point_data.values
			dev_data_y=y_point_data.values
			dev_data_z=z_point_data.values

			if(noise_type=='uniform'):
				measurement_noise_x= np.random.uniform(low=-noise_level, high=noise_level, size=(point_dim))
				measurement_noise_y= np.random.uniform(low=-noise_level, high=noise_level, size=(point_dim))
				measurement_noise_z= np.random.uniform(low=-noise_level, high=noise_level, size=(point_dim))
			else:
				measurement_noise_x=np.random.gauss(0,noise_level, size=(point_dim))
				measurement_noise_y=np.random.gauss(0,noise_level, size=(point_dim))
				measurement_noise_z=np.random.gauss(0,noise_level, size=(point_dim))

			dev_data_x=dev_data_x+measurement_noise_x
			dev_data_y=dev_data_y+measurement_noise_y
			dev_data_z=dev_data_z+measurement_noise_z

			cop_dev_data=np.zeros((voxel_dim,voxel_dim,voxel_dim,dev_channel))    
		
			for p in range(point_dim):
				x_index=int(point_index[p,0])
				y_index=int(point_index[p,1])
				z_index=int(point_index[p,2])
				cop_dev_data[x_index,y_index,z_index,:]=get_dev_data(cop_dev_data[x_index,y_index,z_index,0],dev_data_x[p],cop_dev_data[x_index,y_index,z_index,1],dev_data_y[p],cop_dev_data[x_index,y_index,z_index,2],dev_data_z[p])
			
			input_conv_data[index,:,:,:]=cop_dev_data

		print("Number of not convergent solutions: ",not_convergent)
		return input_conv_data, kcc_dump,kpi_dump

	def data_convert_voxel_sc(self,vrm_system,dataset,point_index):
			
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

				y_point_data=dataset[1].iloc[index, 0:point_dim]
				dev_data_y=y_point_data.values

				if(noise_type=='uniform'):
					measurement_noise_y= np.random.uniform(low=-noise_level, high=noise_level, size=(point_dim))
				else:
					measurement_noise_y=np.random.gauss(0,noise_level, size=(point_dim))


				dev_data_y=dev_data_y+measurement_noise_y

				cop_dev_data=np.zeros((voxel_dim,voxel_dim,voxel_dim,dev_channel))    
			
				for p in range(point_dim):
					x_index=int(point_index[p,0])
					y_index=int(point_index[p,1])
					z_index=int(point_index[p,2])
					cop_dev_data[x_index,y_index,z_index,0]=get_dev_data(cop_dev_data[x_index,y_index,z_index,0],dev_data_x[p])
				
				input_conv_data[index,:,:,:]=cop_dev_data

			return input_conv_data, kcc_dump,kpi_dump

if (__name__=="__main__"):
	#Importing Datafiles
	print('Function for importing and preprocessing Cloud-of-Point Data')
	
