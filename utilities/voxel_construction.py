import os
import sys
current_path=os.path.dirname(__file__)
parentdir = os.path.dirname(current_path)

#Adding Path to various Modules
sys.path.append("../core")
sys.path.append("../visualization")
sys.path.append("../utilities")
sys.path.append("../config")

import pandas as pd
import numpy as np
from tqdm import tqdm

#Importing Config files
import assemblyconfig_halostamping as config
import voxel_config as vc

#Importing required modules from the package
from measurement_system import HexagonWlsScanner
from assembly_system import VRMSimulationModel


class VoxelConstruct:

	def __init__(self,x_dim,y_dim,z_dim):
		self.x_dim=x_dim
		self.y_dim=y_dim
		self.z_dim=z_dim
	

	def get_dev_data(self,x1,y1,z1,x2,y2,z2):   
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
		
		retval=[x_dev,y_dev,z_dev]
		return retval

	def construct_voxel(self,nominal_cop):
		
		def distance_func(x1,y1,z1,x2,y2,z2):
			import math
			dist = math.sqrt((x1-x2)**2 + (y1-y2)**2+(z1-z2)**2)
			return dist

		x_dim=self.x_dim
		y_dim=self.y_dim
		z_dim=self.z_dim

		point_dim=len(nominal_cop)

		df_point_index=np.zeros((point_dim,3))

		x_cor_max=max(nominal_cop[:,0])
		y_cor_max=max(nominal_cop[:,1])
		z_cor_max=max(nominal_cop[:,2])

		x_cor_min=min(nominal_cop[:,0])
		y_cor_min=min(nominal_cop[:,1])
		z_cor_min=min(nominal_cop[:,2])

		voxel_unit_x=int((x_cor_max-x_cor_min)/x_dim)
		voxel_unit_y=int((y_cor_max-y_cor_min)/y_dim)
		voxel_unit_z=int((z_cor_max-z_cor_min)/z_dim)

		array_locator=np.zeros((x_dim,y_dim,z_dim,3))

		for i in range(self.x_dim):
			array_locator[i,:,:,0]=x_cor_max
			x_cor_max=x_cor_max-voxel_unit_x

		for j in range(self.y_dim):
			array_locator[:,j,:,1]=y_cor_max
			y_cor_max=y_cor_max-voxel_unit_y

		for k in range(self.z_dim):
			array_locator[:,:,k,2]=z_cor_max
			z_cor_max=z_cor_max-voxel_unit_z

		for p in tqdm(range(point_dim)):
				min_distance=10000
				for i in range(x_dim):
					for j in range(y_dim):
						for k in range(z_dim):
							distance=distance_func(array_locator[i,j,k,0],array_locator[i,j,k,1],array_locator[i,j,k,2],nominal_cop[p,0],nominal_cop[p,1],nominal_cop[p,2])         
							if(distance<min_distance):
								min_distance=distance
								x_index=i
								y_index=j
								z_index=k
				df_point_index[p,0]=x_index
				df_point_index[p,1]=y_index
				df_point_index[p,2]=z_index

		return df_point_index

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
	file_names=config.assembly_system['data_files']
	system_noise=config.assembly_system['system_noise']
	aritifical_noise=config.assembly_system['aritifical_noise']
	
	print('Intilizing the Assembly System and Measurement System....')
	measurement_system=HexagonWlsScanner(data_type,application,system_noise,part_type,data_format)
	vrm_system=VRMSimulationModel(assembly_type,assembly_kccs,assembly_kpis,part_name,part_type,voxel_dim,voxel_channels,point_dim,aritifical_noise)
	#Import from File
	cop_file_name=vc.voxel_parameters['nominal_cop_filename']

	#import from Database
	#Format of connection String Databasetype + username:password + @IPaddress:Port_number + database
	#table_name=car_door_halo_nominal_cop
	#database_type=vc.voxel_parameters['database_type']
	#username=vc.voxel_parameters['username']
	#password=vc.voxel_parameters['password']
	#ip_address=vc.voxel_parameters['ip_address']
	#port_number=vc.voxel_parameters['port_number']
	#database_name=vc.voxel_parameters['database_name']
	
	#conn_string=data_basetype+':'+username+':'+password+'@'+ip_address+':'+password+'/'+database_name
	#'postgresql://postgres:sumit123!@10.255.1.130:5432/IPQI'

	file_path='../resources/mapping_files'+cop_file_name
	#Read cop from csv file
	print('Importing Nominal COP')
	nominal_cop=vrm_system.get_nominal_cop(file_path)

	#Read cop from SQL database
	#nominal_cop=vrm.get_nominal_cop_database(self,conn_string,table_name)
	
	#Passing Voxel
	print('voxelizing and creating mapping files...')
	voxel_construct=VoxelConstruct(voxel_dim,voxel_dim,voxel_dim)
	df_point_index=voxel_construct.construct_voxel(nominal_cop)
	
	#Dump Voxel
	name_cop=part_name+'_'+str(voxel_dim)+"_voxel_mapping.dat"
	df_point_index.dump(name_cop)

	print('Mapping file saved as: ',name_cop)


	


