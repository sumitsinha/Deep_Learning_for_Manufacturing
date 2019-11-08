import pandas as pd
import numpy as np
import tqdm

class VoxelConstruct:

	def __init__(self,x_dim,y_dim,z_dim):

		x_dim=self.x_dim
		y_dim=self.y_dim
		z_dim=self.z_dim


	def distance_func(self,x1,y1,z1,x2,y2,z2):
	    dist = math.sqrt((x1-x2)**2 + (y1-y2)**2+(z1-z2)**2)
	    return dist

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
		
		x_dim=self.x_dim
		y_dim=self.y_dim
		z_dim=self.z_dim

		point_dim=len(nominal_cop)

		df_point_index=np.zeros((,3))

		for p in tqdm(range(8047)):
		        min_distance=10000
		        for i in range(54):
		            for j in range(127):
		                for k in range(26):
		                    distance=distance_func(array_locator[i,j,k,0],array_locator[i,j,k,1],array_locator[i,j,k,2],df_nom[p,2],df_nom[p,0],df_nom[p,1])         
		                    if(distance<min_distance):
		                        min_distance=distance
		                        x_index=i
		                        y_index=j
		                        z_index=k
		        df_point_index[p,0]=x_index
		        df_point_index[p,1]=y_index
		        df_point_index[p,2]=z_index


#%%
		x_cor_max=max(nominal_cop[]:,0)
		y_cor_max=max(nominal_cop[]:,1)
		z_cor_max=max(nominal_cop[]:,2)

		x_cor_min=min(nominal_cop[]:,0)
		y_cor_min=min(nominal_cop[]:,1)
		z_cor_min=min(nominal_cop[]:,2)

		voxel_unit_x=int((x_cor_max-x_cor_min)/x_dim)
		voxel_unit_y=int((y_cor_max-y_cor_min)/y_dim)
		voxel_unit_z=int((z_cor_max-z_cor_min)/z_dim)

		for i in range(self.x_dim):
		    array_locator[i,:,:,0]=x_cor
		    x_cor=x_cor-13

		for j in range(self.y_dim):
		    array_locator[:,j,:,1]=y_cor
		    y_cor=y_cor-10

		for k in range(self.z_dim):
		    array_locator[:,:,k,2]=z_cor
		    z_cor=z_cor-10

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Arguments to initiate Measurement System Class and Assembly System Class")
    parser.add_argument("-D", "--data_type", help = "Example: 3D Point Cloud Data", required = False, default = "3D Point Cloud Data")
    parser.add_argument("-A", "--application", help = "Example: Inline Root Cause Analysis", required = False, default = "Inline Root Cause Analysis")
    parser.add_argument("-P", "--part_type", help = "Example: Door Inner and Hinge Assembly", required = False, default = "Door Inner and Hinge Assembly")
    parser.add_argument("-F", "--data_format", help = "Example: Complete vs Partial Data", required = False, default = "Complete")
	parser.add_argument("-S", "--assembly_type", help = "Example: Multi-Stage vs Single-Stage", required = False, default = "Single-Stage")
    parser.add_argument("-C", "--assembly_kccs", help = "Number of KCCs for the Assembly", required = False, default =15,type=int )
    parser.add_argument("-I", "--assembly_kpis	", help = "Number of KPIs for the Assembly", required = False, default = 6,type=int)
    parser.add_argument("-V", "--voxel_dim", help = "The Granularity of Voxels - 32 64 128", required = False, default = 64,type=int)
    parser.add_argument("-P", "--point_dim", help = "Number of key Nodes", required = True, type=int)
    parser.add_argument("-C", "--voxel_channels", help = "Number of Channels - 1 or 3", required = False, default = 1,type=int)
    parser.add_argument("-N", "--noise_levels", help = "Amount of Artificial Noise to add while training", required = False, default = 0.1,type=float)
    parser.add_argument("-T", "--noise_type", help = "Type of noise to be added uniform/Gaussian default uniform", required = False, default = "uniform")
	argument = parser.parse_args()
	
	data_type=argument.data_type
	application=argument.application
	part_type=argument.part_type
	data_format=argument.data_format
	assembly_type=argument.assembly_type	
	assembly_kccs=argument.assembly_kccs	
	assembly_kpis=argument.assembly_kpis
	voxel_dim=argument.voxel_dim
	point_dim=argument.point_dim
	voxel_channels=argument.voxel_channels
	noise_levels=argument.noise_levels
	noise_type=argument.noise_type

	#Objects of Measurement System and Assembly System
	measurement_system=HexagonWlsScanner(data_type,application, system_noise,part_type,data_format)
	vrm_system=VRMSimulationModel(assembly_type,assembly_kccs,assembly_kpis,voxel_dim,point_dim,voxel_channels,noise_levels,noise_type)

	#Import from File
	cop_file_name=''

	#import from Database
	#Format of connection String Databasetype + username:password + @IPaddress:Port_number + database
	table_name=car_door_halo_nominal_cop
	database_type='postgresql://'
	username='postgres'
	password='sumit123!'
	ip_address='10.255.1.130'
	port_number='5432'
	database_name='IPQI'
	conn_string=data_basetype+':'+username+':'+password+'@'+ip_address+':'+password+'/'+database_name
	#'postgresql://postgres:sumit123!@10.255.1.130:5432/IPQI'

	#Read cop from csv file
	vrm_system.get_nominal_cop(self,file_name)

	#Read cop from SQL database
	vrm.get_nominal_cop_database(self,conn_string,table_name)


	


