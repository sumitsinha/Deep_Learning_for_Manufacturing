import numpy as np
import sys
import pandas as pd
""" Contains classes and methods to import reuired files to process measurement data """

class GetInferenceData():

	"""Infrence Data Class

		Import required files to deploy model on measurment systems
		
	"""

	def load_mapping_index(self,index_file):
		
		try:
			voxel_point_index = np.load(index_file,allow_pickle=True)
		except AssertionError as error:
			print(error)
			print('Voxel Mapping File not found !')

		return voxel_point_index

	def load_measurement_file(self,measurement_file_name):
		try:
			measurement_data=pd.read_csv(measurement_file_name,delim_whitespace=True,skiprows=25,low_memory=False,error_bad_lines=False)
		except AssertionError as error:
			print(error)
			print('Measurement data file not found !')

		return measurement_data

	def data_pre_processing(self,measurement_data,voxel_channels=1):
		
		measurement_data_subset=measurement_data.loc[(measurement_data['Name'].str[0:2] == 'SF')]
		nominal_coordinates=measurement_data_subset.iloc[:,5:8]
		actual_coordinates=measurement_data_subset.iloc[:,10:13]
		deviations=actual_coordinates.values-nominal_coordinates.values
		imputed_deviations= np.nan_to_num(deviations)
		
		if(voxel_channels==1):
			y_dev_data_filtered=imputed_deviations[:,1:2]
		if(voxel_channels==3):
			y_dev_data_filtered=imputed_deviations[:,1:4]
		
		return y_dev_data_filtered

	def voxel_mapping(self,y_dev_data_filtered,voxel_point_index,point_dim,voxel_dim,voxel_channels):
		

		def get_dev_data(y1,y2,p):   
		    if(abs(y1)>abs(y2)):
		        y_dev=y1
		    else:
		        y_dev=y2
		    retval=y_dev
		    return retval
		
		voxel_dev_data=np.zeros((1,voxel_dim,voxel_dim,voxel_dim,voxel_channels))    

		for p in range(point_dim):
		    x_index=int(voxel_point_index[int(p),0])
		    y_index=int(voxel_point_index[int(p),1])
		    z_index=int(voxel_point_index[int(p),2])
		    voxel_dev_data[0,x_index,y_index,z_index,0]=get_dev_data(voxel_dev_data[0,x_index,y_index,z_index,0],y_dev_data_filtered[int(p)],int(p))

		return voxel_dev_data
		y_pred = final_model.predict(voxel_dev_data)