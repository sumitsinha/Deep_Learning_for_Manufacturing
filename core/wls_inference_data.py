import numpy as np
import pandas as pd

class get_data():
	
	def get_dev_data(y1,y2,p):   
	    if(abs(y1)>abs(y2)):
	        y_dev=y1
	    else:
	        y_dev=y2
	    retval=y_dev
	    return retval

	def load_mapping_index(index_file='Halo_cov_index_data.dat'):
		
		try:
			voxel_point_index = np.load(index_file)
		except AssertionError as error:
			print(error)
			print('Voxel Mapping File not found !')

		return voxel_point_index

	def load_measurement_file(measurement_file_name)
		file_path="./Core_view_AM/"+measurement_file_name
		try:
			measurement_data=pd.read_csv(file_path,skiprows=25,low_memory=False,sep='\t', lineterminator='\r',error_bad_lines=False)
		except AssertionError as error:
			print(error)
			print('Measurement data file not found !')

		return measurement_data

	def data_pre_processing(voxel_channels=1)
		measurement_file_name="Check"
		measurement_data=load_measurement_file(measurement_file_name)
		measurement_data_subset=measurement_data.loc[(measurement_data['Name'].str[0:2] == 'SF')]
		nominal_coordinates=measurement_data_subset.iloc[:,5:8]
		actual_coordinates=measurement_data_subset.iloc[:,10:13]
		deviations=actual_coordinates.values-nominal_coordinates.values
		imputed_deviations= np.nan_to_num(deviations)
		y_dev_data_filtered=imputed_deviations[:,1:2]
		return y_dev_data_filtered

	def voxel_mapping(voxel_dim=64):
		
		y_dev_data_filtered=data_pre_processing();
		voxel_point_index=load_mapping_index();
		voxel_dev_data=np.zeros((1,voxel_dim,voxel_dim,voxel_dim,1))    

		for p in range(point_dim):
		    x_index=int(voxel_point_index[int(p),0])
		    y_index=int(voxel_point_index[int(p),1])
		    z_index=int(voxel_point_index[int(p),2])
		    voxel_dev_data[0,x_index,y_index,z_index,0]=get_dev_data(voxel_dev_data[0,x_index,y_index,z_index,0],y_dev_data_filtered[int(p)],int(p))

		return voxel_dev_data
		y_pred = final_model.predict(voxel_dev_data)