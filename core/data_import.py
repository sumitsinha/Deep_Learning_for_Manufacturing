import pandas as pd
import numpy as np
import tqdm

class GetTrainData(VRMSimulationModel)
    
    def data_import():
        pass
        
    def data_convert_voxel(self,dataset):
    	
        point_dim=self.point_dim
        voxel_dim=self.voxel_dim
        dev_channel=self.dev_channel
        noise_level=self.noise_level
        noise_type=self.noise_type

        #Declaring the variables for intilizing input data structure intilization  
    	start_index=0
    	end_index=len(dataset)
    	
        #end_index=50000
    	run_length=end_index-start_index
    	input_conv_data=np.zeros((length,voxel_dim,voxel_dim,voxel_dim,dev_channel))

    	for index in tqdm(run_length):
        y_point_data=dataset.iloc[index, 0:point_dim]
        dev_data=y_point_data.values
        if(noise_type='uniform'):
            measurement_noise= np.random.uniform(low=-noise_level, high=noise_level, size=(point_dim))
        else:
            measurement_noise=np.random.gauss(0,noise_level, size=(point_dim))
        dev_data=dev_data+measurement_noise
        cop_dev_data=np.zeros((voxel_dim,voxel_dim,voxel_dim,dev_channel))    
        
        for p in range(point_dim):
            x_index=int(df_point_index[p,0])
            y_index=int(df_point_index[p,1])
            z_index=int(df_point_index[p,2])
            cop_dev_data[x_index,y_index,z_index,0]=get_dev_data(cop_dev_data[x_index,y_index,z_index,0],dev_data[p])
        input_conv_data[index,:,:,:]=cop_dev_data

    	kcc_subset_dump=kcc_dump[start_index:end_index,:]

    	return input_conv_data, kcc_subset_dump

    if __name__=="__main__"
        #Importing Datafiles
        print('Importing and preprocessing Cloud-of-Point Data')
        dataset_0 = pd.read_csv('./Data/car_halo_run1_ydev.csv',header=None)
        dataset_1 = pd.read_csv('./Data/car_halo_run2_ydev.csv',header=None)
        dataset_2 = pd.read_csv('./Data/car_halo_run3_ydev.csv',header=None)
        dataset_3 = pd.read_csv('./Data/car_halo_run4_ydev.csv',header=None)
        dataset_4 = pd.read_csv('./Data/car_halo_run5_ydev.csv',header=None)
        dataset_5 = pd.read_csv('./Data/car_halo_run6_ydev.csv',header=None)
        dataset_6 = pd.read_csv('./Data/car_halo_run7_ydev.csv',header=None)
        dataset_7 = pd.read_csv('./Data/car_halo_run8_ydev.csv',header=None)

        #Importing Dataset and calling data import function
        dataset = pd.concat([dataset_0, dataset_1,dataset_2,dataset_3,dataset_4,dataset_5,dataset_6,dataset_7], ignore_index=True)
        input_conv_data,kcc_subset_dump=data_import()
        print('Data import and processing completed')
