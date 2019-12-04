import os
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

#Importing Required Modules
import pathlib
import gdown

#Importing Config files
import assemblyconfig_halostamping as config
import download_config as downloadconfig

url = 'https://drive.google.com/uc?id=1--nXi2N2cFpF_mXirqUrxzfBCWMy537h'
output = '../datasets/check.csv'
gdown.download(url, output, quiet=False)


if __name__ == '__main__':

	print('Parsing from Assembly Config File....')

	data_type=config.assembly_system['data_type']
	application=config.assembly_system['application']
	part_type=config.assembly_system['part_type']
	part_name=config.assembly_system['part_name']
	data_format=config.assembly_system['data_format']
	
	mapping_index=config.assembly_system['mapping_index']
	file_names_x=config.assembly_system['data_files_x']
	file_names_y=config.assembly_system['data_files_y']
	file_names_z=config.assembly_system['data_files_z']
	system_noise=config.assembly_system['system_noise']
	
	data_folder=config.assembly_system['data_folder']
	kcc_folder=config.assembly_system['kcc_folder']
	kcc_files=config.assembly_system['kcc_files']

	print('Parsing from Download Config File')

	mapping_index=downloadconfig.download_params['mapping_index']
	nominal_cop_filename=downloadconfig.download_params['nominal_cop_filename']
	file_names_x=downloadconfig.download_params['data_files_x']
	file_names_y=downloadconfig.download_params['data_files_y']
	file_names_z=downloadconfig.download_params['data_files_z']
	kcc_files=downloadconfig.download_params['kcc_files']

	test_file_names_x=downloadconfig.download_params['test_data_files_x']
	test_file_names_y=downloadconfig.download_params['test_data_files_y']
	test_file_names_z=downloadconfig.download_params['test_data_files_z']
	test_kcc_files=downloadconfig.download_params['test_kcc_files']
	
	id_kcc_files=downloadconfig.download_params['id_kcc_files'],
	id_test_kcc_files=downloadconfig.download_params[]
	   'id_data_files_x':['nXi2N2cFpF_mXirqUrxzfBCWMy537h'],
	   'id_data_files_y':['1sUfusVW7119DgdlZylZH2jEyBXXtVVcs'],
	   'id_data_files_z':['1MHhk9Xn7r7S0_PbA-QAKlEp5n0tFfKl_'],
	   'id_test_data_files_x':['nXi2N2cFpF_mXirqUrxzfBCWMy537h'],
	   'id_test_data_files_y':['1sUfusVW7119DgdlZylZH2jEyBXXtVVcs'],
	   'id_test_data_files_z':['1MHhk9Xn7r7S0_PbA-QAKlEp5n0tFfKl_'],
	   'id_mapping_index':['1yELJOyzgDOsrP5pP6xAy-LifC7gd2aqb'],
	   'id_nominal_cop':['1m2FWTnZ70_fftrG-APs9DZR-NuC73AQW']
	
	print('Creating file Structure....')
	
	folder_name=part_type
	train_path='../datasets/'+part_type
	pathlib.Path(train_path).mkdir(parents=True, exist_ok=True) 

	nominal_cop_path='../resources/nominal_cop_files'
	pathlib.Path(train_path).mkdir(parents=True, exist_ok=True) 
	
	mapping_files_path='../resources/mapping_files'
	pathlib.Path(train_path).mkdir(parents=True, exist_ok=True) 

	kcc_files_path='../active_learning/sample_input/'+part_type
	pathlib.Path(train_path).mkdir(parents=True, exist_ok=True)

