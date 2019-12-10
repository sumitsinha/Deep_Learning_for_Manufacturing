""" Containts classes and methods to create the input file structure and download the reuired data
	The program parses from the download_config.py file
	Currently leverages gdown library (https://pypi.org/project/gdown/) to download large files from google drive
	Main Function parses from the download_config.py file and downloads the input, output and support files for the corresponding case study and then places them in a pre-specified file sturture to be used for model training, deployment and data study
"""

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
import assembly_config as config
import download_config as downloadconfig

class DataDownload:
	"""Data Download Class

		:param base_url: consists of the base URL of the server file location
		:type base_url: str (required)

		:param download_type: Type of download, currently google drive is used host the datafiles
		:type download_type: str (required)

		:param download_flag: used to store the number of downloads done using one instance of the Data Download class, can be used to ensure Quality Checks on the downloaded data
		:type download_flag: int 
	"""

	def __init__(self,base_url,download_type,download_flag=0):
			self.base_url=base_url
			self.download_type=download_type
			self.download_flag=download_flag
			
	def google_drive_downloader(self,file_id,output):
		"""google_drive_downloader combines object initilization with the file ID to download to the desired output file

			:param file_id: Server file ID of the file to be downloaded
			:type file_id: str (required)

			:param output: output path of the downloaded file
			:type output: str (required)
		"""
		print('Attempting download from ', self.download_type, ' for output: ',output)
		url=self.base_url+file_id
		gdown.download(url, output, quiet=False)
		print('Download Comleted for: ',output)
		self.download_flag=self.download_flag+1


if __name__ == '__main__':
	""" Main Function parses from the download_config.py file and downloads the input, output and support files for the corresponding case study and then places them in a pre-specified file sturture to be used for model training, deployment and data study"""

	print('Parsing from Assembly Config File....')

	data_type=config.assembly_system['data_type']
	application=config.assembly_system['application']
	part_type=config.assembly_system['part_type']
	part_name=config.assembly_system['part_name']
	data_format=config.assembly_system['data_format']
	
	mapping_index=config.assembly_system['mapping_index']
	nominal_cop_filename=config.assembly_system['nominal_cop_filename']

	file_names_x=config.assembly_system['data_files_x']
	file_names_y=config.assembly_system['data_files_y']
	file_names_z=config.assembly_system['data_files_z']
	kcc_files=config.assembly_system['kcc_files']

	data_folder=config.assembly_system['data_folder']
	kcc_folder=config.assembly_system['kcc_folder']
	
	test_file_names_x=config.assembly_system['test_data_files_x']
	test_file_names_y=config.assembly_system['test_data_files_y']
	test_file_names_z=config.assembly_system['test_data_files_z']
	test_kcc_files=config.assembly_system['test_kcc_files']
	
	print('Parsing from Download Config File')

	id_kcc_files=downloadconfig.download_params['id_kcc_files']
	id_test_kcc_files=downloadconfig.download_params['id_test_kcc_files']
	
	id_data_files_x=downloadconfig.download_params['id_data_files_x']
	id_data_files_y=downloadconfig.download_params['id_data_files_y']
	id_data_files_z=downloadconfig.download_params['id_data_files_z']
	
	id_test_data_files_x=downloadconfig.download_params['id_test_data_files_x']
	id_test_data_files_y=downloadconfig.download_params['id_test_data_files_y']
	id_test_data_files_z=downloadconfig.download_params['id_test_data_files_z']
	   
	id_mapping_index=downloadconfig.download_params['id_mapping_index']
	id_nominal_cop=downloadconfig.download_params['id_nominal_cop']
	
	download_type=downloadconfig.download_params['download_type']
	base_url=downloadconfig.download_params['base_url']

	print('Creating file Structure for downloaded files....')
	
	pathlib.Path(data_folder).mkdir(parents=True, exist_ok=True)
	pathlib.Path(kcc_folder).mkdir(parents=True, exist_ok=True)

	nominal_cop_path='../resources/nominal_cop_files'
	pathlib.Path(nominal_cop_path).mkdir(parents=True, exist_ok=True) 
	
	mapping_files_path='../resources/mapping_files'
	pathlib.Path(mapping_files_path).mkdir(parents=True, exist_ok=True)

	data_download=DataDownload(base_url,download_type)

	data_download.google_drive_downloader(id_mapping_index,(mapping_files_path+'/'+mapping_index))
	data_download.google_drive_downloader(id_nominal_cop,(nominal_cop_path+'/'+nominal_cop_filename))

	for i, file in enumerate(file_names_x):
		data_download.google_drive_downloader(id_data_files_x[i],(data_folder+'/'+file))

	for i, file in enumerate(file_names_y):
		data_download.google_drive_downloader(id_data_files_y[i],(data_folder+'/'+file))

	for i, file in enumerate(file_names_z):
		data_download.google_drive_downloader(id_data_files_z[i],(data_folder+'/'+file))

	for i, file in enumerate(kcc_files):
		data_download.google_drive_downloader(id_kcc_files[i],(kcc_folder+'/'+file))
	
	for i, file in enumerate(test_file_names_x):
		data_download.google_drive_downloader(id_test_data_files_x[i],(data_folder+'/'+file))

	for i, file in enumerate(test_file_names_y):
		data_download.google_drive_downloader(id_test_data_files_y[i],(data_folder+'/'+file))

	for i, file in enumerate(test_file_names_z):
		data_download.google_drive_downloader(id_test_data_files_z[i],(data_folder+'/'+file))

	for i, file in enumerate(test_kcc_files):
		data_download.google_drive_downloader(id_test_kcc_files[i],(kcc_folder+'/'+file))

	print('File Structure built and Download completed for a no of file: ',data_download.download_flag)	