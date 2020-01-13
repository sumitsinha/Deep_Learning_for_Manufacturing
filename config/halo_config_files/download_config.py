"""The download config file consists of parameters required to download files
		
	:param download_params['download_type']: Server name of the file location, currently supports Google Drive 
	:type download_params['download_type']: str (required)

	:param download_params['base_url']: base URL of the server 
	:type download_params['base_url']: str (required)

	:param download_params['id_kcc_files']: List of KCC file IDs on the server, minimum one file id in the list is required
	:type download_params['id_kcc_files']: list (required)

	:param download_params['id_test_kcc_files']: List of test KCC file IDs on the server, minimum one file id in the list is required
	:type download_params['id_test_kcc_files']: list (required)

	:param download_params['id_data_files_x']: List of data files IDs consisting of x-deviation of nodes on the server, minimum one file id in the list is required
	:type download_params['id_data_files_x']: list (required)

	:param download_params['id_data_files_y']: List of data files IDs consisting of y-deviation of nodes on the serve, minimum one file id in the list is required
	:type download_params['id_data_files_y']: list (required)

	:param download_params['id_data_files_z']: List of data files IDs consisting of z-deviation of nodes on the serve, minimum one file id in the list is required
	:type download_params['id_data_files_z']: list (required)

	:param download_params['id_test_data_files_x']: List of test data files IDs consisting of x-deviation of nodes on the server, minimum one file id in the list is required
	:type download_params['id_test_data_files_x']: list (required)

	:param download_params['id_test_data_files_y']: List of test data files IDs consisting of y-deviation of nodes on the serve, minimum one file id in the list is required
	:type download_params['id_test_data_files_y']: list (required)

	:param download_params['id_test_data_files_z']: List of test data files IDs consisting of z-deviation of nodes on the serve, minimum one file id in the list is required
	:type download_params['id_test_data_files_z']: list (required)

	:param download_params['id_mapping_index']: Mapping File ID
	:type download_params['id_mapping_index']: str (required)

	:param download_params['id_nominal_cop']: Nominal COP file ID
	:type download_params['id_nominal_cop']: str (required)
"""

download_params={
	   'download_type':'google_drive',
	   'base_url':'https://drive.google.com/uc?id=',
	   'id_kcc_files':['1kJBIgU9qQAbsfIpcwOdqOy1qInkVL72T'],
	   'id_test_kcc_files':['1rEGnmIkJDvQo_PU82s5_Ba_oZf_Gjzyi'],
	   'id_data_files_x':['11li7eax_4uyspoHpuE0DMLlwMm_Qa10c'],
	   'id_data_files_y':['1w_YD7oXF1E43E_pHRaRe35tMbftV8jO3'],
	   'id_data_files_z':['184N3w3EDY3q6o5NYMNrPl-H39Hmh6Jh2'],
	   'id_test_data_files_x':['15yjxWV1nH_qoZ2Hmxtuh9WR1w_kz7Srh'],
	   'id_test_data_files_y':['1VAJow5YOXdbtx-6Ed3wDir7ksji7njVu'],
	   'id_test_data_files_z':['1hW4LGTGJUsJfN75hnrE4MUr7WyFhOh4K'],
	   'id_mapping_index':'1yELJOyzgDOsrP5pP6xAy-LifC7gd2aqb',
	   'id_nominal_cop':'1m2FWTnZ70_fftrG-APs9DZR-NuC73AQW'
}

