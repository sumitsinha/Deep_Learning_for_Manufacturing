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

	:param download_params['id_test_data_files_y']: List of test data files IDs consisting of y-deviation of nodes on the server, minimum one file id in the list is required
	:type download_params['id_test_data_files_y']: list (required)

	:param download_params['id_test_data_files_z']: List of test data files IDs consisting of z-deviation of nodes on the server, minimum one file id in the list is required
	:type download_params['id_test_data_files_z']: list (required)

	:param download_params['id_mapping_index']: Mapping File ID
	:type download_params['id_mapping_index']: str (required)

	:param download_params['id_nominal_cop']: Nominal COP file ID
	:type download_params['id_nominal_cop']: str (required)
"""

download_params={
	   'download_type':'google_drive',
	   'base_url':'https://drive.google.com/uc?id=',
	   'id_kcc_files':['1yjDnSjyjw6-RYcrb7xtpedwnvMzAOo9V'],
	   'id_test_kcc_files':['1yjDnSjyjw6-RYcrb7xtpedwnvMzAOo9V'],
	   'id_data_files_x':['14NvGKB2eYLagkINPaMqio4ylxmK2qj4L'],
	   'id_data_files_y':['1sUfusVW7119DgdlZylZH2jEyBXXtVVcs'],
	   'id_data_files_z':['1MHhk9Xn7r7S0_PbA-QAKlEp5n0tFfKl_'],
	   'id_test_data_files_x':['1--nXi2N2cFpF_mXirqUrxzfBCWMy537h'],
	   'id_test_data_files_y':['1sUfusVW7119DgdlZylZH2jEyBXXtVVcs'],
	   'id_test_data_files_z':['1MHhk9Xn7r7S0_PbA-QAKlEp5n0tFfKl_'],
	   'id_mapping_index':'1yELJOyzgDOsrP5pP6xAy-LifC7gd2aqb',
	   'id_nominal_cop':'1m2FWTnZ70_fftrG-APs9DZR-NuC73AQW'
}
