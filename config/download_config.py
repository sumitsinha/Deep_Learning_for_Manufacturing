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
	   'id_kcc_files':['1MjUef2sgOA0aLrK-2yp0O8tMUYme9ksb'],
	   'id_test_kcc_files':['1RpD1fDkEt3M9Gxl1XWg-Mi37MW2Bxx0x'],
	   'id_data_files_x':['1lxNnDQF77FGquQJlGhfFGu-J6XNhae7_'],
	   'id_data_files_y':['1Xje2PLy7d4BmPQr4K1xo8dV0fgZOXHAP'],
	   'id_data_files_z':['1kWXaBJ283ifdmW0BuWh8wtAvAtfaYT-R'],
	   'id_test_data_files_x':['19bcf6AqqR_95NZFGd9QjgqtPAS1DFNhK'],
	   'id_test_data_files_y':['1DadKeGZ0C0vDH0qW4RZ4hWkREDCJLKEe'],
	   'id_test_data_files_z':['1YENg66GhvFaz8G4lRBUO6r7IL13fMz-d'],
	   'id_mapping_index':'1RibrkROYqHLRVljSGfe5jVIhpv_H9abP',
	   'id_nominal_cop':'1hrTlEnHXHg7T53XvXEjhIPSrRSRJLqoo'
}

