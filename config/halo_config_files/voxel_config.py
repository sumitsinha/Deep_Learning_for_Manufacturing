"""The Voxelization configuration files consists of the parameters required when a mapping file for a different voxel resolution needs to be created, by default a mapping file of resolution 64*64*64 is downloaded by default when running any case stusy
                
        :param voxel_parameters['voxel_size']: Voxel resolution considering the voxel to be cubical
        :type voxel_parameters['voxel_size']: int (required)

        :param voxel_parameters['nominal_cop_filename']: The filename of the nominal cloud of point to be voxelized
        :type voxel_parameters['nominal_cop_filename']: str (required)

        Other parameters are needing in case data is to be pulled from the database server. by default the nominal_cop_file comes with the downloaded set of data files
"""

voxel_parameters = {	
        'voxel_size':64,
        'nominal_cop_filename':'halo_nominal_cop.csv',
        'table_name':'car_door_halo_nominal_cop',
		'database_type':'postgresql://',
		'username':'postgres',
		'password':'XXXXX',
		'ip_address':'10.255.1.130',
		'port_number':'5432',
		'database_name':'IPQI'
        }