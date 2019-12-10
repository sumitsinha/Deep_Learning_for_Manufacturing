"""The assembly config file required to initialize the assembly class and perform model training and deployment

        :param assembly_system['data_type']: The incoming data type from the assembly/VRM system, currently defaults to 3D point cloud data 
        :type assembly_system['data_type']: str (required)

        :param assembly_system['application']: The application for which the libraray is being used, currently defaults to Inline Root Cause Analysis 
        :type assembly_system['application']: str (required)

        :param assembly_system['part_type']: The typre pf the part within the assembly, used to create the file structure within trained_models and active_learning 
        :type assembly_system['part_type']: str (required)

        :param assembly_system['part_name']: Name of the part, to be exteded to a list when used for multi-stage systems
        :type assembly_system['part_name']: str (required)

        :param assembly_system['data_format']: The data format currently defaults to complete, to be extended to partial data input refer: https://github.com/manojkumrb/spaceTimeV2 for more details 
        :type assembly_system['data_format']: str (required)

        :param assembly_system['assembly_type']: Type of assembly system, single stage vs multiple stages 
        :type assembly_system['assembly_type']: str (required)

        :param assembly_system['assembly_kccs']: The number of kccs in the assembly system used to determine the number of output neurons for the system
        :type assembly_system['assembly_kccs']: int (required)

        :param assembly_system['assembly_kpis']: the number of asembly KPIs in the system (the first KPI is convergency flag to incidate if the VRM simulation has converged), minimum value should be 1
        :type assembly_system['assembly_kpis']: int (required)

        :param assembly_system['voxel_dim']: The dimension/resolution of the voxel required to initlize the input to the model currently defaults to 64
        :type assembly_system['voxel_dim']: int (required)

        :param assembly_system['point_dim']: The number of nodes in the input mesh of the assembly 
        :type assembly_system['point_dim']: int (required)

        :param assembly_system['system_noise']: Noise parameter for the system, used to make model training more robust to actual system noise
        :type assembly_system['system_noise']: int (required)

        :param assembly_system['aritifical_noise']: Noise parameter for the model, used to make model training more robust to actual system noise (usually same as system noise)
        :type assembly_system['aritifical_noise']: int (required)

        :param assembly_system['noise_type']: The number of data channels/features in each voxel, currently defaults to 3 consiting of x_deviations, y_deviations and z_deviations 
        :type assembly_system['noise_type']: int (required)

        :param assembly_system['voxel_channels']: The number of data channels/features in each voxel, currently defaults to 3 consiting of x_deviations, y_deviations and z_deviations 
        :type assembly_system['voxel_channels']: int (required)
"""

assembly_system = {	
        'data_type': '3D Point Cloud Data',
        'application': 'Inline Root Cause Analysis',
        'part_type': 'Halo_debug_run',
        'part_name':'Halo',
        'data_format': 'Complete',
        'assembly_type':"Single-Stage",
        'assembly_kccs':3,
        'assembly_kpis':1,
        'voxel_dim':64,
        'point_dim':8047,
        'voxel_channels':3,
        'system_noise':0.0,
        'aritifical_noise':0.0,
        'noise_type':'uniform',
        'mapping_index':'Halo_64_voxel_mapping.dat',
        'nominal_cop_filename':'halo_nominal_cop.csv',
        'data_folder':'../datasets/halo_debug_run',
        'kcc_folder':'../active_learning/sample_input/halo_debug_run',
        'kcc_files':['input_X.csv'],
        'test_kcc_files':['test_input_X.csv'],
        'data_files_x':['test_output_table_x.csv'],
        'data_files_y':['test_output_table_y.csv'],
        'data_files_z':['test_output_table_z.csv'],
        'test_data_files_x':['output_table_x.csv'],
        'test_data_files_y':['output_table_y.csv'],
        'test_data_files_z':['output_table_z.csv'],
        }

#Assert that all config values conform to the libarary requirements
assert type(assembly_system['assembly_kccs']) is int, "Assembly KCCs is not an integer: %r" %assembly_system[assembly_kccs]
assert type(assembly_system['assembly_kpis']) is int, "Assembly KPIs is not an integer: %r" % assembly_system[assembly_kpis]
assert type(assembly_system['voxel_dim']) is int, "Voxel Dim is not an integer: %r" % assembly_system[voxel_dim]
assert type(assembly_system['point_dim']) is int, "Point Dim is not an integer: %r" % assembly_system[point_dim]
assert type(assembly_system['voxel_channels']) is int, "Voxel Channels is not an integer: %r" % assembly_system[voxel_channels]
assert type(assembly_system['system_noise']) is float, "Noise Level is not float: %r" % assembly_system[noise_levels]
