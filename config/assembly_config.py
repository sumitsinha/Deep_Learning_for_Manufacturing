"""The assembly config file required to initialize the assembly class and perform model training and deployment

        :param assembly_system['data_type']: The incoming data type from the assembly/VRM system, currently defaults to 3D point cloud data 
        :type assembly_system['data_type']: str (required)

        :param assembly_system['application']: The application for which the library is being used, currently defaults to In line Root Cause Analysis 
        :type assembly_system['application']: str (required)

        :param assembly_system['part_type']: The type of the part within the assembly, used to create the file structure within trained_models and active_learning 
        :type assembly_system['part_type']: str (required)

        :param assembly_system['part_name']: Name of the part, to be extended to a list when used for multi-stage systems
        :type assembly_system['part_name']: str (required)

        :param assembly_system['data_format']: The data format currently defaults to complete, to be extended to partial data input refer: https://github.com/manojkumrb/spaceTimeV2 for more details 
        :type assembly_system['data_format']: str (required)

        :param assembly_system['assembly_type']: Type of assembly system, single stage vs multiple stages 
        :type assembly_system['assembly_type']: str (required)

        :param assembly_system['assembly_kccs']: The number of kccs in the assembly system used to determine the number of output neurons for the system
        :type assembly_system['assembly_kccs']: int (required)

        :param assembly_system['assembly_kpis']: the number of assembly KPIs in the system (the first KPI is convergence flag to indicate if the VRM simulation has converged), minimum value should be 1
        :type assembly_system['assembly_kpis']: int (required)

        :param assembly_system['voxel_dim']: The dimension/resolution of the voxel required to initialize the input to the model currently defaults to 64
        :type assembly_system['voxel_dim']: int (required)

        :param assembly_system['point_dim']: The number of nodes in the input mesh of the assembly 
        :type assembly_system['point_dim']: int (required)

        :param assembly_system['system_noise']: Noise parameter for the system, used to make model training more robust to actual system noise
        :type assembly_system['system_noise']: int (required)

        :param assembly_system['aritifical_noise']: Noise parameter for the model, used to make model training more robust to actual system noise (usually same as system noise)
        :type assembly_system['aritifical_noise']: int (required)

        :param assembly_system['noise_type']: The distribution of the noise , defaults to uniform random value between +- system noise, in case of Gaussian it corresponds to the standard deviation (mean is zero)
        :type assembly_system['noise_type']: float (required)

        :param assembly_system['mapping_index']: File name of the mapping index, after download is complete the file is saved with this name
        :type assembly_system['mapping_index']: str (required)

        :param assembly_system['nominal_cop_filename']: File name of the nominal cop, after download is complete the file is saved with this name
        :type assembly_system['nominal_cop_filename']: str (required)

        :param assembly_system['data_folder']: Path to input data location
        :type assembly_system['data_folder']: str (required)

        :param assembly_system['kcc_folder']: Path to Output data/KCC location
        :type assembly_system['kcc_folder']: str (required)

        :param assembly_system['kcc_files']: List of kcc files, after download is complete the file is saved with this name
        :type assembly_system['kcc_files']: str (required)

        :param assembly_system['test_kcc_files']: List of test kcc files, after download is complete the file is saved with this name
        :type assembly_system['test_kcc_files']: str (required)

        :param assembly_system['data_files_x']: List of x deviation input files, after download is complete the file is saved with this name
        :type assembly_system['data_files_x']: str (required)

        :param assembly_system['data_files_y']: List of y deviation input files, after download is complete the file is saved with this name
        :type assembly_system['data_files_y']: str (required)

        :param assembly_system['data_files_z']: List of z deviation input files, after download is complete the file is saved with this name
        :type assembly_system['data_files_z']: str (required)

        :param assembly_system['test_data_files_x']: List of x deviation test input files, after download is complete the file is saved with this name
        :type assembly_system['test_data_files_x']: str (required)

        :param assembly_system['test_data_files_y']: List of y deviation test input files, after download is complete the file is saved with this name
        :type assembly_system['test_data_files_y']: str (required)

        :param assembly_system['test_data_files_z']: List of z deviation test input files, after download is complete the file is saved with this name
        :type assembly_system['test_data_files_z']: str (required)
"""

assembly_system = {	
        'data_type': '3D Point Cloud Data',
        'application': 'InLine Root Cause Analysis',
        'part_type': 'inner_rf_assembly',
        'part_name':'inner_rf',
        'data_format': 'Complete',
        'assembly_type':"multi-Stage",
        'assembly_kccs':6,
        'assembly_kpis':1,
        'voxel_dim':64,
        'point_dim':10841,
        'voxel_channels':3,
        'system_noise':0.0,
        'aritifical_noise':0.0,
        'noise_type':'uniform',
        'mapping_index':'inner_rf_64_voxel_mapping.dat',
        'nominal_cop_filename':'inner_rf_nominal_cop.csv',
        'data_folder':'../datasets/inner_rf_assembly',
        'kcc_folder':'../active_learning/sample_input/inner_rf_assembly',
        'kcc_files':['inner_rf_samples_datagen1_set.csv'],
        'test_kcc_files':['inner_rf_samples_datagen10_gui_demo_test.csv'],
        'data_files_x':['VRM-m-Stageoutput_table_x_datagen1_3.csv'],
        'data_files_y':['VRM-m-Stageoutput_table_y_datagen1_3.csv'],
        'data_files_z':['VRM-m-Stageoutput_table_z_datagen1_3.csv'],
        'test_data_files_x':['output_table_x_gui_demo10_test_3.csv'],
        'test_data_files_y':['output_table_y_gui_demo10_test_3.csv'],
        'test_data_files_z':['output_table_z_gui_demo10_test_3.csv'],
        }

#Assert that all config values conform to the library requirements
assert type(assembly_system['assembly_kccs']) is int, "Assembly KCCs is not an integer: %r" %assembly_system[assembly_kccs]
assert type(assembly_system['assembly_kpis']) is int, "Assembly KPIs is not an integer: %r" % assembly_system[assembly_kpis]
assert type(assembly_system['voxel_dim']) is int, "Voxel Dim is not an integer: %r" % assembly_system[voxel_dim]
assert type(assembly_system['point_dim']) is int, "Point Dim is not an integer: %r" % assembly_system[point_dim]
assert type(assembly_system['voxel_channels']) is int, "Voxel Channels is not an integer: %r" % assembly_system[voxel_channels]
assert type(assembly_system['system_noise']) is float, "Noise Level is not float: %r" % assembly_system[noise_levels]
