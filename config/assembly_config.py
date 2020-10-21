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
        'part_type': 'cross_member_assembly',
        'part_name':'cross_member',
        'data_format': 'Complete',
        'assembly_type':"multi-Stage",
        'assembly_stages':2,
        'assembly_kccs':148,
        'categorical_kccs':25,
        'assembly_kpis':1,
        'voxel_dim':64,
        'point_dim':11875,
        'voxel_channels':3,
        'system_noise':0.0,
        'aritifical_noise':0.0,
        'noise_type':'uniform',
        'mapping_index':'cross_member_64_voxel_mapping.dat',
        'nominal_cop_filename':'cross_member_nominal_cop.csv',
        'data_folder':'../datasets/cross_member_assembly',
        'kcc_folder':'../active_learning/sample_input/cross_member_assembly',
        'kcc_files':['AI_Input_Parameters_1.csv','AI_Input_Parameters_2.csv','AI_Input_Parameters_3.csv'],
        'test_kcc_files':['AI_Input_Parameters_test_1.csv'],
        'data_files_x':['DX_stage_13_hybrid_1.csv','DX_stage_13_hybrid_2.csv','DX_stage_13_hybrid_3.csv'],
        'data_files_y':['DY_stage_13_hybrid_1.csv','DY_stage_13_hybrid_2.csv','DY_stage_13_hybrid_3.csv'],
        'data_files_z':['DZ_stage_13_hybrid_1.csv','DZ_stage_13_hybrid_2.csv','DZ_stage_13_hybrid_3.csv'],
        'test_data_files_x':['DX_stage_13_hybrid_test_1.csv'],
        'test_data_files_y':['DY_stage_13_hybrid_test_1.csv'],
        'test_data_files_z':['DZ_stage_13_hybrid_test_1.csv'],
        }

encode_decode_construct = {     
        'input_data_files_x':['DX_stage_13_hybrid_1.csv','DX_stage_13_hybrid_2.csv','DX_stage_13_hybrid_3.csv'],
        'input_data_files_y':['DY_stage_13_hybrid_1.csv','DY_stage_13_hybrid_2.csv','DY_stage_13_hybrid_3.csv'],
        'input_data_files_z':['DZ_stage_13_hybrid_1.csv','DZ_stage_13_hybrid_2.csv','DZ_stage_13_hybrid_3.csv'],
        'input_test_data_files_x':['DX_stage_13_test_1.csv'],
        'input_test_data_files_y':['DY_stage_13_test_1.csv'],
        'input_test_data_files_z':['DZ_stage_13_test_1.csv'],

        'output_data_files_x':['DX_crossmember_test1_3.csv'],
        'output_data_files_y':['DY_crossmember_test1_3.csv'],
        'output_data_files_z':['DZ_crossmember_test1_3.csv'],
        'output_test_data_files_x':['DX_crossmember_test1_3.csv'],
        'output_test_data_files_y':['DY_crossmember_test1_3.csv'],
        'output_test_data_files_z':['DZ_crossmember_test1_3.csv'],
        }

encode_decode_multi_output_construct=[]

encode_decode_multi_output_construct.append({
        'stage_id':5,
        'output_data_files_x':['DX_stage_5_hybrid_1.csv','DX_stage_5_hybrid_2.csv','DX_stage_5_hybrid_3.csv'],
        'output_data_files_y':['DY_stage_5_hybrid_1.csv','DY_stage_5_hybrid_2.csv','DY_stage_5_hybrid_3.csv'],
        'output_data_files_z':['DZ_stage_5_hybrid_1.csv','DZ_stage_5_hybrid_2.csv','DZ_stage_5_hybrid_3.csv'],
        'output_test_data_files_x':['DX_stage_5_test_1.csv'],
        'output_test_data_files_y':['DY_stage_5_test_1.csv'],
        'output_test_data_files_z':['DZ_stage_5_test_1.csv'],      
        })

encode_decode_multi_output_construct.append({
        'stage_id':9,
        'output_data_files_x':['DX_stage_9_hybrid_1.csv','DX_stage_9_hybrid_2.csv','DX_stage_9_hybrid_3.csv'],
        'output_data_files_y':['DY_stage_9_hybrid_1.csv','DY_stage_9_hybrid_2.csv','DY_stage_9_hybrid_3.csv'],
        'output_data_files_z':['DZ_stage_9_hybrid_1.csv','DZ_stage_9_hybrid_2.csv','DZ_stage_9_hybrid_3.csv'],
        'output_test_data_files_x':['DX_stage_9_test_1.csv'],
        'output_test_data_files_y':['DY_stage_9_test_1.csv'],
        'output_test_data_files_z':['DZ_stage_9_test_1.csv'],         
        })

multi_stage_data_construct=[]

multi_stage_data_construct.append({'station_id':0,
                'stage_id':2,
                'stage_type':'positioning',#'clamping','fastening','release','non_ideal'
                'station_name':'pocket_rf_joining',
                'data_files_x':['DX_stage_13_hybrid_1.csv','DX_stage_13_hybrid_2.csv','DX_stage_13_hybrid_3.csv'],
                'data_files_y':['DY_stage_13_hybrid_1.csv','DY_stage_13_hybrid_2.csv','DY_stage_13_hybrid_3.csv'],
                'data_files_z':['DZ_stage_13_hybrid_1.csv','DZ_stage_13_hybrid_2.csv','DZ_stage_13_hybrid_3.csv'],
                'test_data_files_x':['DX_stage_13_test_1.csv'],
                'test_data_files_y':['DY_stage_13_test_1.csv'],
                'test_data_files_z':['DZ_stage_13_test_1.csv'],  
        })

multi_stage_data_construct.append({'station_id':2,
                'stage_id':10,
                'stage_type':'release',#'clamping','fastening','release','non_ideal'
                'station_name':'cross_member_pocket_joining',
                'data_files_x':['DX_stage_9_hybrid_1.csv','DX_stage_9_hybrid_2.csv','DX_stage_9_hybrid_3.csv'],
                'data_files_y':['DY_stage_9_hybrid_1.csv','DY_stage_9_hybrid_2.csv','DY_stage_9_hybrid_3.csv'],
                'data_files_z':['DZ_stage_9_hybrid_1.csv','DZ_stage_9_hybrid_2.csv','DZ_stage_9_hybrid_3.csv'],
                'test_data_files_x':['DX_stage_9_test_1.csv'],
                'test_data_files_y':['DY_stage_9_test_1.csv'],
                'test_data_files_z':['DZ_stage_9_test_1.csv'],  
        })

multi_stage_sensor_config = {   
        'eval_metric': 'R2',#MSE, RMSE, R2
        'eval_metric_threshold' :0.95, #mm
        'max_stages':4,
        'inital_stage_list':[10],
        }

multi_stage_sensor_construct=[]

multi_stage_sensor_construct.append({'station_id':0,
                'stage_id':0,
                'stage_type':'positioning',#'clamping','fastening','release'
                'station_name':'pocket_rf_joining',
                'process_param_ids':[5,6,7],
                'data_files_x':['DX_crossmember_1.csv'],
                'data_files_y':['DY_crossmember_1.csv'],
                'data_files_z':['DZ_crossmember_1.csv'],
                'test_data_files_x':['DX_crossmember_test_1.csv'],
                'test_data_files_y':['DY_crossmember_test_1.csv'],
                'test_data_files_z':['DZ_crossmember_test_1.csv'],  
        })

multi_stage_sensor_construct.append({'station_id':0,
                'stage_id':1,
                'stage_type':'fastening',#'clamping','fastening','release'
                'station_name':'pocket_rf_joining',
                'process_param_ids':[],
                'data_files_x':['DX_crossmember_2.csv'],
                'data_files_y':['DY_crossmember_2.csv'],
                'data_files_z':['DZ_crossmember_2.csv'],
                'test_data_files_x':['DX_crossmember_test_2.csv'],
                'test_data_files_y':['DY_crossmember_test_2.csv'],
                'test_data_files_z':['DZ_crossmember_test_2.csv'],  
        })

multi_stage_sensor_construct.append({'station_id':0,
                'stage_id':2,
                'stage_type':'release',#'clamping','fastening','release'
                'station_name':'pocket_rf_joining',
                'process_param_ids':[],
                'data_files_x':['DX_crossmember_3.csv'],
                'data_files_y':['DY_crossmember_3.csv'],
                'data_files_z':['DZ_crossmember_3.csv'],
                'test_data_files_x':['DX_crossmember_test_3.csv'],
                'test_data_files_y':['DY_crossmember_test_3.csv'],
                'test_data_files_z':['DZ_crossmember_test_3.csv'],  
        })

multi_stage_sensor_construct.append({'station_id':1,
                'stage_id':3,
                'stage_type':'non_ideal',#'clamping','fastening','release'
                'station_name':'cross_member_rf_joining',
                'process_param_ids':[0,1,2,3],
                'data_files_x':['DX_crossmember_4.csv'],
                'data_files_y':['DY_crossmember_4.csv'],
                'data_files_z':['DZ_crossmember_4.csv'],
                'test_data_files_x':['DX_crossmember_test_4.csv'],
                'test_data_files_y':['DY_crossmember_test_4.csv'],
                'test_data_files_z':['DZ_crossmember_test_4.csv'],  
        })

multi_stage_sensor_construct.append({'station_id':1,
                'stage_id':4,
                'stage_type':'positioning',#'clamping','fastening','release'
                'station_name':'cross_member_rf_joining',
                'process_param_ids':[8,9,10,11],
                'data_files_x':['DX_crossmember_5.csv'],
                'data_files_y':['DY_crossmember_5.csv'],
                'data_files_z':['DZ_crossmember_5.csv'],
                'test_data_files_x':['DX_crossmember_test_5.csv'],
                'test_data_files_y':['DY_crossmember_test_5.csv'],
                'test_data_files_z':['DZ_crossmember_test_5.csv'],  
        })

multi_stage_sensor_construct.append({'station_id':1,
                'stage_id':5,
                'stage_type':'fastening',#'clamping','fastening','release'
                'station_name':'cross_member_rf_joining',
                'process_param_ids':[],
                'data_files_x':['DX_crossmember_6.csv'],
                'data_files_y':['DY_crossmember_6.csv'],
                'data_files_z':['DZ_crossmember_6.csv'],
                'test_data_files_x':['DX_crossmember_test_6.csv'],
                'test_data_files_y':['DY_crossmember_test_6.csv'],
                'test_data_files_z':['DZ_crossmember_test_6.csv'],  
        })

multi_stage_sensor_construct.append({'station_id':1,
                'stage_id':6,
                'stage_type':'release',#'clamping','fastening','release'
                'station_name':'cross_member_rf_joining',
                'process_param_ids':[],
                'data_files_x':['DX_crossmember_7.csv'],
                'data_files_y':['DY_crossmember_7.csv'],
                'data_files_z':['DZ_crossmember_7.csv'],
                'test_data_files_x':['DX_crossmember_test_7.csv'],
                'test_data_files_y':['DY_crossmember_test_7.csv'],
                'test_data_files_z':['DZ_crossmember_test_7.csv'],  
        })
multi_stage_sensor_construct.append({'station_id':2,
                'stage_id':7,
                'stage_type':'positioning',#'clamping','fastening','release'
                'station_name':'cross_member_pocket_joining',
                'process_param_ids':[4],
                'data_files_x':['DX_crossmember_8.csv'],
                'data_files_y':['DY_crossmember_8.csv'],
                'data_files_z':['DZ_crossmember_8.csv'],
                'test_data_files_x':['DX_crossmember_test_8.csv'],
                'test_data_files_y':['DY_crossmember_test_8.csv'],
                'test_data_files_z':['DZ_crossmember_test_8.csv'],  
        })
multi_stage_sensor_construct.append({'station_id':2,
                'stage_id':8,
                'stage_type':'clamping',#'clamping','fastening','release'
                'station_name':'cross_member_pocket_joining',
                'process_param_ids':[],
                'data_files_x':['DX_crossmember_9.csv'],
                'data_files_y':['DY_crossmember_9.csv'],
                'data_files_z':['DZ_crossmember_9.csv'],
                'test_data_files_x':['DX_crossmember_test_9.csv'],
                'test_data_files_y':['DY_crossmember_test_9.csv'],
                'test_data_files_z':['DZ_crossmember_test_9.csv'],  
        })
multi_stage_sensor_construct.append({'station_id':2,
                'stage_id':9,
                'stage_type':'fastening',#'clamping','fastening','release'
                'station_name':'cross_member_pocket_joining',
                'process_param_ids':[],
                'data_files_x':['DX_crossmember_10.csv'],
                'data_files_y':['DY_crossmember_10.csv'],
                'data_files_z':['DZ_crossmember_10.csv'],
                'test_data_files_x':['DX_crossmember_test_10.csv'],
                'test_data_files_y':['DY_crossmember_test_10.csv'],
                'test_data_files_z':['DZ_crossmember_test_10.csv'],  
        })
multi_stage_sensor_construct.append({'station_id':2,
                'stage_id':10,
                'stage_type':'release',#'clamping','fastening','release'
                'station_name':'cross_member_pocket_joining',
                'process_param_ids':[],
                'data_files_x':['DX_crossmember_11.csv'],
                'data_files_y':['DY_crossmember_11.csv'],
                'data_files_z':['DZ_crossmember_11.csv'],
                'test_data_files_x':['DX_crossmember_test_11.csv'],
                'test_data_files_y':['DY_crossmember_test_11.csv'],
                'test_data_files_z':['DZ_crossmember_test_11.csv'],  
        })




#Assert that all config values conform to the library requirements
assert type(assembly_system['assembly_kccs']) is int, "Assembly KCCs is not an integer: %r" %assembly_system[assembly_kccs]
assert type(assembly_system['assembly_kpis']) is int, "Assembly KPIs is not an integer: %r" % assembly_system[assembly_kpis]
assert type(assembly_system['voxel_dim']) is int, "Voxel Dim is not an integer: %r" % assembly_system[voxel_dim]
assert type(assembly_system['point_dim']) is int, "Point Dim is not an integer: %r" % assembly_system[point_dim]
assert type(assembly_system['voxel_channels']) is int, "Voxel Channels is not an integer: %r" % assembly_system[voxel_channels]
assert type(assembly_system['system_noise']) is float, "Noise Level is not float: %r" % assembly_system[noise_levels]
