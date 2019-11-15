import preprocessing

assembly_system = {
		'data_type': '3D Point Cloud Data',
        'application': 'Inline Root Cause Analysis',
        'part_type': 'Door Inner and Hinge Assembly',
        'data_format': 'Complete',
        'assembly_type':"Single-Stage",
        'assembly_kccs':15,
        'assembly_kpis':6,
        'voxel_dim':64,
        'point_dim':8047,
        'voxel_channels':1,
        'noise_levels':0.1,
        'noise_type':'uniform',
        'mapping_index':'index_conv'
        'data_files':['car_halo_run1_ydev.csv','car_halo_run2_ydev.csv','car_halo_run3_ydev.csv','car_halo_run4_ydev.csv','car_halo_run5_ydev.csv']
        }

preprocessing_queue = [preprocessing.scale_and_center,
                       preprocessing.dot_reduction,
                       preprocessing.connect_lines]
assert type(assembly_system[assembly_kccs]) is IntType "Assembly KCCs is not an integer: %r" % assembly_system[assembly_kccs]
assert type(assembly_system[assembly_kpis]) is IntType "Assembly KPIs is not an integer: %r" % assembly_system[assembly_kpis]
assert type(assembly_system[voxel_dim]) is IntType "Voxel Dim is not an integer: %r" % assembly_system[voxel_dim]
assert type(assembly_system[point_dim]) is IntType "Point Dim is not an integer: %r" % assembly_system[point_dim]
assert assembly_system[voxel_channels] == 1 "Voxel Channels is not an 1 (Multi-Channel Support to be added): %r" % assembly_system[voxel_channels]
assert type(assembly_system[noise_levels]) is FloatType "Noise Level is not float: %r" % assembly_system[noise_levels]


use_anonymous = True