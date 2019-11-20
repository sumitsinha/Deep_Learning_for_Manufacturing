
assembly_system = {
		'data_type': '3D Point Cloud Data',
        'application': 'Inline Root Cause Analysis',
        'part_type': 'Halo Stamping Patterns',
        'part_name':'Halo',
        'data_format': 'Complete',
        'assembly_type':"Single-Stage",
        'assembly_kccs':5,
        'assembly_kpis':0,
        'voxel_dim':64,
        'point_dim':8047,
        'voxel_channels':1,
        'system_noise':0.1,
        'aritifical_noise':0.1,
        'noise_type':'uniform',
        'mapping_index':'Halo_cov_index_data_64.dat',
        'data_files':['car_halo_run1_ydev.csv']
        }


assert type(assembly_system['assembly_kccs']) is int, "Assembly KCCs is not an integer: %r" %assembly_system[assembly_kccs]
assert type(assembly_system['assembly_kpis']) is int, "Assembly KPIs is not an integer: %r" % assembly_system[assembly_kpis]
assert type(assembly_system['voxel_dim']) is int, "Voxel Dim is not an integer: %r" % assembly_system[voxel_dim]
assert type(assembly_system['point_dim']) is int, "Point Dim is not an integer: %r" % assembly_system[point_dim]
assert assembly_system['voxel_channels'] == 1, "Voxel Channels is not an 1 (Multi-Channel Support to be added): %r" % assembly_system[voxel_channels]
assert type(assembly_system['system_noise']) is float, "Noise Level is not float: %r" % assembly_system[noise_levels]
