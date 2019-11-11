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
        'noise_type':'uniform'
        }

preprocessing_queue = [preprocessing.scale_and_center,
                       preprocessing.dot_reduction,
                       preprocessing.connect_lines]

use_anonymous = True