
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import xgboost as xgb
from sklearn importRandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

def benchmarking_models(bench_mark='all',model_type=0):

	bn_models=[None]*10

	bn_models[0]=MultiOutputRegressor(xgb.XGBRegressor(colsample_bytree=0.4,gamma=0.045,learning_rate=0.07,max_depth=500,min_child_weight=1.5,n_estimators=500,reg_alpha=0.65,reg_lambda=0.45,subsample=0.95,n_jobs=-1,verbose=True))
	bn_models[1]=MultiOutputRegressor(RandomForestRegressor(n_estimators=1000,max_depth=50,n_jobs=-1,verbose=True))
	bn_models[2]=MultiOutputRegressor(SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1))
	bn_models[3]=MLPRegressor(hidden_layer_sizes=(10,),  activation='relu', solver='adam',    alpha=0.001,batch_size='auto',
               learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
               random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
               nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
               epsilon=1e-08)
	bn_models[4]=
	bn_models[5]=
	bn_models[6]=

	return bn_models
if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Arguments to initiate Measurement System Class and Assembly System Class")
    parser.add_argument("-D", "--data_type", help = "Example: 3D Point Cloud Data", required = False, default = "3D Point Cloud Data")
    parser.add_argument("-A", "--application", help = "Example: Inline Root Cause Analysis", required = False, default = "Inline Root Cause Analysis")
    parser.add_argument("-P", "--part_type", help = "Example: Door Inner and Hinge Assembly", required = False, default = "Door Inner and Hinge Assembly")
    parser.add_argument("-F", "--data_format", help = "Example: Complete vs Partial Data", required = False, default = "Complete")
	parser.add_argument("-S", "--assembly_type", help = "Example: Multi-Stage vs Single-Stage", required = False, default = "Single-Stage")
    parser.add_argument("-C", "--assembly_kccs", help = "Number of KCCs for the Assembly", required = False, default =15,type=int )
    parser.add_argument("-I", "--assembly_kpis	", help = "Number of KPIs for the Assembly", required = False, default = 6,type=int)
    parser.add_argument("-V", "--voxel_dim", help = "The Granularity of Voxels - 32 64 128", required = False, default = 64,type=int)
    parser.add_argument("-P", "--point_dim", help = "Number of key Nodes", required = True, type=int)
    parser.add_argument("-C", "--voxel_channels", help = "Number of Channels - 1 or 3", required = False, default = 1,type=int)
    parser.add_argument("-N", "--noise_levels", help = "Amount of Artificial Noise to add while training", required = False, default = 0.1,type=float)
    parser.add_argument("-T", "--noise_type", help = "Type of noise to be added uniform/Gaussian default uniform", required = False, default = "uniform")
	argument = parser.parse_args()
	
	data_type=argument.data_type
	application=argument.application
	part_type=argument.part_type
	data_format=argument.data_format
	assembly_type=argument.assembly_type	
	assembly_kccs=argument.assembly_kccs	
	assembly_kpis=argument.assembly_kpis
	voxel_dim=argument.voxel_dim
	point_dim=argument.point_dim
	voxel_channels=argument.voxel_channels
	noise_levels=argument.noise_levels
	noise_type=argument.noise_type

	#Objects of Measurement System and Assembly System
	measurement_system=HexagonWlsScanner(data_type,application, system_noise,part_type,data_format)
	vrm_system=VRMSimulationModel(assembly_type,assembly_kccs,assembly_kpis,voxel_dim,point_dim,voxel_channels,noise_levels,noise_type)

	print('Importing and preprocessing Cloud-of-Point Data')
	
	file_names=['car_halo_run1_ydev.csv','car_halo_run2_ydev.csv','car_halo_run3_ydev.csv','car_halo_run4_ydev.csv','car_halo_run5_ydev.csv']
	get_train_data=GetTrainData(vrm_system)
	dataset=get_train_data.data_import(file_names)

	kcc_id=[]
	point_data=dataset[:, 0:point_dim]
	regr = MultiOutputRegressor(knn)
    print('Benchmarking for all Algorithims')
	for i in range(benchmarking_algo):
		kcc_name="KCC_"+str(i+1)
		kcc_id.append(kcc_name)
    	selected_kcc=dataset[:,point_dim:]
        

   