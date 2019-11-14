
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import xgboost as xgb
from sklearn importRandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor

def benchmarking_models(bench_mark='all'):

	bn_models=[None]*20
	bn_models_name=[None]*20
	model_name = type(model).__name__
	#Add upto 20 models to bench mark with
	bn_models[0]=MultiOutputRegressor(xgb.XGBRegressor(colsample_bytree=0.4,gamma=0.045,learning_rate=0.07,max_depth=500,min_child_weight=1.5,n_estimators=500,reg_alpha=0.65,reg_lambda=0.45,subsample=0.95,n_jobs=-1,verbose=True))
	bn_models_name[0] = type(bn_models[0]).__name__
	bn_models[1]=MultiOutputRegressor(RandomForestRegressor(n_estimators=1000,max_depth=50,n_jobs=-1,verbose=True))
	bn_models_name[1] = type(bn_models[1]).__name__
	bn_models[2]=MultiOutputRegressor(SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1))
	bn_models_name[2] = type(bn_models[2]).__name__
	bn_models[3]=MLPRegressor(hidden_layer_sizes=(512,256,),  activation='relu', solver='adam',    alpha=0.001,batch_size='auto',
               learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
               random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
               nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
               epsilon=1e-08)
	bn_models_name[3] = type(bn_models[3]).__name__
	bn_models[4]=MultiOutputRegressor(DecisionTreeRegressor(max_depth=10))
	bn_models_name[4] = type(bn_models[4]).__name__
	bn_models[5]=MultiOutputRegressor(Ridge(alpha=0.1))
	bn_models_name[5] = type(bn_models[5]).__name__
	bn_models[6]=MultiOutputRegressor(Lasso(alpha=0.1))
	bn_models_name[6] = type(bn_models[6]).__name__

	bn_models=[x for x in bn_models if x is not None]
	bn_models_name=[x for x in bn_models_name if x is not None]
	return bn_models,bn_models_name

def benchmarking_models_eval(bn_models,point_data,kcc_data,metrics_eval):

	
	X_train, X_test, y_train, y_test = train_test_split(point_data, kcc_data, test_size = 0.2)
	
	
	bnoutput_mae=np.zeroes(len(bn_models),metrics_eval.assembly_kccs)
	bnoutput_mse=np.zeroes(len(bn_models),metrics_eval.assembly_kccs)
	bnoutput_rmse=np.zeroes(len(bn_models),metrics_eval.assembly_kccs)
	bnoutput_r2=np.zeroes(len(bn_models),metrics_eval.assembly_kccs)

	for i in range(len(bn_models)):
		model=bn_models[i]
		model.fit(X_train,y_train)
		y_pred=model.predict(X_test,y_test)
		eval_metrics=metrics_eval.metrics_eval(y_pred,y_test)
    	bnoutput_mae[i,:]=eval_metrics["Mean Absolute Error"]
    	bnoutput_mse[i,:]=eval_metrics["Mean Squared Error"]
    	bnoutput_rmse[i,:]=eval_metrics["Root Mean Squared Error"]
    	bnoutput_r2[i,:]=eval_metrics["R Squared"]

    bneval_metrics= {
			"Mean Absolute Error" : bnoutput_mae,
			"Mean Squared Error" : bnoutput_mse,
			"Root Mean Squared Error" : bnoutput_rmse,
			"R Squared" : bnoutput_r2
		}
	return bneval_metrics

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
	metrics_eval=-MetricsEval(vrm_system)
	print('Importing and preprocessing Cloud-of-Point Data')
	
	file_names=['car_halo_run1_ydev.csv','car_halo_run2_ydev.csv','car_halo_run3_ydev.csv','car_halo_run4_ydev.csv','car_halo_run5_ydev.csv']
	get_train_data=GetTrainData(vrm_system)
	dataset=get_train_data.data_import(file_names)

	point_data=dataset[:, 0:point_dim]
	kcc_data=dataset[:,point_dim:]
	bn_models=benchmarking_models(point_data,kcc_data,bench_mark='all')
    number_of_runs=20
    print('Benchmarking for all Algorithims')
    bn_models,bn_models_name=benchmarking_models()

    mr_bnoutput_mae=np.zeroes(number_of_runs,len(bn_models),metrics_eval.assembly_kccs)
	mr_bnoutput_mse=np.zeroes(number_of_runs,len(bn_models),metrics_eval.assembly_kccs)
	mr_bnoutput_rmse=np.zeroes(number_of_runs,len(bn_models),metrics_eval.assembly_kccs)
	mr_bnoutput_r2=np.zeroes(number_of_runs,len(bn_models),metrics_eval.assembly_kccs)

	for i in range(number_of_runs):
		bneval_metrics=benchmarking_models(bn_models,point_data,kcc_data,metrics_eval)
		mr_bnoutput_mae[i,:,:]=bneval_metrics["Mean Absolute Error"]
		mr_bnoutput_mse[i,:,:]=bneval_metrics["Mean Squared Error"]
    	mr_bnoutput_rmse[i,:,:]=bneval_metrics["Root Mean Squared Error"]
    	mr_bnoutput_r2[i,:,:]=bneval_metrics["R Squared"]
        

   avg_mrbn_mae=np.mean(mr_bnoutput_mae,axis=0)
   avg_mrbn_mse=np.mean(mr_bnoutput_mse,axis=0)
   avg_mrbn_rmse=np.mean(mr_bnoutput_rmse,axis=0)
   avg_mrbn_r2=np.mean(mr_bnoutput_r2,axis=0)

   std_mr_bn_mae=np.std(mr_bnoutput_mae,axis=0, ddof=1)
   std_mr_bn_mse=np.std(mr_bnoutput_mse,axis=0, ddof=1)
   std_mr_bn_rmse=np.std(mr_bnoutput_rmse,axis=0, ddof=1)
   std_mr_bn_r2=np.std(mr_bnoutput_r2,axis=0, ddof=1)