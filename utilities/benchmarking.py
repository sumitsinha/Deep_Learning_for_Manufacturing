
from assemblyconfig import assembly_systemf
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

import xgboost as xgb
from sklearn importRandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor

def benchmarking_models(max_models):

	#Add upto benchmark_max in 
	bn_models=[None]*benchmark_max
	bn_models_name=[None]*benchmark_max

	
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

	parser = argparse.ArgumentParser(description="Arguments for Benchmarking against")
	parser.add_argument("-N", "--number_of_runs", help = "Number of Benchmarking Runs", required = False, default = 20,type=int)
	parser.add_argument("-M", "--max_models", help = "Maximum number of Models to Run Benchmarking", required = False, default = 10,type=int)
	argument = parser.parse_args()
	
	number_of_runs=argument.number_of_runs
	max_models=argument.max_models

	data_type=assembly_system['data_type']
	application=assembly_system['application']
	part_type=assembly_system['part_type']
	data_format=assembly_system['data_format']
	assembly_type=assembly_system['assembly_type']
	assembly_kccs=assembly_system['assembly_kccs']	
	assembly_kpis=assembly_system['assembly_kpis']
	voxel_dim=assembly_system['voxel_dim']
	point_dim=assembly_system['point_dim']
	voxel_channels=assembly_system['voxel_channels']
	noise_levels=assembly_system['noise_levels']
	noise_type=assembly_system['noise_type']

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
	
	print('Benchmarking for all Algorithims')
	bn_models,bn_models_name=benchmarking_models(max_models)

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

   avg_kcc_mrbn_mae=np.mean(avg_mrbn_mae,axis=1)
   avg_kcc_mrbn_mse=np.mean(avg_mrbn_mse,axis=1)
   avg_kcc_mrbn_rmse=np.mean(avg_mrbn_rmse,axis=1)
   avg_kcc_mrbn_mae=np.mean(avg_mrbn_r2,axis=1)

   avg_std_kcc_mae=np.mean(std_mr_bn_mae,axis=1)
   avg_std_kcc_mse=np.mean(std_mr_bn_mse,axis=1)
   avg_std_kcc_mae=np.mean(std_mr_bn_rmse,axis=1)
   avg_std_kcc_mae=np.mean(std_mr_bn_r2,axis=1)

   print("Average performance considering all KCCs")
   for i in range(len(bn_models_name)):
		print(f'Name model: {bn_models_name[i]} ')
		print(f'MAE: {avg_kcc_mrbn_mae[i]}, MAE Standard Deviation: {avg_std_kcc_mae[i]}')
		print(f'MSE: {avg_kcc_mrbn_mse[i]}, MSE Standard Deviation: {avg_std_kcc_mse[i]}')
		print(f'RMSE: {avg_kcc_mrbn_rmse[i]}, RMSE Standard Deviation: {avg_std_kcc_rmse[i]}')
		print(f'R2: {avg_kcc_mrbn_mae[i]}, R2 Standard Deviation: {avg_std_kcc_mae[i]}')



