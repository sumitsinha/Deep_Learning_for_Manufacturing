import os
import sys
import pathlib
current_path=os.path.dirname(__file__)
parentdir = os.path.dirname(current_path)

#Adding Path to various Modules
sys.path.append("../core")
sys.path.append("../visualization")
sys.path.append("../utilities")
sys.path.append("../datasets")
sys.path.append("../trained_models")
sys.path.append("../config")

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor

#Importing Config files
import assembly_config as config
import model_config as cftrain

#Importing required modules from the package
from measurement_system import HexagonWlsScanner
from assembly_system import VRMSimulationModel
from data_import import GetTrainData
from metrics_eval import MetricsEval

def benchmarking_models(max_models):

	#Add upto benchmark_max in 
	bn_models=[None]*max_models
	bn_models_name=[None]*max_models

	bn_models[0]=MultiOutputRegressor(estimator=xgb.XGBRegressor(colsample_bytree=0.4,gamma=0.045,learning_rate=0.07,max_depth=500,min_child_weight=1.5,n_estimators=500,reg_alpha=0.65,reg_lambda=0.45,subsample=0.95,n_jobs=-1,verbose=True))
	bn_models_name[0] = type(bn_models[0].estimator).__name__
	bn_models[1]=MultiOutputRegressor(estimator=RandomForestRegressor(n_estimators=1000,max_depth=50,n_jobs=-1,verbose=True))
	bn_models_name[1] = type(bn_models[1].estimator).__name__
	bn_models[2]=MultiOutputRegressor(estimator=SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1))
	bn_models_name[2] = type(bn_models[2].estimator).__name__
	bn_models[3]=MLPRegressor(hidden_layer_sizes=(512,256,),  activation='relu', solver='adam',    alpha=0.001,batch_size='auto',
			   learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
			   random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
			   nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
			   epsilon=1e-08)

	bn_models_name[3] = type(bn_models[3]).__name__
	bn_models[4]=MultiOutputRegressor(estimator=DecisionTreeRegressor(max_depth=10))
	bn_models_name[4] = type(bn_models[4].estimator).__name__
	bn_models[5]=MultiOutputRegressor(estimator=Ridge(alpha=0.1))
	bn_models_name[5] = type(bn_models[5].estimator).__name__
	bn_models[6]=MultiOutputRegressor(estimator=Lasso(alpha=0.1))
	bn_models_name[6] = type(bn_models[6].estimator).__name__

	bn_models=[x for x in bn_models if x is not None]
	bn_models_name=[x for x in bn_models_name if x is not None]
	return bn_models,bn_models_name

def benchmarking_models_eval(bn_models,point_data,kcc_dataset,assembly_kccs,bm_path,test_size = 0.2):

	X_train, X_test, y_train, y_test = train_test_split(point_data, kcc_dataset, test_size=test_size)
	bnoutput_mae=np.zeros((len(bn_models),assembly_kccs))
	bnoutput_mse=np.zeros((len(bn_models),assembly_kccs))
	bnoutput_rmse=np.zeros((len(bn_models),assembly_kccs))
	bnoutput_r2=np.zeros((len(bn_models),assembly_kccs))

	for i in range(len(bn_models)):
		model=bn_models[i]
		model.fit(X_train,y_train)
		y_pred=model.predict(X_test)
		eval_metrics,accuracy_metrics_df=metrics_eval.metrics_eval_base(y_pred,y_test,bm_path)
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

	print('Parsing from Assembly Config File....')

	data_type=config.assembly_system['data_type']
	application=config.assembly_system['application']
	part_type=config.assembly_system['part_type']
	part_name=config.assembly_system['part_name']
	data_format=config.assembly_system['data_format']
	assembly_type=config.assembly_system['assembly_type']
	assembly_kccs=config.assembly_system['assembly_kccs']   
	assembly_kpis=config.assembly_system['assembly_kpis']
	voxel_dim=config.assembly_system['voxel_dim']
	point_dim=config.assembly_system['point_dim']
	voxel_channels=config.assembly_system['voxel_channels']
	noise_type=config.assembly_system['noise_type']
	mapping_index=config.assembly_system['mapping_index']
	file_names_x=config.assembly_system['data_files_x']
	file_names_y=config.assembly_system['data_files_y']
	file_names_z=config.assembly_system['data_files_z']
	system_noise=config.assembly_system['system_noise']
	aritifical_noise=config.assembly_system['aritifical_noise']
	data_folder=config.assembly_system['data_folder']
	kcc_folder=config.assembly_system['kcc_folder']
	kcc_files=config.assembly_system['kcc_files']

	print('Parsing from Training Config File')

	max_models=cftrain.bm_params['max_models']
	runs=cftrain.bm_params['runs']
   
	print('Creating file Structure....')
	
	folder_name=part_type
	train_path='../trained_models/'+part_type
	pathlib.Path(train_path).mkdir(parents=True, exist_ok=True)

	bm_path=train_path+'/benchmarking'
	pathlib.Path(bm_path).mkdir(parents=True, exist_ok=True)

	print('Intilizing the Assembly System and Measurement System....')
	
	measurement_system=HexagonWlsScanner(data_type,application,system_noise,part_type,data_format)
	vrm_system=VRMSimulationModel(assembly_type,assembly_kccs,assembly_kpis,part_name,part_type,voxel_dim,voxel_channels,point_dim,aritifical_noise)
	get_data=GetTrainData();
	metrics_eval=MetricsEval();
	
	print('Importing and preprocessing Cloud-of-Point Data')
	
	dataset=[]
	dataset.append((get_data.data_import(file_names_x,data_folder)).iloc[:,0:point_dim])
	dataset.append((get_data.data_import(file_names_y,data_folder)).iloc[:,0:point_dim])
	dataset.append((get_data.data_import(file_names_z,data_folder)).iloc[:,0:point_dim])
	
	kcc_dataset=get_data.data_import(kcc_files,kcc_folder)
	point_data=pd.concat([dataset[0],dataset[1],dataset[2]],axis=1,ignore_index=True)

	print('Benchmarking for all Algorithims')
	bn_models,bn_models_name=benchmarking_models(max_models)

	mr_bnoutput_mae=np.zeros((runs,len(bn_models),assembly_kccs))
	mr_bnoutput_mse=np.zeros((runs,len(bn_models),assembly_kccs))
	mr_bnoutput_rmse=np.zeros((runs,len(bn_models),assembly_kccs))
	mr_bnoutput_r2=np.zeros((runs,len(bn_models),assembly_kccs))

	for i in range(runs):
		bneval_metrics=benchmarking_models_eval(bn_models,point_data,kcc_dataset,assembly_kccs,bm_path)
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
	avg_kcc_mrbn_r2=np.mean(avg_mrbn_r2,axis=1)

	avg_std_kcc_mae=np.mean(std_mr_bn_mae,axis=1)
	avg_std_kcc_mse=np.mean(std_mr_bn_mse,axis=1)
	avg_std_kcc_rmse=np.mean(std_mr_bn_rmse,axis=1)
	avg_std_kcc_r2=np.mean(std_mr_bn_r2,axis=1)

	print("Average performance considering all KCCs...")

	for i in range(len(bn_models_name)):
		print(f'Name model: {bn_models_name[i]} ')
		print(f'MAE: {avg_kcc_mrbn_mae[i]}, MAE Standard Deviation: {avg_std_kcc_mae[i]}')
		print(f'MSE: {avg_kcc_mrbn_mse[i]}, MSE Standard Deviation: {avg_std_kcc_mse[i]}')
		print(f'RMSE: {avg_kcc_mrbn_rmse[i]}, RMSE Standard Deviation: {avg_std_kcc_rmse[i]}')
		print(f'R2: {avg_kcc_mrbn_r2[i]}, R2 Standard Deviation: {avg_std_kcc_r2[i]}')



