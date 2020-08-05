"""Benchmarking is used to test the 3D CNN model with standard machine learning models. The utility comes with existing models but the user can add, remove or tweak models based on his requirement. Refer scikit learn for more information about the models: https://scikit-learn.org/stable/supervised_learning.html#supervised-learning"""

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
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor, MLPClassifier

#Importing Config files
import assembly_config as config
import model_config as cftrain

#Importing required modules from the package
from measurement_system import HexagonWlsScanner
from assembly_system import VRMSimulationModel
from data_import import GetTrainData
from metrics_eval import MetricsEval
import hybrid_utils as hy_util

def benchmarking_models_regression(max_models):
	"""benchmarking_models returns a list of models and model names less that or equal to max_models
		
		:param max_models: maximum number of models
		:type max_models: int (required)

		:returns: bn_models: list of models used for benchmarking
		:rtype: list

		:returns: bn_models_name: list of model names used for benchmarking 
		:rtype: list

	"""
	bn_models=[None]*max_models
	bn_models_name=[None]*max_models

	bn_models[0]=MultiOutputRegressor(estimator=xgb.XGBRegressor(colsample_bytree=0.4,gamma=0.045,learning_rate=0.07,max_depth=500,min_child_weight=1.5,n_estimators=50,reg_alpha=0.65,reg_lambda=0.45,subsample=0.95,n_jobs=-1,verbose=True))
	bn_models_name[0] = type(bn_models[0].estimator).__name__
	bn_models[1]=MultiOutputRegressor(estimator=RandomForestRegressor(n_estimators=500,max_depth=20,n_jobs=-1,verbose=True))
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

def benchmarking_models_classification(max_models):
	"""benchmarking_models returns a list of models and model names less that or equal to max_models
		
		:param max_models: maximum number of models
		:type max_models: int (required)

		:returns: bn_models: list of models used for benchmarking
		:rtype: list

		:returns: bn_models_name: list of model names used for benchmarking 
		:rtype: list

	"""
	bn_models=[None]*max_models
	bn_models_name=[None]*max_models

	bn_models[0]=MultiOutputClassifier(estimator=xgb.XGBClassifier(colsample_bytree=0.4,gamma=0.045,learning_rate=0.07,max_depth=500,min_child_weight=1.5,n_estimators=50,reg_alpha=0.65,reg_lambda=0.45,subsample=0.95,n_jobs=-1,verbose=True))
	bn_models_name[0] = type(bn_models[0].estimator).__name__
	bn_models[1]=MultiOutputClassifier(estimator=RandomForestClassifier(n_estimators=500,max_depth=20,n_jobs=-1,verbose=True))
	bn_models_name[1] = type(bn_models[1].estimator).__name__
	bn_models[2]=MultiOutputClassifier(estimator=SVC(kernel='rbf', C=100, gamma=0.1))
	bn_models_name[2] = type(bn_models[2].estimator).__name__
	bn_models[3]=MLPClassifier(hidden_layer_sizes=(512,256,),  activation='relu', solver='adam', alpha=0.001,batch_size='auto',
			   learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
			   random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
			   nesterovs_momentum=True, early_stopping=False, validation_fraction=0.2, beta_1=0.9, beta_2=0.999,
			   epsilon=1e-08)

	bn_models_name[3] = type(bn_models[3]).__name__
	bn_models[4]=MultiOutputClassifier(estimator=DecisionTreeClassifier(max_depth=10))
	bn_models_name[4] = type(bn_models[4].estimator).__name__
	bn_models[5]=MultiOutputClassifier(estimator=LogisticRegression())
	bn_models_name[5] = type(bn_models[5].estimator).__name__


	bn_models=[x for x in bn_models if x is not None]
	bn_models_name=[x for x in bn_models_name if x is not None]
	
	return bn_models,bn_models_name

def benchmarking_models_eval_cla(bn_models,point_data,kcc_dataset,assembly_kccs,bm_path,test_size):
	"""benchmarking_models_evals trains each of the model based on the dataset and returns 
		
		:param bn_models: list of models to be benchmarked
		:type bn_model: list (required)

		:param point_data: input data consisting of node deviations
		:type point_data: numpy.array (samples*nodes) (required)

		:param kcc_dataset: output data consisting of process parameters/KCCs
		:type kcc_dataset: numpy.array (samples*kccs) (required)

		:param assembly_kccs: number of assembly KCCs
		:type assembly_kccs: int (required)

		:param bm_path: Benchmarking path to save benchmarking results
		:type bm_path: str (required)

		:param test_size: The test size split
		:type assembly_kccs: float (required)

		:returns: bn_metrics_eval: Benchmarking metrics 
		:rtype: numpy.array [bn_models*kccs*metrics]

	"""

	X_train, X_test, y_train, y_test = train_test_split(point_data, kcc_dataset, test_size=test_size)
	bnoutput_acc=np.zeros((len(bn_models),assembly_kccs))
	bnoutput_f1=np.zeros((len(bn_models),assembly_kccs))
	bnoutput_pre=np.zeros((len(bn_models),assembly_kccs))
	bnoutput_recall=np.zeros((len(bn_models),assembly_kccs))
	bnoutput_roc_auc=np.zeros((len(bn_models),assembly_kccs))
	bnoutput_kappa=np.zeros((len(bn_models),assembly_kccs))

	y_pred=np.zeros_like(y_test,dtype=float)
	
	for i in range(len(bn_models)):
		print(i)
		model=bn_models[i]
		model.fit(X_train,y_train)
		
		try: 
			y_pred_list=model.predict_proba(X_test)

			#Handling Predict Proba as it returns list of arrays
			for j,item in enumerate(y_pred_list):
				y_pred[:,j]=item[:,1]
		except:
			print("Estimator Doesn't Allow Predict Probability")
			y_pred=model.predict(X_test)

		eval_metrics,accuracy_metrics_df=metrics_eval.metrics_eval_classification(y_pred,y_test,bm_path)
		bnoutput_acc[i,:]=eval_metrics["Accuracy"]
		bnoutput_f1[i,:]=eval_metrics["F1"]
		bnoutput_pre[i,:]=eval_metrics["Precision"]
		bnoutput_recall[i,:]=eval_metrics["Recall"]
		bnoutput_roc_auc[i,:]=eval_metrics["ROC_AUC"]
		bnoutput_kappa[i,:]=eval_metrics["Kappa"]

	bneval_metrics_cla= {
			"Accuracy" : bnoutput_acc,
			"F1" : bnoutput_f1,
			"Precision" : bnoutput_pre,
			"Recall" : bnoutput_recall,
			"ROC_AUC":bnoutput_roc_auc,
			"Kappa":bnoutput_kappa
		}

	return bneval_metrics_cla

def benchmarking_models_eval_reg(bn_models,point_data,kcc_dataset,assembly_kccs,bm_path,test_size):
	"""benchmarking_models_evals trains each of the model based on the dataset and returns 
		
		:param bn_models: list of models to be benchmarked
		:type bn_model: list (required)

		:param point_data: input data consisting of node deviations
		:type point_data: numpy.array (samples*nodes) (required)

		:param kcc_dataset: output data consisting of process parameters/KCCs
		:type kcc_dataset: numpy.array (samples*kccs) (required)

		:param assembly_kccs: number of assembly KCCs
		:type assembly_kccs: int (required)

		:param bm_path: Benchmarking path to save benchmarking results
		:type bm_path: str (required)

		:param test_size: The test size split
		:type assembly_kccs: float (required)

		:returns: bn_metrics_eval: Benchmarking metrics 
		:rtype: numpy.array [bn_models*kccs*metrics]

	"""

	X_train, X_test, y_train, y_test = train_test_split(point_data, kcc_dataset, test_size=test_size)
	bnoutput_mae=np.zeros((len(bn_models),assembly_kccs))
	bnoutput_mse=np.zeros((len(bn_models),assembly_kccs))
	bnoutput_rmse=np.zeros((len(bn_models),assembly_kccs))
	bnoutput_r2=np.zeros((len(bn_models),assembly_kccs))

	for i in range(len(bn_models)):
		print(i)
		model=bn_models[i]
		model.fit(X_train,y_train)
		y_pred=model.predict(X_test)
		eval_metrics,accuracy_metrics_df=metrics_eval.metrics_eval_base(y_pred,y_test,bm_path)
		bnoutput_mae[i,:]=eval_metrics["Mean Absolute Error"]
		bnoutput_mse[i,:]=eval_metrics["Mean Squared Error"]
		bnoutput_rmse[i,:]=eval_metrics["Root Mean Squared Error"]
		bnoutput_r2[i,:]=eval_metrics["R Squared"]

	bneval_metrics_reg= {
			"Mean Absolute Error" : bnoutput_mae,
			"Mean Squared Error" : bnoutput_mse,
			"Root Mean Squared Error" : bnoutput_rmse,
			"R Squared" : bnoutput_r2
		}
	return bneval_metrics_reg

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

	#added for hybrid model
	categorical_kccs=config.assembly_system['categorical_kccs']
	regression_kccs=assembly_kccs-categorical_kccs

	print('Parsing from Training Config File')

	max_models=cftrain.bm_params['max_models']
	runs=cftrain.bm_params['runs']
	split_ratio=cftrain.bm_params['split_ratio']

	print('Creating file Structure....')
	
	folder_name=part_type
	train_path='../trained_models/'+part_type
	pathlib.Path(train_path).mkdir(parents=True, exist_ok=True)

	bm_path=train_path+'/benchmarking/'
	pathlib.Path(bm_path).mkdir(parents=True, exist_ok=True)

	print('Initializing the Assembly System and Measurement System....')
	
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

	#Splitting Regression and Classification KCCs
	kcc_regression,kcc_classification=hy_util.split_kcc(kcc_dataset.values)
	point_data=pd.concat([dataset[0],dataset[1],dataset[2]],axis=1,ignore_index=True)

	point_data=point_data.values
	print(point_data.shape,kcc_regression.shape,kcc_classification.shape)
	print('Benchmarking for all Algorithims')
	bn_models_reg,bn_models_name_reg=benchmarking_models_regression(max_models)
	
	bn_models_cla,bn_models_name_cla=benchmarking_models_classification(max_models)

	bn_models_reg_len=len(bn_models_reg)
	bn_models_cla_len=len(bn_models_cla)
	
	#Regression Metrics
	mr_bnoutput_mae=np.zeros((runs,len(bn_models_reg),regression_kccs))
	mr_bnoutput_mse=np.zeros((runs,len(bn_models_reg),regression_kccs))
	mr_bnoutput_rmse=np.zeros((runs,len(bn_models_reg),regression_kccs))
	mr_bnoutput_r2=np.zeros((runs,len(bn_models_reg),regression_kccs))

	#Classification Metrics
	mr_bnoutput_acc=np.zeros((runs,len(bn_models_cla),categorical_kccs))
	mr_bnoutput_f1=np.zeros((runs,len(bn_models_cla),categorical_kccs))
	mr_bnoutput_pre=np.zeros((runs,len(bn_models_cla),categorical_kccs))
	mr_bnoutput_recall=np.zeros((runs,len(bn_models_cla),categorical_kccs))
	mr_bnoutput_roc_auc=np.zeros((runs,len(bn_models_cla),categorical_kccs))
	mr_bnoutput_kappa=np.zeros((runs,len(bn_models_cla),categorical_kccs))

	#Run Regression Models
	for i in range(runs):
		print("Run ID: ", i)
		bneval_metrics=benchmarking_models_eval_reg(bn_models_reg,point_data,kcc_regression,regression_kccs,bm_path,split_ratio)
		mr_bnoutput_mae[i,:,:]=bneval_metrics["Mean Absolute Error"]
		mr_bnoutput_mse[i,:,:]=bneval_metrics["Mean Squared Error"]
		mr_bnoutput_rmse[i,:,:]=bneval_metrics["Root Mean Squared Error"]
		mr_bnoutput_r2[i,:,:]=bneval_metrics["R Squared"]

	#Run Classification Models
	for i in range(runs):
		print("Run ID: ", i)
		bneval_metrics=benchmarking_models_eval_cla(bn_models_cla,point_data,kcc_classification,categorical_kccs,bm_path,split_ratio)
		mr_bnoutput_acc[i,:,:]=bneval_metrics["Accuracy"]
		mr_bnoutput_f1[i,:,:]=bneval_metrics["F1"]
		mr_bnoutput_pre[i,:,:]=bneval_metrics["Precision"]
		mr_bnoutput_recall[i,:,:]=bneval_metrics["Recall"]
		mr_bnoutput_roc_auc[i,:,:]=bneval_metrics["ROC_AUC"]
		mr_bnoutput_kappa[i,:,:]=bneval_metrics["Kappa"]
		
	#Classification Metrics Calculation
	avg_mrbn_acc=np.mean(mr_bnoutput_acc,axis=0)
	avg_mrbn_f1=np.mean(mr_bnoutput_f1,axis=0)
	avg_mrbn_pre=np.mean(mr_bnoutput_pre,axis=0)
	avg_mrbn_recall=np.mean(mr_bnoutput_recall,axis=0)
	avg_mrbn_roc_auc=np.mean(mr_bnoutput_roc_auc,axis=0)
	avg_mrbn_kappa=np.mean(mr_bnoutput_kappa,axis=0)

	std_mrbn_acc=np.std(mr_bnoutput_acc,axis=0, ddof=1)
	std_mrbn_f1=np.std(mr_bnoutput_f1,axis=0, ddof=1)
	std_mrbn_pre=np.std(mr_bnoutput_pre,axis=0, ddof=1)
	std_mrbn_recall=np.std(mr_bnoutput_recall,axis=0, ddof=1)
	std_mrbn_roc_auc=np.std(mr_bnoutput_roc_auc,axis=0, ddof=1)
	std_mrbn_kappa=np.std(mr_bnoutput_kappa,axis=0, ddof=1)

	avg_kcc_mrbn_acc=np.mean(avg_mrbn_acc,axis=1)
	avg_kcc_mrbn_f1=np.mean(avg_mrbn_f1,axis=1)
	avg_kcc_mrbn_pre=np.mean(avg_mrbn_pre,axis=1)
	avg_kcc_mrbn_recall=np.mean(avg_mrbn_recall,axis=1)
	avg_kcc_mrbn_roc_auc=np.mean(avg_mrbn_roc_auc,axis=1)
	avg_kcc_mrbn_kappa=np.mean(avg_mrbn_kappa,axis=1)

	std_avg_kcc_mrbn_acc=np.mean(std_mrbn_acc,axis=1)
	std_avg_kcc_mrbn_f1=np.mean(std_mrbn_f1,axis=1)
	std_avg_kcc_mrbn_pre=np.mean(std_mrbn_pre,axis=1)
	std_avg_kcc_mrbn_recall=np.mean(std_mrbn_recall,axis=1)
	std_avg_kcc_mrbn_roc_auc=np.mean(std_mrbn_roc_auc,axis=1)
	std_avg_kcc_mrbn_kappa=np.mean(std_mrbn_kappa,axis=1)

	np.savetxt((bm_path+"avg_kcc_mrbn_acc.csv"), avg_kcc_mrbn_acc, delimiter=",")
	np.savetxt((bm_path+"avg_kcc_mrbn_f1.csv"), avg_kcc_mrbn_f1, delimiter=",")
	np.savetxt((bm_path+"avg_kcc_mrbn_pre.csv"), avg_kcc_mrbn_pre, delimiter=",")
	np.savetxt((bm_path+"avg_kcc_mrbn_recall.csv"), avg_kcc_mrbn_recall, delimiter=",")
	np.savetxt((bm_path+"avg_kcc_mrbn_roc_auc.csv"), avg_kcc_mrbn_roc_auc, delimiter=",")
	np.savetxt((bm_path+"avg_kcc_mrbn_kappa.csv"), avg_kcc_mrbn_kappa, delimiter=",")

	np.savetxt((bm_path+"std_avg_kcc_mrbn_acc.csv"), std_avg_kcc_mrbn_acc, delimiter=",")
	np.savetxt((bm_path+"std_avg_kcc_mrbn_f1.csv"), std_avg_kcc_mrbn_f1, delimiter=",")
	np.savetxt((bm_path+"std_avg_kcc_mrbn_pre.csv"), std_avg_kcc_mrbn_pre, delimiter=",")
	np.savetxt((bm_path+"std_avg_kcc_mrbn_recall.csv"), std_avg_kcc_mrbn_recall, delimiter=",")
	np.savetxt((bm_path+"std_avg_kcc_mrbn_roc_auc.csv"), std_avg_kcc_mrbn_roc_auc, delimiter=",")
	np.savetxt((bm_path+"std_avg_kcc_mrbn_kappa.csv"), std_avg_kcc_mrbn_kappa, delimiter=",")

	#Regression Metrics Calculation
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

	print("Average performance considering all KCCs Saved to disk...")

	# for i in range(len(bn_models_name_reg)):
	# 	print('Name model: ', bn_models_name_reg[i])
	# 	print('MAE: ', avg_kcc_mrbn_mae[i], ' MAE Standard Deviation: ' , avg_std_kcc_mae[i])
	# 	print('MSE: ', avg_kcc_mrbn_mse[i], ' MSE Standard Deviation: ',avg_std_kcc_mse[i])
	# 	print('RMSE: ', avg_kcc_mrbn_rmse[i],' RMSE Standard Deviation: ', avg_std_kcc_rmse[i])
	# 	print('R2: ', avg_kcc_mrbn_r2[i], ' R2 Standard Deviation: ', avg_std_kcc_r2[i])

	np.savetxt((bm_path+"avg_kcc_mrbn_mae.csv"), avg_kcc_mrbn_mae, delimiter=",")
	np.savetxt((bm_path+"avg_kcc_mrbn_rmse.csv"), avg_kcc_mrbn_rmse, delimiter=",")
	np.savetxt((bm_path+"avg_kcc_mrbn_r2.csv"), avg_kcc_mrbn_r2, delimiter=",")

	np.savetxt((bm_path+"avg_std_kcc_mae.csv"), avg_std_kcc_mae, delimiter=",")
	np.savetxt((bm_path+"avg_std_kcc_rmse.csv"), avg_std_kcc_rmse, delimiter=",")
	np.savetxt((bm_path+"avg_std_kcc_r2.csv"), avg_std_kcc_r2, delimiter=",")


	np.savetxt((bm_path+"avg_mrbn_mae.csv"), avg_mrbn_mae, delimiter=",")
	np.savetxt((bm_path+"avg_mrbn_rmse.csv"), avg_mrbn_rmse, delimiter=",")
	np.savetxt((bm_path+"avg_mrbn_r2.csv"), avg_mrbn_r2, delimiter=",")

	np.savetxt((bm_path+"std_mrbn_mae.csv"), std_mr_bn_mae, delimiter=",")
	np.savetxt((bm_path+"std_mrbn_rmse.csv"), std_mr_bn_rmse, delimiter=",")
	np.savetxt((bm_path+"std_mrbn_r2.csv"), std_mr_bn_r2, delimiter=",")

