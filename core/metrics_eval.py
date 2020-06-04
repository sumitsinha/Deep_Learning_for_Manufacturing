""" Contains classes and methods to obtain various regression based metrics to evaluate"""
from sklearn import metrics
import numpy as np
import pandas as pd
import math
import sys
sys.path.append("../config")

class MetricsEval:
	"""MetricsEval Class

		Evaluate metrics to evaluate model performance
		
	"""	
	def metrics_eval_base(self,predicted_y, test_y,logs_path,run_id=0):
		"""Get predicted and actual value for all KCCs and return regression metrics namely: Mean Absolute Error, Mean Squared Error, Root Mean Squared Error, R-Squared Value
			
			:param predicted_y: predicted values for the process parameters 
			:type conn_str: numpy.array [test_samples*kccs] (required)

			:param predicted_y: actual values for the process parameters 
			:type conn_str: numpy.array [test_samples*kccs] (required)

			:param logs_path: Logs path to save the evaluation metrics
			:type logs_path: str (required)

			:returns: dictionary of all metrics for each KCC
			:rtype: dict

			:returns: dataframe of all metrics for each KCC
			:rtype: pandas.dataframe
		"""

		kcc_dim=test_y.shape[1]
		
		import kcc_config as kcc_config
		kcc_struct=kcc_config.get_kcc_struct()
		# Calculating Regression Based Evaluation Metrics
		mae_KCCs=np.zeros((kcc_dim))
		mse_KCCs=np.zeros((kcc_dim))
		r2_KCCs=np.zeros((kcc_dim))
		#print(kcc_dim)
		kcc_id=[]
		
		for kcc in kcc_struct:
			
			if(kcc['kcc_type']==0):
				kcc_name=kcc['kcc_id']
				kcc_id.append(kcc_name)
			
		mae_KCCs=metrics.mean_absolute_error(predicted_y, test_y,multioutput='raw_values')
		mse_KCCs=metrics.mean_squared_error(predicted_y, test_y,multioutput='raw_values')
		r2_KCCs = metrics.r2_score(predicted_y, test_y,multioutput='raw_values')

		rmse_KCCs=np.sqrt(mse_KCCs)
		eval_metrics= {
			"Mean Absolute Error" : mae_KCCs,
			"Mean Squared Error" : mse_KCCs,
			"Root Mean Squared Error" : rmse_KCCs,
			"R Squared" : r2_KCCs
		}

		#print(len(kcc_id),len(mae_KCCs),len(mae_KCCs),len(rmse_KCCs),len(r2_KCCs))
		#print(eval_metrics)
		accuracy_metrics_df=pd.DataFrame({'KCC_ID':kcc_id,'MAE':mae_KCCs,'MSE':mse_KCCs,'RMSE':rmse_KCCs,'R2':r2_KCCs},columns=['KCC_ID','MAE','MSE','RMSE','R2'])
		accuracy_metrics_df=accuracy_metrics_df.set_index('KCC_ID')
		#accuracy_metrics_df.to_csv(logs_path+'/metrics.csv') #moved to function call
		return eval_metrics,accuracy_metrics_df

	def metrics_eval_classification(self,y_pred, y_true,logs_path,run_id=0):
		"""Get predicted and actual value for all KCCs and return regression metrics namely: Mean Absolute Error, Mean Squared Error, Root Mean Squared Error, R-Squared Value
			
			:param predicted_y: predicted values for the process parameters 
			:type conn_str: numpy.array [test_samples*kccs] (required)

			:param predicted_y: actual values for the process parameters 
			:type conn_str: numpy.array [test_samples*kccs] (required)

			:param logs_path: Logs path to save the evaluation metrics
			:type logs_path: str (required)

			:returns: dictionary of all metrics for each KCC
			:rtype: dict

			:returns: dataframe of all metrics for each KCC
			:rtype: pandas.dataframe
		"""

		kcc_dim=y_true.shape[1]
		
		import kcc_config as kcc_config
		kcc_struct=kcc_config.get_kcc_struct()
		# Calculating Regression Based Evaluation Metrics

		kcc_id=[]
		
		for kcc in kcc_struct:
			if(kcc['kcc_type']==1):
				kcc_name=kcc['kcc_id']
				kcc_id.append(kcc_name)
			
		acc_kccs=[]
		f1_kccs=[]
		pre_kccs=[]
		recall_kccs=[]
		roc_auc_kccs=[]
		kappa_kccs=[]

		from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score,cohen_kappa_score

		for i in range(y_true.shape[1]):
			
			#Binary Prediction arrray
			y_pred_bin=np.where(y_pred[:,i] > 0.5, 1, 0)
			
			acc_kccs.append(accuracy_score(y_true[:,i],y_pred_bin))
			f1_kccs.append(f1_score(y_true[:,i],y_pred_bin))
			pre_kccs.append(precision_score(y_true[:,i],y_pred_bin))
			recall_kccs.append(recall_score(y_true[:,i],y_pred_bin))
			kappa_kccs.append(cohen_kappa_score(y_true[:,i],y_pred_bin))
			
			#Probablity based Scoring
			roc_auc_kccs.append(roc_auc_score(y_true[:,i],y_pred[:,i]))

		eval_metrics= {
			"KCC_ID":kcc_id,
			"Accuracy" : acc_kccs,
			"F1" : f1_kccs,
			"Precision" : pre_kccs,
			"Recall" : recall_kccs,
			"ROC_AUC":roc_auc_kccs,
			"Kappa":kappa_kccs
		}
		
		accuracy_metrics_df=pd.DataFrame.from_dict(eval_metrics)
		accuracy_metrics_df=accuracy_metrics_df.set_index('KCC_ID')
		#accuracy_metrics_df.to_csv(logs_path+'/metrics.csv') #moved to function call
		return eval_metrics,accuracy_metrics_df

	def metrics_eval_cop(self,predicted_y, test_y,logs_path,run_id=0):
		"""Get predicted and actual value for all KCCs and return regression metrics namely: Mean Absolute Error, Mean Squared Error, Root Mean Squared Error, R-Squared Value
			
			:param predicted_y: predicted values for the process parameters 
			:type conn_str: numpy.array [test_samples*kccs] (required)

			:param predicted_y: actual values for the process parameters 
			:type conn_str: numpy.array [test_samples*kccs] (required)

			:param logs_path: Logs path to save the evaluation metrics
			:type logs_path: str (required)

			:returns: dictionary of all metrics for each KCC
			:rtype: dict

			:returns: dataframe of all metrics for each KCC
			:rtype: pandas.dataframe
		"""

		kcc_dim=test_y.shape[1]
		
		mae_KCCs=np.zeros((kcc_dim))
		mse_KCCs=np.zeros((kcc_dim))
		r2_KCCs=np.zeros((kcc_dim))
   
		mae_KCCs=metrics.mean_absolute_error(predicted_y, test_y,multioutput='raw_values')
		mse_KCCs=metrics.mean_squared_error(predicted_y, test_y,multioutput='raw_values')
		r2_KCCs = metrics.r2_score(predicted_y, test_y,multioutput='raw_values')

		rmse_KCCs=np.sqrt(mse_KCCs)
		
		r2_adjusted=np.zeros(kcc_dim)

		from tqdm import tqdm
		for i in tqdm(range(kcc_dim)):
			y_cop_test_flat=test_y[:,i]
			y_cop_pred_flat=predicted_y[:,i]
			combined_array=np.stack([y_cop_test_flat,y_cop_pred_flat],axis=1)
			filtered_array=combined_array[np.where(abs(combined_array[:,0]) >= 0.01)]
			y_cop_test_vector=filtered_array[:,0:1]
			y_cop_pred_vector=filtered_array[:,1:2]
			#print(y_cop_pred_vector.shape)
			r2_adjusted[i] = metrics.r2_score(y_cop_test_vector,y_cop_pred_vector,multioutput='raw_values')[0]
		
		eval_metrics= {
			"Mean Absolute Error" : mae_KCCs,
			"Mean Squared Error" : mse_KCCs,
			"Root Mean Squared Error" : rmse_KCCs,
			"R Squared" : r2_KCCs,
			"R Squared Adjusted" : r2_adjusted
		}
		
		accuracy_metrics_df=pd.DataFrame({'MAE':mae_KCCs,'MSE':mse_KCCs,'RMSE':rmse_KCCs,'R2':r2_KCCs,"R2_Adjusted":r2_adjusted},columns=['MAE','MSE','RMSE','R2',"R2_Adjusted"])
		#accuracy_metrics_df.to_csv(logs_path+'/metrics.csv') #moved to function call
		return eval_metrics,accuracy_metrics_df
		
	def metrics_eval_aleatoric_model(self,predicted_y, test_y,logs_path):

			kcc_dim=test_y.shape[1]
			log_variance=y_pred[:,kcc_dim]
			variance=np.exp(log_variance)
			
			predicted_y_sub=predicted_y[:,0:(kcc_dim-1)]
			standard_deviation=np.sqrt(variance)
			avg_aleatoric_SD=np.mean(standard_deviation)

			# Calculating Regression Based Evaluation Metrics
			mae_KCCs=np.zeros((kcc_dim))
			mse_KCCs=np.zeros((kcc_dim))
			r2_KCCs=np.zeros((kcc_dim))
			kcc_id=[]

			for i in range(kcc_dim):  
				kcc_name="KCC_"+str(i+1)
				kcc_id.append(kcc_name)
			
			mae_KCCs=metrics.mean_absolute_error(predicted_y_sub, test_y,multioutput='raw_values')
			mse_KCCs=metrics.mean_squared_error(predicted_y_sub, test_y,multioutput='raw_values')
			r2_KCCs = metrics.r2_score(predicted_y_sub, test_y,multioutput='raw_values')

			rmse_KCCs=sqrt(mse_KCCs)

			eval_metrics= {
				"Mean Absolute Error" : mae_KCCs,
				"Mean Squared Error" : mse_KCCs,
				"Root Mean Squared Error" : rmse_KCCs,
				"R Squared" : r2_KCCs,
				"Aleatoric Standard Deviation":avg_aleatoric_SD
			}

			accuracy_metrics_df=pd.DataFrame({'KCC_ID':kcc_id,'MAE':mae_KCCs,'MSE':mse_KCCs,'RMSE':rmse_KCCs,'R2':r2_KCCs})
			accuracy_metrics_df.columns = ['KCC_ID','MAE','MSE','RMSE','R2']
			accuracy_metrics_df.to_csv(logs_path+'/metrics.csv')
			return eval_metrics