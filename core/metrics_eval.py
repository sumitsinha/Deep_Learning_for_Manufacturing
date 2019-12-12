""" Contains classes and methods to obtain various regression based metrics to evaluate"""
from sklearn import metrics
import numpy as np
import pandas as pd
import math

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

		# Calculating Regression Based Evaluation Metrics
		mae_KCCs=np.zeros((kcc_dim))
		mse_KCCs=np.zeros((kcc_dim))
		r2_KCCs=np.zeros((kcc_dim))

		kcc_id=[]

		for i in range(kcc_dim):  
		    kcc_name="KCC_"+str(i+1)
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
		
		accuracy_metrics_df=pd.DataFrame({'KCC_ID':kcc_id,'MAE':mae_KCCs,'MSE':mse_KCCs,'RMSE':rmse_KCCs,'R2':r2_KCCs},columns=['KCC_ID','MAE','MSE','RMSE','R2'])
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