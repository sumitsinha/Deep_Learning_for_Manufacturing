from sklearn import metrics
import numpy as np

class MetricsEval():
	
	def metrics_eval(self,predicted_y, test_y):

		kcc_dim=test_y.shape[1]
		print(kcc_dim)
		print(test_y.shape)
		print(predicted_y.shape)

		# Calculating Regression Based Evaluation Metrics
		mae_KCCs=np.zeros((kcc_dim))
		mse_KCCs=np.zeros((kcc_dim))
		r2_KCCs=np.zeros((kcc_dim))

		kcc_id=[]

		for i in range(kcc_dim):
		    
		    kcc_name="KCC_"+str(i+1)
		    kcc_id.append(kcc_name)
		    
		    #mae_KCCs[i]=metrics.mean_absolute_error(predicted_y[:,i], test_y[:,i])
		    mse_KCCs[i]=metrics.mean_squared_error(predicted_y[:,i], test_y[:,i])
		    r2_KCCs[i] = metrics.r2_score(predicted_y[:,i], test_y[:,i])

		rmse_KCCs=sqrt(mse_KCCs)

		eval_metrics= {
			"Mean Absolute Error" : mae_KCCs,
			"Mean Squared Error" : mse_KCCs,
			"Root Mean Squared Error" : rmse_KCCs,
			"R Squared" : r2_KCCs
		}
		
		accuracy_metrics_df=pd.DataFrame({'KCC_ID':kcc_id,'MAE':mae_KCCs,'MSE':mse_KCCs,'RMSE':rmse_KCCs,'R2':r2_KCCs})
		accuracy_metrics_df.columns = ['KCC_ID','MAE','MSE','RMSE','R2']
		accuracy_metrics_df.to_csv('../logs/metrics.csv')
		return eval_metrics

		def metrics_eval_aleatoric_model(self,predicted_y, test_y):

			kcc_dim=test_y.shape[1]
			output_dim=test_y.shape[1]
			log_variance=y_pred[:,kcc_dim]
			variance=np.exp(log_variance)
			
			standard_deviation=np.sqrt(variance)
			avg_aleatoric_SD=np.mean(standard_deviation)

			# Calculating Regression Based Evaluation Metrics
			mae_KCCs=np.zeros((kcc_dim))
			mse_KCCs=np.zeros((kcc_dim))
			r2_KCCs=np.zeros((kcc_dim))

			for i in range(kcc_dim):
			    mae_KCCs[i]=metrics.mean_absolute_error(predicted_y[:,i], test_y[:,i])
			    mse_KCCs[i]=metrics.mean_squared_error(predicted_y[:,i], test_y[:,i])
			    r2_KCCs[i] = metrics.r2_score(predicted_y[:,i], test_y[:,i])

			rmse_KCCs=sqrt(mse_KCCs)

			eval_metrics= {
				"Mean Absolute Error" : mae_KCCs,
				"Mean Squared Error" : mse_KCCs,
				"Root Mean Squared Error" : rmse_KCCs,
				"R Squared" : r2_KCCs,
				"Aleatoric Standard Deviation":avg_aleatoric_SD
			}

			return eval_metrics