from sklearn import metrics

def metrics_eval(predicted_y, test_y, kcc_dim=test_y.shape[1]):

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
		"R Squared" : r2_KCCs
	}

	return eval_metrics

