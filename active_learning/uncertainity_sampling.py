import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd

tfd = tfp.distributions

import sys
sys.path.append("../config")

class UncertainitySampling():

	def __init__(self,adaptive_samples_dim,num_mix,weight=0.5):
		self.adaptive_samples_dim=adaptive_samples_dim
		self.weight=weight
		self.num_mix=num_mix
	
	def get_distribution_samples(self,y_actual,y_pred,y_std,weight_fac=0.5):

		error=y_actual-y_pred
		
		abs_error=np.absolute(error)
		error_sum=abs_error.sum(axis=1)
		error_sum /= np.max(error_sum)
		
		std_sum=y_std.sum(axis=1)
		std_sum /= np.max(std_sum)

		#Sampling metric
		sampling_metric=error_sum*weight_fac+std_sum*(1-weight_fac)
		combined_distribution_metric=np.zeros((y_actual.shape[0],y_actual.shape[1]+1))
		combined_distribution_metric[:,0]=sampling_metric
		combined_distribution_metric[:,1:]=y_actual
		combined_distribution_metric_sorted = combined_distribution_metric[combined_distribution_metric[:,0].argsort()]
		#np.savetxt('../trained_models/sampling_check/combined.csv', combined_distribution_metric_sorted, delimiter=",")
		
		#Initialize GMM models parameter storage
		gmm_mixture_coeff=[]
		gmm_means=[]
		gmm_variance=[]
		components=[]
		gmm_model_params=np.zeros((self.num_mix,y_actual.shape[1]*(y_actual.shape[1]+1)))
		sample_size=int(len(y_actual)/self.num_mix)

		intial_idx=0
		end_idx=sample_size

		for i in range(self.num_mix):
			
			obs_sampling_metric=combined_distribution_metric_sorted[intial_idx:end_idx,0:1]
			obs_data=combined_distribution_metric_sorted[intial_idx:end_idx,1:]

			gmm_mixture_coeff.append(obs_sampling_metric.sum())

			gmm_means.append(tf.reduce_mean(obs_data, axis=0))

			print(obs_data)
			gmm_variance.append(tfp.stats.cholesky_covariance(obs_data,sample_axis=0))
			print(gmm_variance)
			component=tfd.MultivariateNormalTriL(gmm_means[i],gmm_variance[i])
			components.append(component)

			intial_idx=end_idx
			end_idx=end_idx+sample_size

		gmm_mixture_coeff = [float(i)/sum(gmm_mixture_coeff) for i in gmm_mixture_coeff]
		print(gmm_mixture_coeff)
		gmm_mixture_model=tfd.Mixture(cat=tfd.Categorical(probs=gmm_mixture_coeff),components=components)
		samples=tfd.Sample(gmm_mixture_model,sample_shape=self.adaptive_samples_dim)
		output=samples.sample()
		output=np.array(output) 
		#np.savetxt('../trained_models/sampling_check/samples.csv', output, delimiter=",")
		gmm_model_params[:,0]=np.array(gmm_mixture_coeff)
		
		for j in range(self.num_mix):
			gmm_model_params[j,1:1+y_actual.shape[1]]=gmm_means[j]
			gmm_model_params[j,y_actual.shape[1]:]=gmm_variance[j].numpy().flatten()

		#np.savetxt('../trained_models/sampling_check/gmm_model_params.csv', gmm_model_params, delimiter=",")
		from kcc_config import kcc_struct

		for i,kcc in enumerate(kcc_struct):   
			output[i,:]= np.clip(output[i,:], kcc['kcc_min'], kcc['kcc_max'])

		return output,gmm_model_params

if __name__ == '__main__':

	adaptive_samples_dim=200
	num_mix=10

	unsap=UncertainitySampling(adaptive_samples_dim,num_mix)

	y_pred=pd.read_csv('../trained_models/sampling_check/y_pred.csv',header=None)
	y_actual=pd.read_csv('../trained_models/sampling_check/y_actual.csv',header=None)
	y_std=pd.read_csv('../trained_models/sampling_check/y_std.csv',header=None)


	samples=unsap.get_distribution_samples(y_actual.values,y_pred.values,y_std.values)