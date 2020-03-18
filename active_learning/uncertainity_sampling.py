import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd

tfd = tfp.distributions

class UncertainitySampling():

	def __init__(self,adaptive_samples_dim,num_mix,weight=0.5):
		self.adaptive_samples=adaptive_samples
		self.weight=weight
		self.num_mix=num_mix
	
	def get_distribution_smaples(self,y_actual,y_pred,y_std):

		error=y_actual-y_pred
		abs_error=np.absolute(error)

		error_sum=abs_error.sum(axis=1)
		print(error_sum.shape)

		std_sum=y_std.sum(axis=1)

		print(error_sum.shape)

		#Sampling metric
		sampling_metric=error_sum*self.weight+std_sum*(1-self.weight)

		combined_distribution_metric=np.zeros((y_actual.shape[0],y_actual.shape[1]+1))

		combined_distribution_metric[:,0:1]=sampling_metric

		combined_distribution_metric[:,1:]=y_actual

		combined_distribution_metric_sorted = combined_distribution_metric[combined_distribution_metric[:,0].argsort()]

		gmm_mixture_coeff=[]

		gmm_means=[]

		gmm_variance=[]

		components=[]
		sample_size=len(y_actual)/self.num_mix

		intial_idx=0
		end_idx=sample_size

		for i in range(self.num_mix):
			
			obs_sampling_metric=combined_distribution_metric[intial_idx:end_idx,0:1]
			obs_data=combined_distribution_metric[intial_idx:end_idx,1:]

			gmm_mixture_coeff.append(obs_sampling_metric.sum())

			gmm_means.append(mu = tf.reduce_mean(obs_data, axis=0))

			gmm_variance.append(tfp.stats.cholesky_covariance(obs_data,sample_axis=0))

			component=tfd.MultivariateNormalTriL(gmm_means[i],gmm_variance[i])
			components.append(component)

			intial_idx=end_idx
			end_idx=end_idx+sample_size

		gmm_mixture_coeff = [float(i)/sum(gmm_mixture_coeff) for i in gmm_mixture_coeff]

		gmm_mixture_model=tfd.Mixture(cat=tfd.Categorical(probs=gmm_mixture_coeff),components=components)

		samples=tfd.Sample(gmm_mixture_model,sample_shape=self.adaptive_samples_dim)

		output=samples.sample()

		return samples

