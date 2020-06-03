import os
import sys
current_path=os.path.dirname(__file__)
parentdir = os.path.dirname(current_path)

""" Contains sampling classes and methods to enable active learning between VRM and 3D CNN model """

#Adding Path to various Modules
sys.path.append("../core")
sys.path.append("../visualization")
sys.path.append("../utilities")
sys.path.append("../datasets")
sys.path.append("../trained_models")
sys.path.append("../config")

from pyDOE import lhs
from scipy.stats import uniform,norm
import numpy as np

#Importing Config Files
import kcc_config as kcc_config
import sampling_config as sampling_config

class AdaptiveSampling():
	"""Assembly System Class

		:param sample_dim: The number of initial samples to be generated
		:type sample_dim: int (required)

		:param sample_type: Type of sampling to be used for generating initial samples
		:type sample_dim: str (required)

		:param adaptive_sample_dim: The number of samples to be generated with each adaptive sample run
		:type adaptive_sample_dim: int (required) 

		:param adaptive_runs: The maximum number of adaptive runs to be conducted
		:type adaptive_runs: int (required) 
	"""
	def __init__(self,sample_dim,sample_type,adaptive_samples_dim,adaptive_runs):
		self.sample_dim=sample_dim
		self.sample_type=sample_type
		self.adaptive_samples_dim=adaptive_samples_dim
		self.adaptive_runs=adaptive_runs
	
	def inital_sampling_lhs(self,kcc_struct,sample_dim):
		"""Generates multi-variate LHS samples for each KCC and scales then based on the KCC maximum and minimum value

			:param kcc_struct: list of dictionaries for each KCC from kcc_config file
			:type file_name: list (required)

			:param sample_dim: The number of initial samples to be generated
			:type sample_dim: int (required)

			:returns: numpy array of sampled KCCs
			:rtype: numpy.array [sample_dim*kcc_dim]
		"""
		kcc_dim=len(kcc_struct)
		sample_type=self.sample_type

		samples =lhs(kcc_dim,samples=sample_dim,criterion='center')
		initial_samples=np.zeros_like(samples)
		inital_samples_type=np.zeros((sample_dim,kcc_dim))
		index=0
		for kcc in kcc_struct:   
			if(kcc['kcc_type']==0):
				initial_samples[:,index]=np.random.uniform(kcc['kcc_min'],kcc['kcc_max'],sample_dim)
				
			
			if(kcc['kcc_type']==1):
				initial_samples[:,index]=np.random.randint(2, size=sample_dim)
				#flagging for categoric
				inital_samples_type[:,index]=1

			index=index+1
		return initial_samples,inital_samples_type

	def inital_sampling_uniform_random(self,kcc_struct,sample_dim):
		"""Generates multi-variate uniform random samples for each KCC and scales then based on the KCC maximum and minimum value

			:param kcc_struct: list of dictionaries for each KCC from kcc_config file
			:type file_name: list (required)

			:param sample_dim: The number of initial samples to be generated
			:type sample_dim: int (required)

			:returns: numpy array of sampled KCCs
			:rtype: numpy.array [sample_dim*kcc_dim]
		"""
		kcc_dim=len(kcc_struct)
		sample_type=self.sample_type
		initial_samples=np.zeros((sample_dim,kcc_dim))
		inital_samples_type=np.zeros((sample_dim,kcc_dim))
		index=0
		for kcc in kcc_struct:
			
			if(kcc['kcc_type']==0):
				initial_samples[:,index]=np.random.uniform(kcc['kcc_min'],kcc['kcc_max'],sample_dim)
				
			
			if(kcc['kcc_type']==1):
				initial_samples[:,index]=np.random.randint(2, size=sample_dim)
				#flagging for categoric
				inital_samples_type[:,index]=1

			index=index+1
		#pp_making=1

		return initial_samples, inital_samples_type
	
	def adpative_samples_gen(self,kcc_struct,run_id):
		"""Adaptive samples based on model uncertainty, currently this is Work in Progress

		"""
		adaptive_samples_dim=self.adaptive_samples_dim

		adaptive_samples=[]

		index=0
		for kcc in kcc_struct:
			adaptive_samples.append(self.inital_sampling(kcc_struct,adaptive_samples_dim))
			for i in run_id:
				adaptive_samples[index][index+i,:]=0

		return adaptive_samples

	def get_uncertaninity(self,kcc_struct,model,samples):
		pass


if __name__ == '__main__':
	
	#kcc_struct=kcc_config.kcc_struct
	kcc_struct=kcc_config.get_kcc_struct()
	sampling_config=sampling_config.sampling_config

	adaptive_sampling=AdaptiveSampling(sampling_config['sample_dim'],sampling_config['sample_type'],sampling_config['adaptive_sample_dim'],sampling_config['adaptive_runs'])

	print('Generating initial samples')

	if(adaptive_sampling.sample_type=='lhs'):
		initial_samples=adaptive_sampling.inital_sampling_lhs(kcc_struct,sampling_config['sample_dim'])
	else:
		initial_samples,inital_samples_type=adaptive_sampling.inital_sampling_uniform_random(kcc_struct,sampling_config['sample_dim'])

	pp_masking=sampling_config['pp_masking']

	folder_name=sampling_config['output_folder_name']
	file_name=sampling_config['output_file_name']
	
	if(pp_masking==1):
		import random
		mask_array=np.zeros_like(initial_samples)
			
		for i in range(initial_samples.shape[0]):
			random_index=random.randint(0,initial_samples.shape[1]-1)
			mask_array[i,random_index]=initial_samples[i,random_index]

		initial_samples=mask_array
			
	file_path='./sample_input/'+folder_name+'/'+file_name+'.csv'
	np.savetxt(file_path, initial_samples, delimiter=",")

	file_path='./sample_input/'+folder_name+'/'+file_name+'_type'+'.csv'
	np.savetxt(file_path, inital_samples_type, delimiter=",")
	
	print('Initial Samples Saved to path: ',file_path)

	#WIP to integrate adaptive sampling from VRM Oracle
	#@Run VRM Oracle on initial samples
	#@Train Model on initial samples

	"""
	for i in range(sampling_config['adaptive_runs']):
		adaptive_samples=adaptive_sampling.adpative_samples_gen(kcc_struct,i)
		sample_uncertaninity=model_uncertaninity.get_uncertaninity(adaptive_samples)
		adaptive_sample_id=sample_uncertaninity.index(min(sample_uncertaninity))
		selected_adaptive_samples=adaptive_samples[adaptive_sample_id]
		
		#@Run Matlab model on selected adaptive samples
		#@Fine Tune Model on the genrated adaptive samples
		#@Check model on test dataset, stop if accuracy crietria is met
	
	"""