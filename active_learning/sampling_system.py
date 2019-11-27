import os
import sys
current_path=os.path.dirname(__file__)
parentdir = os.path.dirname(current_path)

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

import kcc_config as kcc_config
import sampling_config as sampling_config

class adaptive_sampling():

	def __init__(self,sample_dim,sample_type,adaptive_samples_dim,adaptive_runs):
		self.sample_dim=sample_dim
		self.sample_type=sample_type
		self.adaptive_samples_dim=adaptive_samples_dim
		self.adaptive_runs=adaptive_runs
	
	def inital_sampling(self,kcc_struct,sample_dim):

		kcc_dim=len(kcc_struct)
		sample_type=self.sample_type

		samples =lhs(kcc_dim,samples=sample_dim,criterion='center')
		initial_samples=np.zeros_like(samples)
		index=0
		for kcc in kcc_struct:   
			if(sample_type=='uniform'):
				#initial_samples[:,index]=uniform.ppf(samples[:,index], loc=kcc['kcc_nominal'], scale=kcc['kcc_max']-kcc['kcc_min'])
				initial_samples[:,index]=samples[:,index]*(kcc['kcc_max']-kcc['kcc_min'])+kcc['kcc_min']
			else:
				initial_samples[:,index]=norm.ppf(samples[:,index], loc=(kcc['kcc_nominal']+kcc['kcc_max'])/2, scale=(kcc['kcc_max']-kcc['kcc_min'])/6)
			index=index+1

		return initial_samples

	def adpative_samples_gen(self,kcc_struct,run_id):
		
		adaptive_samples_dim=self.adaptive_samples_dim

		adaptive_samples=[]

		index=0
		for kcc in kcc_struct:
			adaptive_samples.append(self.inital_sampling(kcc_struct,adaptive_samples_dim))
			for i in run_id:
				adaptive_samples[index][index+i,:]=0

		return adaptive_samples

if __name__ == '__main__':
	
	kcc_struct=kcc_config.kcc_struct
	sampling_config=sampling_config.sampling_config

	adaptive_sampling=adaptive_sampling(sampling_config['sample_dim'],sampling_config['sample_type'],sampling_config['adaptive_sample_dim'],sampling_config['adaptive_runs'])

	print('Generating inital samples')
	initial_samples=adaptive_sampling.inital_sampling(kcc_struct,sampling_config['sample_dim'])

	file_name="initial_samples_data_check.csv"
	file_path='./sample_input/'+file_name
	np.savetxt(file_path, initial_samples, delimiter=",")

	print('Inital Samples Saved to path: ',file_path)

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