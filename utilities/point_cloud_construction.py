
import os
import sys
current_path=os.path.dirname(__file__)
parentdir = os.path.dirname(current_path)

os.environ["CUDA_VISIBLE_DEVICES"]="0" # Nvidia Quadro GV100
#os.environ["CUDA_VISIBLE_DEVICES"]="1" # Nvidia Quadro M2000

#Adding Path to various Modules
sys.path.append("../core")
sys.path.append("../visualization")
sys.path.append("../utilities")
sys.path.append("../datasets")
sys.path.append("../trained_models")
sys.path.append("../config")
#path_var=os.path.join(os.path.dirname(__file__),"../utilities")
#sys.path.append(path_var)
#sys.path.insert(0,parentdir) 

#Importing Required Modules
import pathlib
import numpy as np
import pandas as pd

class GetPointCloud:

	def getcopdev(self,voxel_data,mapping_index,nominal_cop):

		from tqdm import tqdm
		point_cloud_dev=np.zeros_like(nominal_cop)

		for i in range(len(point_cloud_dev)):		
			point_cloud_dev[i,:]=voxel_data[int(mapping_index[i,0]),int(mapping_index[i,1]),int(mapping_index[i,2]),:]

		return point_cloud_dev

