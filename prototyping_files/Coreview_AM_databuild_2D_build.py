# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:18:58 2019

@author: sinha_s
"""
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing, model_selection, metrics
from sklearn.utils import shuffle
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from keras import backend as K
from sklearn.externals import joblib
from keras.models import model_from_json
K.clear_session()
import numpy as np
import pandas as pd
from numpy import zeros, newaxis
import psycopg2
from sqlalchemy import create_engine
import zipfile  
import os
#from open3d import *
from numpy import zeros, newaxis
from tqdm import tqdm
import math

#%%importing pre-exisiting resources
df_point_index_core_view = np.load("Halo_cov_index_data_2D_mapping_10.dat")
final_model=load_model('./model/pointdevnet_demo_transformed_2D_TL.h5')
#normal_point_list = pd.read_csv('normal_refined_points.csv',header=None)
#normal_point_index=normal_point_list-1
#normal_index_list=list(normal_point_index.values[:,0])
#%% importing measurement data
#final_model=load_model('./model/pointdevnet_demo_transformed_classifier.h5')
##%%
#engine = create_engine('postgresql://postgres:sumit123!@10.255.4.40:5432/IPQI')
#squery='select cox,coy,coz from order by index,id'
#df_nom = pd.read_sql_query(squery,con=engine)
#nom_cop=df_nom.values
#%%
measurement_data=pd.read_csv("M2_C5000.txt",skiprows=25,low_memory=False,sep='\t', lineterminator='\r',error_bad_lines=False)
measurement_data_subset=measurement_data.loc[(measurement_data['Name'].str[0:2] == 'SF')]
nominal_coordinates=measurement_data_subset.iloc[:,5:8]
actual_coordinates=measurement_data_subset.iloc[:,10:13]
deviations=actual_coordinates.values-nominal_coordinates.values
imputed_deviations= np.nan_to_num(deviations)
y_dev_data_filtered=imputed_deviations[:,1:2]
node_IDs=measurement_data_subset.iloc[:,1:2]
node_IDs=node_IDs['Name'].str[2:]
node_IDs=node_IDs.astype(int).values
voxel_dev_data=np.zeros((1,10,10,1))    
sim_data = pd.read_csv("voxel_validation.csv",header=None)
sim_data_selected=sim_data.iloc[:,1:2].values
#y_dev_data_filtered=sim_data_selected
for i in range(10):
    for j in range(10):
        point_pool_check= np.where((df_point_index_core_view[:,1:3]==(i,j)).all(axis=1))[0]
        node_subsets=df_point_index_core_view[point_pool_check,0:1]
        print(len(node_subsets))
        value_array=y_dev_data_filtered[(node_subsets[:,0]-1).astype(int),:]
        median_val=np.median(value_array)
        voxel_dev_data[0,i,j,:]=median_val

voxel_dev_data= np.nan_to_num(voxel_dev_data)   
display=voxel_dev_data[0,:,:,:]        

y_pred = final_model.predict(voxel_dev_data)
print(y_pred)












    
    