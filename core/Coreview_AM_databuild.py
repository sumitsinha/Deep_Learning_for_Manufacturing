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

#%%
#Get deviation data
VOXEL_NODE_IDs=[]
def get_dev_data(y1,y2,p):   
    if(abs(y1)>abs(y2)):
        y_dev=y1
        VOXEL_NODE_IDs.append(p+1)
    else:
        y_dev=y2
    retval=y_dev
    return retval
#%%importing pre-exisiting resources
df_point_index_core_view = np.load("Halo_cov_index_data.dat")
point_import=np.loadtxt("sampled_points.csv")
#point_import=pd.read_csv("import_input.csv")
#point_data=(point_import.iloc[:,2].values-1).tolist() #Extracting the point indexes
#%% importing measurement data

final_model=load_model('./model/pointdevnet_demo_transformed_classifier.h5')
#%%
engine = create_engine('postgresql://postgres:sumit123!@10.255.4.40:5432/IPQI')
squery='select cox,coy,coz from order by index,id'
df_nom = pd.read_sql_query(squery,con=engine)
np.savetxt("voxel_IDS.csv", VOXEL_NODE_IDs, delimiter=",", fmt='%s')
#database_input=dataset = pd.concat([measurement_data_subset.iloc[:,1:2],nominal_coordinates,actual_coordinates], ignore_index=True,axis=1)
#%%
#Filtering co-ordinates
#x_data=measurement_data_subset.loc[(measurement_data_subset['Type'] == 'x')]
#y_data=measurement_data_subset.loc[(measurement_data_subset['Type'] == 'y')]
#z_data=measurement_data_subset.loc[(measurement_data_subset['Type'] == 'z')]
measurement_data=pd.read_csv("./Core_view_AM_Features_08_08/M3_C0-500.txt",skiprows=25,low_memory=False,sep='\t', lineterminator='\r',error_bad_lines=False)
measurement_data_subset=measurement_data.loc[(measurement_data['Name'].str[0:2] == 'SF')]
nominal_coordinates=measurement_data_subset.iloc[:,5:8]
actual_coordinates=measurement_data_subset.iloc[:,10:13]
deviations=actual_coordinates.values-nominal_coordinates.values
imputed_deviations= np.nan_to_num(deviations)
y_dev_data_filtered=imputed_deviations[:,1:2]
node_IDs=measurement_data_subset.iloc[:,1:2]
node_IDs=node_IDs['Name'].str[2:]
node_IDs=node_IDs.astype(int).values

voxel_dev_data=np.zeros((1,54,127,26,1))    

for p in range(len(point_import)):
    x_index=int(df_point_index_core_view[int(point_import[p]-1),0])
    y_index=int(df_point_index_core_view[int(point_import[p]-1),1])
    z_index=int(df_point_index_core_view[int(point_import[p]-1),2])
    voxel_dev_data[0,x_index,y_index,z_index,0]=get_dev_data(voxel_dev_data[0,x_index,y_index,z_index,0],y_dev_data_filtered[int(point_import[p]-1)],int(point_import[p]-1))

y_pred = final_model.predict(voxel_dev_data)
#log_variance=y_pred[:,5]
#variance=np.exp(log_variance)
#standard_deviation=np.sqrt(variance)
KCCs=y_pred[:,0:4]
print(KCCs)
















    
    