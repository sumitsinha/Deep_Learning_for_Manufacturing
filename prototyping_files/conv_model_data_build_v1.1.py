# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 18:55:14 2018

@author: Sinha_S
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

#%%
#Functions
#Importing Nominal COP from the database
def import_nominal_cop(nominal_table):
    engine = create_engine('postgresql://postgres:sumit123!@10.255.1.130:5432/IPQI')
    squery='select cox,coy,coz from ' + nominal_table+' order by index,id'
    df_nom = pd.read_sql_query(squery,con=engine)
    df_nom=df_nom.values
    return df_nom

#final_tophat_conv_data=np.zeros((8991,196,385,3))
import math
def distance_func(x1,y1,x2,y2):
    dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return dist
def get_dev_data(x1,y1,z1,x2,y2,z2):   
    if(abs(x1)>abs(x2)):
        x_dev=x1
    else:
        x_dev=x2
    if(abs(y1)>abs(y2)):
        y_dev=y1
    else:
        y_dev=y2    
    if(abs(z1)>abs(z2)):
        z_dev=z1
    else:
        z_dev=z2
    retval=[x_dev,y_dev,z_dev]
    return retval

#Array Creation
array_locator=np.zeros((196,385,2))
#%%
x_cor=-262
y_cor=89
for i in range(385):
    array_locator[:,i,0]=x_cor
    x_cor=x_cor+1

#%%
for j in range(196):
    array_locator[j,:,1]=y_cor
    y_cor=y_cor-1

#%%


nominal_table='tophat_nominal'
df_nom=import_nominal_cop(nominal_table)

df_point_index=np.zeros((2222,2))

for p in range(2222):
        min_distance=1000
        for i in range(196):
            for j in range(385):
                distance=distance_func(array_locator[i,j,0],array_locator[i,j,1],df_nom[p,0],df_nom[p,1])
                if(distance<min_distance):
                    min_distance=distance
                    x_index=i
                    y_index=j
        df_point_index[p,0]=x_index
        df_point_index[p,1]=y_index

#%% Data Read
#Importing the dataset
print("Initiating Data Read....")
dataset_0 = pd.read_csv('Data_pull_python/data_tophat_run1_zdev.csv',header=None)
dataset_1 = pd.read_csv('data_pull_python/data_tophat_run1_xdev.csv',header=None)
dataset_2 = pd.read_csv('data_pull_python/data_tophat_run1_ydev.csv',header=None)
print("Data Read Completed....")

#%%

kcc_data_1=dataset_0.iloc[:, 2222:2245]
kpi_data=dataset_0.iloc[:, 2245:2646]
kcc_dump=kcc_data_1.values
kpi-dump=kpi_data.values

#%%
#final_tophat_conv_data=np.zeros((8991,196,385,3))

start_index=0
end_index=8991
length=end_index-start_index
final_tophat_conv_data=np.zeros((length,196,385,3))
for index in tqdm(range(start_index,end_index)):
    x_point_data=dataset_0.iloc[index, 0:2222]
    y_point_data=dataset_1.iloc[index, 0:2222]
    z_point_data=dataset_2.iloc[index, 0:2222]
    point_data=pd.concat([x_point_data,y_point_data,z_point_data],axis=1,sort=False)
    dev_data=point_data.values
    
    cop_dev_data=np.zeros((196,385,3))    
    
    for p in range(2222):
        min_distance=1000
        x_index=int(df_point_index[p,0])
        y_index=int(df_point_index[p,1])
        cop_dev_data[x_index,y_index,:]=get_dev_data(cop_dev_data[x_index,y_index,0],cop_dev_data[x_index,y_index,1],cop_dev_data[x_index,y_index,2],dev_data[p,0],dev_data[p,1],dev_data[p,2])
    final_tophat_conv_data[index,:,:,:]=cop_dev_data

kcc_subset_dump=kcc_dump[0:end_index,:]

print('Data import and processing completed')
#%%
##Saving to array
#snippet=str(start_index);
#name_cop=snippet+"_tophat_cov_data.dat"
#final_tophat_conv_data.dump(name_cop)
#kcc_dump=kcc_data_1.values
#kcc_subset_dump=kcc_dump[0:end_index,:]
#name_cop=snippet+"_kcc_dump.dat"
#kcc_subset_dump.dump(name_cop)
##final_tophat_conv_data = np.load("tophat_cov_data.dat")
#
##%%
#import pickle
#fp = open("x.dat", "wb")
#pickle.dump(final_tophat_conv_data, fp, protocol = 4)




