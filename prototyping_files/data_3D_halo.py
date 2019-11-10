# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 11:41:13 2019

@author: sinha_s
"""

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
import math
#%%
def get_dev_data(y1,y2):   
    if(abs(y1)>abs(y2)):
        y_dev=y1
    else:
        y_dev=y2
    retval=y_dev
    return retval

#%%
#Data Read
#Importing the dataset

df_point_index = np.load("Halo_cov_index_data.dat")
print("Initiating Data Read....")
dataset_0 = pd.read_csv('car_halo_run1_xdev.csv',header=None)
dataset_1 = pd.read_csv('car_halo_run1_ydev.csv',header=None)
dataset_2 = pd.read_csv('car_halo_run1_zdev.csv',header=None)
print("Data Read Completed....")

kcc_data_1=dataset_0.iloc[:, 8047:8052]
kcc_dump=kcc_data_1.values

#%%
start_index=0
end_index=9
length=end_index-start_index
final_halo_conv_data=np.zeros((length,54,127,26))
for index in tqdm(range(start_index,end_index)):
    y_point_data=dataset_1.iloc[index, 0:8047]
    dev_data=y_point_data.values
    
    cop_dev_data=np.zeros((54,127,26))    
    
    for p in range(8047):
        x_index=int(df_point_index[p,0])
        y_index=int(df_point_index[p,1])
        z_index=int(df_point_index[p,2])
        cop_dev_data[x_index,y_index,z_index]=get_dev_data(cop_dev_data[x_index,y_index,z_index],dev_data[p])
    final_halo_conv_data[index,:,:,:]=cop_dev_data

kcc_subset_dump=kcc_dump[0:end_index,:]

print('Data import and processing completed')

#%%
#DATA Quality Check
grid_cop=np.zeros((3499,3))
index=0
for i in range(54):
    for j in range(127):
        for k in range(26):
            if(cop_dev_data[i,j,k]!=0):
                grid_cop[index,0]=i
                grid_cop[index,1]=j
                grid_cop[index,2]=k
                index=index+1

#%%
#Importing Nominal COP from the database
def import_nominal_cop(nominal_table):
    engine = create_engine('postgresql://postgres:sumit123!@10.255.1.130:5432/IPQI')
    squery='select cox,coy,coz from ' + nominal_table
    df_nom = pd.read_sql_query(squery,con=engine)
    df_nom=df_nom.values
    return df_nom

nominal_table='car_door_halo_nominal_cop'
df_nom=import_nominal_cop(nominal_table)

#%%
grip_cop_values=np.zeros((3494,3))

for i in range(3494):
    for j in range(8047):
        if((grid_cop[i,0]==df_point_index[j,0] and grid_cop[i,1]==df_point_index[j,1] and grid_cop[i,2]==df_point_index[j,2])):
            grip_cop_values[i,:]=df_nom[j,:]
            

#%%
import plotly as py
import plotly.graph_objs as go
trace1 = go.Scatter3d(
    x=grip_cop_values[0:3494,0],
    y=grip_cop_values[0:3494,1],
    z=grip_cop_values[0:3494,2],
    mode='markers',
    marker=dict(
        size=5,
        line=dict(
            color='rgba(217, 217, 217, 5)',
            width=0.1
        ),
        opacity=1
    )
)
        

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
py.offline.plot(fig, filename='car_halo_scatter.html')











































