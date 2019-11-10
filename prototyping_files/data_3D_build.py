# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 13:06:24 2019

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
#Functions
#Importing Nominal COP from the database
def import_nominal_cop(nominal_table):
    engine = create_engine('postgresql://postgres:sumit123!@10.255.1.130:5432/IPQI')
    squery='select cox,coy,coz from ' + nominal_table
    df_nom = pd.read_sql_query(squery,con=engine)
    df_nom=df_nom.values
    return df_nom

def distance_func(x1,y1,z1,x2,y2,z2):
    dist = math.sqrt((x1-x2)**2 + (y1-y2)**2+(z1-z2)**2)
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

#%%
array_locator=np.zeros((54,127,26,3))

x_cor=2639
y_cor=3240
z_cor=940

for i in range(54):
    array_locator[i,:,:,0]=x_cor
    x_cor=x_cor-13

for j in range(127):
    array_locator[:,j,:,1]=y_cor
    y_cor=y_cor-10

for k in range(26):
    array_locator[:,:,k,2]=z_cor
    z_cor=z_cor-10

#%%

nominal_table='car_door_halo_nominal_cop'
df_nom=import_nominal_cop(nominal_table)

df_point_index=np.zeros((8047,3))

for p in tqdm(range(8047)):
        min_distance=10000
        for i in range(54):
            for j in range(127):
                for k in range(26):
                    distance=distance_func(array_locator[i,j,k,0],array_locator[i,j,k,1],array_locator[i,j,k,2],df_nom[p,2],df_nom[p,0],df_nom[p,1])         
                    if(distance<min_distance):
                        min_distance=distance
                        x_index=i
                        y_index=j
                        z_index=k
        df_point_index[p,0]=x_index
        df_point_index[p,1]=y_index
        df_point_index[p,2]=z_index


#%%
name_cop="Halo_cov_index_data.dat"
df_point_index.dump(name_cop)
#%%
#Data_pull check
#df_point_index_check = np.load("Halo_cov_index_data.dat")



















