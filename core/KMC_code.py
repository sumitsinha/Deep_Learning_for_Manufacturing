# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 15:11:46 2018

@author: Sinha_S
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 16:54:32 2018

@author: Sinha_S
"""
# Importing the libraries
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
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import plotly as py
import plotly.graph_objs as go
#%%
#Reading Node IDs
kcc_type=['kcc1','kcc2','kcc3','kcc4','kcc5']
filename=kcc_type[4]+'.csv'
#filename='kcc_combined_node.csv'
node_ids = pd.read_csv(filename)
node_ids=node_ids[node_ids.Feature_Importance!=0]
node_id_kcc=node_ids['node_ID'].tolist()

#%%
#Importing Nominal COP from the database
def import_nominal_cop(nominal_table):
    engine = create_engine('postgresql://postgres:sumit123!@10.255.1.130:5432/IPQI')
    squery='select cox,coy,coz from ' + nominal_table
    df_nom = pd.read_sql_query(squery,con=engine)
    #df_nom=df_nom.values
    return df_nom

#Importing Test part
def databaseimport(table_name):
    engine = create_engine('postgresql://postgres:sumit123!@10.255.1.130:5432/IPQI')
    squery='select cox,coy,coz from '+table_name+' order by index,id'
    df = pd.read_sql_query(squery,con=engine)
    #pattern_list.append(df)
    return df

#%%
nominal_table='car_door_halo_nominal_cop'
#table_name='tophat_check_table'
#Getting Datapoints
df_nom=import_nominal_cop(nominal_table)
#df=databaseimport(table_name)
#Calculating deviations
#df['coz']=df['coz']-df_nom['coz']
kcc_points=df_nom.iloc[node_id_kcc]
df_nom=df_nom.values
kcc_points=kcc_points.values
#%%
trace1 = go.Scatter3d(
    x=df_nom[:,0],
    y=df_nom[:,1],
    z=df_nom[:,2],
    mode='markers',
    marker=dict(
        size=2,
        
        opacity=0.8
    )
)

trace2 = go.Scatter3d(
    x=kcc_points[:,0],
    y=kcc_points[:,1],
    z=kcc_points[:,2],
    mode='markers',
    marker=dict(
        color='rgb(120, 0, 0)',
        size=8,
        symbol='circle',
        line=dict(
            color='rgb(204, 204, 204)',
            width=1
        ),
        opacity=0.7
    )
)

data = [trace1,trace2]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
py.offline.plot(fig, filename='car_halo_scatter_kcc.html')


