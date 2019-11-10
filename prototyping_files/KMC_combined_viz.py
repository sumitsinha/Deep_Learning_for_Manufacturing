# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 11:59:17 2019

@author: sinha_s
"""


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
#Importing Nominal COP from the database
def import_nominal_cop(nominal_table):
    engine = create_engine('postgresql://postgres:sumit123!@10.255.1.130:5432/IPQI')
    squery='select cox,coy,coz from ' + nominal_table
    df_nom = pd.read_sql_query(squery,con=engine)
    #df_nom=df_nom.values
    return df_nom

#%%
#Reading Node IDs
nominal_table='car_door_halo_nominal_cop'

df_nom=import_nominal_cop(nominal_table)
kcc_type=['kcc1','kcc2','kcc3','kcc4','kcc5']
dat_list = []
for i in range(len(kcc_type)):
    filename=kcc_type[i]+'.csv'
    #filename='kcc_combined_node.csv'
    node_ids = pd.read_csv(filename)
    node_ids=node_ids[node_ids.Feature_Importance!=0]
    node_id_kcc=node_ids['node_ID'].tolist()
    kcc_points=df_nom.iloc[node_id_kcc]
    kcc_points=kcc_points.values
    dat_list.append(kcc_points)
#%%
df_nom=df_nom.values
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
    x=dat_list[0][:,0],
    y=dat_list[0][:,1],
    z=dat_list[0][:,2],
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
        
trace3 = go.Scatter3d(
    x=dat_list[1][:,0],
    y=dat_list[1][:,1],
    z=dat_list[1][:,2],
    mode='markers',
    marker=dict(
        color='rgb(128, 0, 128)',
        size=8,
        symbol='circle',
        line=dict(
            color='rgb(204, 204, 204)',
            width=1
        ),
        opacity=0.7
    )
)

trace4 = go.Scatter3d(
    x=dat_list[2][:,0],
    y=dat_list[2][:,1],
    z=dat_list[2][:,2],
    mode='markers',
    marker=dict(
        color='rgb(0, 120, 0)',
        size=8,
        symbol='circle',
        line=dict(
            color='rgb(204, 204, 204)',
            width=1
        ),
        opacity=0.7
    )
)

trace5 = go.Scatter3d(
    x=dat_list[3][:,0],
    y=dat_list[3][:,1],
    z=dat_list[3][:,2],
    mode='markers',
    marker=dict(
        color='rgb(255, 165, 0)',
        size=8,
        symbol='circle',
        line=dict(
            color='rgb(204, 204, 204)',
            width=1
        ),
        opacity=0.7
    )
)
trace6 = go.Scatter3d(
    x=dat_list[4][:,0],
    y=dat_list[4][:,1],
    z=dat_list[4][:,2],
    mode='markers',
    marker=dict(
        color='rgb(0, 0, 128)',
        size=8,
        symbol='circle',
        line=dict(
            color='rgb(204, 204, 204)',
            width=1
        ),
        opacity=0.7
    )
)
data = [trace1,trace2,trace3,trace4,trace5,trace6]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
py.offline.plot(fig, filename='car_halo_scatter_kcc_combined.html')


