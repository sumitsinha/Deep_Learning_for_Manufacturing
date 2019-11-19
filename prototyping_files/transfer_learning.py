# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:55:38 2019

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
K.clear_session()
#from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Model
#%%
print("Initiating Data Read....")
df_point_index_core_view = np.load("Halo_cov_index_data_2D_mapping_15.dat")
dataset_TL_1 = np.load("transfer_learning_dataset_v3.dat")
dataset_TL_2 = np.load("transfer_learning_dataset_v5.dat")
dataset_TL_3 = np.load("transfer_learning_dataset_v7.dat")
dataset_TL=np.concatenate([dataset_TL_1,dataset_TL_2,dataset_TL_3])
#%%
kcc_dump=dataset_TL[:, 8047:8051]
#kcc_dump[kcc_dump!=0]=1
def eliminate_rotation(voxel_cop):
    rows_diffs = np.zeros((15, 15))
    col_diffs = np.zeros((15, 15))
    for i in range(15):
        rows_diffs[i,0:14]=np.ediff1d(voxel_cop[i,:,0])
        col_diffs[0:14,i]=np.ediff1d(voxel_cop[:,i,0])  
    filtered_data = np.zeros((15, 15,2))
    filtered_data[:,:,0]=rows_diffs
    filtered_data[:,:,1]=col_diffs
    return filtered_data
#%%
import random as ra
start_index=0
#end_index=len(dataset)
end_index=len(dataset_TL)
length=end_index-start_index
final_halo_conv_data=np.zeros((length,15,15,1))
for index in tqdm(range(start_index,end_index)):
    y_point_data=dataset_TL[index, 0:8047]
    dev_data=y_point_data
    #system_noise=np.random.uniform(ra.uniform(-0.1,-0.4),ra.uniform(0.1,0.4),len(dev_data))
    dev_data=dev_data
    dev_data=dev_data[:,np.newaxis]
    cop_dev_data=np.zeros((15,15,1))    
    for i in range(15):
        for j in range(15):
            point_pool_check= np.where((df_point_index_core_view[:,1:3]==(i,j)).all(axis=1))[0]
            node_subsets=df_point_index_core_view[point_pool_check,0:1]
            #print(len(node_subsets))
            value_array=dev_data[(node_subsets[:,0]-1).astype(int),:]
            median_val=np.median(value_array)
            cop_dev_data[i,j,:]=median_val
    cop_dev_data= np.nan_to_num(cop_dev_data)
    #filtered_data=eliminate_rotation(cop_dev_data)
    final_halo_conv_data[index,:,:,:]=cop_dev_data

#%%
display=final_halo_conv_data[12,:,:,:]
display1=final_halo_conv_data[13,:,:,:]
#%%
kcc_subset_dump=kcc_dump[start_index:end_index,:]
print('Data import and processing completed')
#%%
X_in=final_halo_conv_data.reshape((length,450))
X_in=X_in[:,:,np.newaxis]
#%%
Y_out=kcc_subset_dump
X_train, X_test, y_train, y_test = train_test_split(X_in, Y_out, test_size = 0.2)
X_train=X_train[:,:,0]
X_test=X_test[:,:,0]
#mm_scaler = preprocessing.StandardScaler()
#X_train = mm_scaler.fit_transform(X_train)
#X_test=mm_scaler.transform(X_test)
#%%
transfer_model=load_model('./model/pointdevnet_demo_baseline_filtered_model_scaled.h5')
#import tensorflow as tf
#
#base_model.summary()
#transfer_model = Model(base_model.inputs, base_model.layers[-5].output)
#transfer_model.summary()
#add_model = Sequential()
#add_model.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01),input_dim=225))
#add_model.add(Dropout(0.25))
#add_model.add(Dense(32, kernel_regularizer=regularizers.l2(0.01),activation='relu'))
#add_model.add(Dropout(0.25))
#add_model.add(Dense(4, activation='linear'))
#add_model.compile(loss='mse', optimizer='RMSprop',metrics=['mae'])
#add_model.summary()
#graph=tf.get_default_graph()
#final_model = Model(inputs=transfer_model.input, outputs=add_model(transfer_model.output))

#%%
for i,layer in enumerate(transfer_model.layers):
  print(i,layer.name)
#%%
for layer in transfer_model.layers[:1]:
    layer.trainable=False
    print(layer.name)
#for layer in transfer_model.layers[3:]:
#    layer.trainable=True

#%%
checkpointer = ModelCheckpoint('./model/pointdevnet_demo_baseline_filtered_model_scaled_fine_tuned_dropout.h5', monitor='val_loss',verbose=1, save_best_only=True)
history=transfer_model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=1000, batch_size=16,callbacks=[checkpointer])
#%%
final_model=load_model('./model/pointdevnet_demo_baseline_filtered_model_scaled_fine_tuned_dropout.h5')
#%%
preds=final_model.predict(X_test)
#%%
mae_KCCs=np.zeros((4))
mse_KCCs=np.zeros((4))
r2_KCCs=np.zeros((4))
for i in range(4):
    mae_KCCs[i]=metrics.mean_absolute_error(y_test[:,i], preds[:,i])
    mse_KCCs[i]=metrics.mean_squared_error(y_test[:,i], preds[:,i])
    r2_KCCs[i] = metrics.r2_score(y_test[:,i], preds[:,i])

rmse_KCCs=np.sqrt(mse_KCCs)
print('MAE for KCC Prediction')
print(mae_KCCs)
print('MSE for KCC Prediction')
print(mse_KCCs)
print('RMSE for KCC Prediction')
print(rmse_KCCs)
print('R2 for KCC Prediction')
print(r2_KCCs)

#%%
f = K.function([final_model.layers[0].input, K.learning_phase()],
               [final_model.layers[-1].output])

def predict_with_uncertainty(f, x, n_iter=100):
    global result
    result = np.zeros((n_iter,4))

    for iter in range(n_iter):
        global temp
        temp=f([x,4])
        result[iter,:] = np.array(f([x,4]))

    prediction = result[:,0:4].mean(axis=0)
    uncertainty = result[:,0:4].var(axis=0)
    uncertainty_epistemic=np.sqrt(uncertainty)
    return prediction, uncertainty_epistemic

uncertaninty_predictions = np.zeros((y_test.shape[0],8))
from tqdm import tqdm
for i in tqdm(range(y_test.shape[0])):
    input_x=X_test[np.newaxis,i,:]
    prediction,uncertainty_epistemic=predict_with_uncertainty(f, input_x)
    uncertaninty_predictions[i,0:4]=prediction
    uncertaninty_predictions[i,4:8]=uncertainty_epistemic








