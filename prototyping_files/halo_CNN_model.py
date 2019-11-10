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
#%%
#dataset = pd.read_csv('car_halo_run1_ydev.csv',header=None)
dataset_0 = pd.read_csv('./Data/car_halo_run1_ydev.csv',header=None)
dataset_1 = pd.read_csv('./Data/car_halo_run2_ydev.csv',header=None)
dataset_2 = pd.read_csv('./Data/car_halo_run3_ydev.csv',header=None)
dataset_3 = pd.read_csv('./Data/car_halo_run4_ydev.csv',header=None)
dataset_4 = pd.read_csv('./Data/car_halo_run5_ydev.csv',header=None)
dataset_5 = pd.read_csv('./Data/car_halo_run6_ydev.csv',header=None)
dataset_6 = pd.read_csv('./Data/car_halo_run7_ydev.csv',header=None)
dataset_7 = pd.read_csv('./Data/car_halo_run8_ydev.csv',header=None)
#%%
dataset = pd.concat([dataset_0, dataset_1,dataset_2,dataset_3,dataset_4,dataset_5,dataset_6,dataset_7], ignore_index=True)
#dataset = shuffle(dataset)
#%%
kcc_data_1=dataset.iloc[:, 8047:8052]
kcc_dump=kcc_data_1.values

#%%
start_index=0
#end_index=len(dataset)
end_index=50000
length=end_index-start_index
final_halo_conv_data=np.zeros((length,54,127,26,1))
for index in tqdm(range(start_index,end_index)):
    y_point_data=dataset.iloc[index, 0:8047]
    dev_data=y_point_data.values
    
    cop_dev_data=np.zeros((54,127,26,1))    
    
    for p in range(8047):
        x_index=int(df_point_index[p,0])
        y_index=int(df_point_index[p,1])
        z_index=int(df_point_index[p,2])
        cop_dev_data[x_index,y_index,z_index,0]=get_dev_data(cop_dev_data[x_index,y_index,z_index,0],dev_data[p])
    final_halo_conv_data[index,:,:,:]=cop_dev_data

kcc_subset_dump=kcc_dump[start_index:end_index,:]

print('Data import and processing completed')
#%%
X_in=final_halo_conv_data
Y_out=kcc_subset_dump
X_train, X_test, y_train, y_test = train_test_split(X_in, Y_out, test_size = 0.2)
#%%
#Model Build
K.clear_session()
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
model = Sequential()
model.add(Conv3D(32, kernel_size=(6,12,4),strides=(3,4,1),activation='relu',input_shape=(54,127,26,1)))
model.add(Conv3D(64, kernel_size=(2,2,3),strides=(2,2,1),activation='relu'))
model.add(Conv3D(64, kernel_size=(1, 1,2),strides=(1,1,1),activation='relu'))
model.add(Flatten())
model.add(Dense(512,kernel_regularizer=regularizers.l2(0.01),activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(256,kernel_regularizer=regularizers.l2(0.01),activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(5, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

#%%
#Model Training
checkpointer = ModelCheckpoint('./model/model_final.h5', verbose=1, save_best_only='mae')
tensorboard = TensorBoard(log_dir='./logs',histogram_freq=1, write_graph=True, write_images=True)
history=model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=50, batch_size=5,callbacks=[tensorboard,checkpointer])

#%%
#Loading Best Model
K.clear_session()
#%%
import tensorflow as tf
with tf.device('/cpu:0'):
    final_model=load_model('./model/model_final.h5')
#%%
y_pred = final_model.predict(X_test)
#%%
np.savetxt('logs/pred_values.csv', y_pred, delimiter=",")
np.savetxt('logs/test_values.csv', y_test, delimiter=",")

mae_KCCs=np.zeros((5))
mse_KCCs=np.zeros((5))
r2_KCCs=np.zeros((5))
for i in range(5):
    mae_KCCs[i]=metrics.mean_absolute_error(y_test[:,i], y_pred[:,i])
    mse_KCCs[i]=metrics.mean_squared_error(y_test[:,i], y_pred[:,i])
    r2_KCCs[i] = metrics.r2_score(y_test[:,i], y_pred[:,i])

rmse_KCCs=np.sqrt(mse_KCCs)
print('MAE for KCC Prediction')
print(mae_KCCs)
print('MSE for KCC Prediction')
print(mse_KCCs)
print('RMSE for KCC Prediction')
print(rmse_KCCs)
print('R2 for KCC Prediction')
print(r2_KCCs)

kcc_type=['kcc1','kcc2','kcc3','kcc4','kcc5']
#accuracy_metrics=np.concatenate((kcc_type,mae_KCCs,mse_KCCs,rmse_KCCs,r2_KCCs),axis=0)
accuracy_metrics_df=pd.DataFrame({'KCC_ID':kcc_type,'MAE':mae_KCCs,'MSE':mse_KCCs,'RMSE':rmse_KCCs,'R2':r2_KCCs})
#accuracy_metrics_df.columns = ['KCC_ID','MAE','MSE','RMSE','R2']
accuracy_metrics_df.to_csv('logs/metrics.csv')
#%%
#Plotting data
# summarize history for accuracy
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model MAE`')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('plots/accuracy.png')
plt.clf()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('plots/loss.png')
plt.clf()
#%%
train_accuracy=np.array(history.history['mean_absolute_error'])
train_accuracy=train_accuracy[:,np.newaxis]
test_accuracy=np.array(history.history['val_mean_absolute_error'])
test_accuracy=test_accuracy[:,np.newaxis]
loss_train=np.array(history.history['loss'])
loss_train=loss_train[:,np.newaxis]
loss_test=np.array(history.history['val_loss'])
loss_test=loss_test[:,np.newaxis]

#%%
final_data=np.concatenate((train_accuracy,test_accuracy,loss_train,loss_test),axis=1)
df = pd.DataFrame(final_data)
df.columns = ['train_accuracy','test_accuracy','loss_train','loss_test']
df.to_csv('logs/log_out.csv')
print('Model logged to log folder')












































