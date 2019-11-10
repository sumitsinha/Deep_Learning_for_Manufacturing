# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:18:50 2019

@author: Sinha_S
"""

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras import objectives
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
import pandas as pd
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
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Flatten
from tqdm import tqdm
K.clear_session()
from MDN import *
#%%
#Data Read
#Importing the dataset
df_point_index = np.load("Halo_cov_index_data.dat")
print("Initiating Data Read....")
#%%
#dataset = pd.read_csv('car_halo_run1_ydev.csv',header=None)
dataset_0 = pd.read_csv('./Data/car_halo_run1_ydev.csv',header=None)
#dataset_1 = pd.read_csv('./Data/car_halo_run2_ydev.csv',header=None)
#dataset_2 = pd.read_csv('./Data/car_halo_run3_ydev.csv',header=None)
#dataset_3 = pd.read_csv('./Data/car_halo_run4_ydev.csv',header=None)
#dataset_4 = pd.read_csv('./Data/car_halo_run5_ydev.csv',header=None)
#dataset_5 = pd.read_csv('./Data/car_halo_run6_ydev.csv',header=None)
#dataset_6 = pd.read_csv('./Data/car_halo_run7_ydev.csv',header=None)
#dataset_7 = pd.read_csv('./Data/car_halo_run8_ydev.csv',header=None)
#%%
dataset = pd.concat([dataset_0], ignore_index=True)
#measurement_noise= np.random.uniform(low=-0.1, high=0.1, size=(1,8047))
#dataset = shuffle(dataset)
#%%
kcc_data_1=dataset.iloc[:, 8047:8052]
kcc_dump=kcc_data_1.values
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
#%%
def get_dev_data(y1,y2):   
    if(abs(y1)>abs(y2)):
        y_dev=y1
    else:
        y_dev=y2
    retval=y_dev
    return retval



#%%
start_index=0
end_index=len(dataset)
#end_index=50000
length=end_index-start_index
final_halo_conv_data=np.zeros((length,54,127,26,1))
for index in tqdm(range(start_index,end_index)):
    y_point_data=dataset.iloc[index, 0:8047]
    dev_data=y_point_data.values
    measurement_noise= np.random.uniform(low=-0.1, high=0.1, size=(8047))
    dev_data=dev_data+measurement_noise
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
def generate(output, testSize, numComponents=3, outputDim=5, M=1):
	out_pi = output[:,:numComponents]
	out_sigma = output[:,numComponents:2*numComponents]
	out_mu = output[:,2*numComponents:]
	out_mu = np.reshape(out_mu, [-1, numComponents, outputDim])
	out_mu = np.transpose(out_mu, [1,0,2])
	# use softmax to normalize pi into prob distribution
	max_pi = np.amax(out_pi, 1, keepdims=True)
	out_pi = out_pi - max_pi
	out_pi = np.exp(out_pi)
	normalize_pi = 1 / (np.sum(out_pi, 1, keepdims=True))
	out_pi = normalize_pi * out_pi
	# use exponential to make sure sigma is positive
	out_sigma = np.exp(out_sigma)
	result = np.random.rand(testSize, M, outputDim)
	rn = np.random.randn(testSize, M)
	mu = 0
	std = 0
	idx = 0
	for j in range(0, M):
		for i in range(0, testSize):
		  for d in range(0, outputDim):
		    idx = np.random.choice(24, 1, p=out_pi[i])
		    mu = out_mu[idx,i,d]
		    std = out_sigma[i, idx]
		    result[i, j, d] = mu + rn[i, j]*std
	return result

def oneDim2OneDim():
    numComponents=3
    outputDim=5
    
    from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(6,12,4),strides=(3,4,1),activation='relu',input_shape=(54,127,26,1)))
    model.add(Conv3D(64, kernel_size=(2,2,3),strides=(2,2,1),activation='relu'))
    model.add(Conv3D(64, kernel_size=(1, 1,2),strides=(1,1,1),activation='relu'))
    model.add(Flatten())
    model.add(Dense(512,kernel_regularizer=regularizers.l2(0.01),activation='relu'))
    #model.add(Dropout(0.3))
    model.add(Dense(256,kernel_regularizer=regularizers.l2(0.01),activation='relu'))
    model.add(MixtureDensity(outputDim,numComponents))    
    model.compile(loss=mdn_loss(numComponents=3, outputDim=outputDim), optimizer='adam')
    checkpointer = ModelCheckpoint('./model/model_final.h5', verbose=1, save_best_only='mae')
    tensorboard = TensorBoard(log_dir='./logs',histogram_freq=1, write_graph=True, write_images=True)
    model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=10, batch_size=10,callbacks=[tensorboard,checkpointer],verbose=1)
    K.clear_session()
    global y_pred
    final_model=load_model('./model/model_final.h5')
    y_pred=final_model.predict(X_test)
    #y_pred = generate(model.predict(X_test), X_test.size)
oneDim2OneDim()
#oneDim2TwoDim()
#%%
def get_mixture_model():
    numComponents=3
    outputDim=5
    output=y_pred
    out_pi = output[:,:numComponents]
    out_sigma = output[:,numComponents:2*numComponents]
    out_mu = output[:,2*numComponents:]
    #out_mu = np.reshape(out_mu, [-1, numComponents, outputDim])
    #out_mu = np.transpose(out_mu, [1,0,2])
    # use softmax to normalize pi into prob distribution
    max_pi = np.amax(out_pi, 1, keepdims=True)
    out_pi = out_pi - max_pi
    out_pi = np.exp(out_pi)
    normalize_pi = 1 / (np.sum(out_pi, 1, keepdims=True))
    out_pi = normalize_pi * out_pi
    # use exponential to make sure sigma is positive
    out_sigma = np.exp(out_sigma)
    return out_pi,out_sigma,out_mu

out_pi,out_sigma,out_mu=get_mixture_model()
    