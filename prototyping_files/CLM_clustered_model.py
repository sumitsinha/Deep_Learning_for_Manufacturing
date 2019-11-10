# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 11:37:56 2018

@author: Sinha_S
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from keras import regularizers
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from keras import backend as K
from sklearn.externals import joblib
K.clear_session()

#%% Data Read
print("Welcome to Model Training....")
print("Initiating Data Read....")

dataset_0 = pd.read_csv('new_data/kcc_uniform_combined0.csv',header=None)
dataset_1 = pd.read_csv('new_data/kcc_uniform_combined1.csv',header=None)
dataset_2 = pd.read_csv('new_data/kcc_uniform_combined2.csv',header=None)
#%%
dataset = pd.concat([dataset_0, dataset_1,dataset_2], ignore_index=True)
dataset = shuffle(dataset)
#%%

point_data=dataset.iloc[:, 0:2601]
dataset_kcc=dataset.iloc[:, 2601:2609]
dataset_kcc.columns = ['kcc_pv','kcc_xtra','kcc_ytra','kcc_zrot','kcc_t1_tra_x','kcc_t1_tra_y','kcc_t1_tra_z','kcc_t2fa_z']
dataset_kcc['label1']=np.where(dataset_kcc['kcc_xtra']>=3, 1, 0)
dataset_kcc['label2']=np.where(dataset_kcc['kcc_ytra']>=3, 1, 0)
dataset_kcc['label3']=np.where(dataset_kcc['kcc_zrot']>=(0.75*0.0175), 1, 0)
dataset_kcc['label4']=np.where(dataset_kcc['kcc_t1_tra_x']>=5, 1, 0)
dataset_kcc['label5']=np.where(dataset_kcc['kcc_t1_tra_y']>=3.5, 1, 0)
dataset_kcc['label6']=np.where(dataset_kcc['kcc_t1_tra_z']<=-0.1, 1, 0)
dataset_kcc['label7']=np.where(dataset_kcc['kcc_t2fa_z']>=0.1, 1, 0)
dataset_kcc['combined_defects']=dataset_kcc['label1']+dataset_kcc['label2']+dataset_kcc['label3']+dataset_kcc['label4']+dataset_kcc['label5']+dataset_kcc['label6']+dataset_kcc['label7']
combined_dataset=pd.concat([point_data,dataset_kcc],axis=1,sort=False)

#%%
point_data_f=combined_dataset.iloc[:, 0:2601]
kcc_type=['kcc_xtra','kcc_ytra','kcc_zrot','kcc_t1_tra_x','kcc_t1_tra_y','kcc_t1_tra_z','kcc_t2fa_z']
TOTAL_KCCs=len(kcc_type)
for i in range(TOTAL_KCCs):
    print('Feature Selection for :', kcc_type[(i)])
    kcc_label='label'+str(i+1)
    kcc_index=kcc_label
    filename= kcc_type[(i)]+'.csv'
    feature_data = pd.read_csv(filename)
    feature_data=feature_data[feature_data.Feature_Importance>.0005]
    node_ID_list=feature_data['node_ID'].tolist()
    num_points=len(node_ID_list)
    filtered_point_data=point_data_f[node_ID_list]
    dataset_kcc_f=combined_dataset[kcc_type[(i)]]
    X=filtered_point_data.values
    Y_out=dataset_kcc_f.values
    Y_out=Y_out[:,np.newaxis]
    #Clearning Prerun Session
    from keras import backend as K
    K.clear_session()
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y_out, test_size = 0.2)
    sc_x = StandardScaler()
    X_train = sc_x.fit_transform(X_train)
    scaler_filename_x = "scaler_kcc_x.save"+kcc_label
    joblib.dump(sc_x, scaler_filename_x)
    X_test = sc_x.transform(X_test)
    print("Data prepared successfully.....")
    print("Training on following number of features")
    print(num_points)
    
    #Build Rergression Model Model
    model = Sequential()
    model.add(Dense(512, input_dim=num_points,kernel_regularizer=regularizers.l2(0.01),activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(256,kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='linear'))
    # Compile model
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    #Fitting to train Data
    filename='./model/model'+kcc_index+'-{epoch:03d}.h5'
    checkpointer = ModelCheckpoint(filename, verbose=1, save_best_only='true')
    tensorboard = TensorBoard(log_dir='./logs',histogram_freq=1, write_graph=True, write_images=True)
    history=model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=100, batch_size=10, callbacks=[tensorboard,checkpointer])
    #%%
    #Predicting From Model
    preds = model.predict(X_test)
    mae=metrics.mean_absolute_error(y_test, preds)
    print('The MAE for KCC Prediction....')
    print(mae)
    mse_kcc=(((y_test-preds)**2).mean(axis=None))
    rmse_kcc=mse_kcc**(0.5)
    print('The RMSE for KCC Prediction...')
    print(rmse_kcc)
    
    #Plotting data
    # summarize history for accuracy
    print(history.history.keys())
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('model MAE`')
    plt.ylabel('MAE')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('plots/accuracy_'+kcc_index +'.png')
    #plt.show()
    plt.clf()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('plots/loss_'+kcc_index +'.png')
    #plt.show()
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
    
    final_data=np.concatenate((train_accuracy,test_accuracy,loss_train,loss_test),axis=1)
    df = pd.DataFrame(final_data)
    df.columns = ['train_accuracy','test_accuracy','loss_train','loss_test']
    
    df.to_csv('logs/log_out'+kcc_index+'.csv')
    print('Model logged to log folder..... ', kcc_index)
    
    
    
    
