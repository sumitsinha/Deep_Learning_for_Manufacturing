# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 17:35:56 2018

@author: Sinha_S
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing, model_selection, metrics
from sklearn.utils import shuffle
from sklearn.externals import joblib
#import model_call

#%% Data Read
#Importing the dataset
print("Welcome to Model Training....")
print("Initiating Data Read....")
dataset_0 = pd.read_csv('new_data/kcc_uniform_combined0.csv',header=None)
dataset_1 = pd.read_csv('new_data/kcc_uniform_combined1.csv',header=None)
dataset_2 = pd.read_csv('new_data/kcc_uniform_combined2.csv',header=None)
#%%
dataset = pd.concat([dataset_0, dataset_1,dataset_2], ignore_index=True)
dataset = shuffle(dataset)
#%%
print("Data imported successfully.....")
#%%
point_data=dataset.iloc[:, 0:2601]
#%%
dataset_kcc=dataset.iloc[:, 2601:2609]
dataset_kcc.columns = ['kcc_pv','kcc_xtra','kcc_ytra','kcc_zrot','kcc_t1_tra_x','kcc_t1_tra_y','kcc_t1_tra_z','kcc_t2fa_z']
dataset_kcc['label1']=np.where(dataset_kcc['kcc_xtra']>=3.5, 1, 0)
dataset_kcc['label2']=np.where(dataset_kcc['kcc_ytra']>=3.5, 1, 0)
dataset_kcc['label3']=np.where(dataset_kcc['kcc_zrot']>=0.75, 1, 0)
dataset_kcc['label4']=np.where(dataset_kcc['kcc_t1_tra_x']>=5, 1, 0)
dataset_kcc['label5']=np.where(dataset_kcc['kcc_t1_tra_y']>=3.5, 1, 0)
dataset_kcc['label6']=np.where(dataset_kcc['kcc_t1_tra_z']<=-0.1, 1, 0)
dataset_kcc['label7']=np.where(dataset_kcc['kcc_t2fa_z']>=0.1, 1, 0)
dataset_kcc['combined_defects']=dataset_kcc['label1']+dataset_kcc['label2']+dataset_kcc['label3']+dataset_kcc['label4']+dataset_kcc['label5']+dataset_kcc['label6']+dataset_kcc['label7']

#%%
combined_dataset=pd.concat([point_data,dataset_kcc],axis=1,sort=False)
#%%
kcc_type=['kcc_xtra','kcc_ytra','kcc_zrot','kcc_t1_tra_x','kcc_t1_tra_y','kcc_t1_tra_z','kcc_t2fa_z']
TOTAL_KCCs=7
for i in range(TOTAL_KCCs):
    print('Feature Selection for :', kcc_type[(i)])
    kcc_label='label'+str(i+1)    
    #%%
    point_data_f=combined_dataset.iloc[:, 0:2601]
    dataset_kcc_f=combined_dataset[kcc_type[(i)]]
    
    #%% Preparing dataset
    from sklearn.model_selection import train_test_split
    train_X, test_X, train_y, test_y = train_test_split(point_data_f, dataset_kcc_f, test_size = 0.2)
    
    #%% Use XGB and Random Forest for point importance
    import xgboost as xgb
    #from sklearn.ensemble import RandomForestRegressor

    #%%
    train=train_X
    target=train_y
    train.index=range(0,train.shape[0])
    target.index=range(0,train.shape[0])
    
    #%%
    print('Tree Based Model Training for :', kcc_type[(i)])
    #model=RandomForestRegressor(n_estimators=1000,max_depth=700,n_jobs=-1,verbose=True)
    model=xgb.XGBRegressor(colsample_bytree=0.4,gamma=0.045,learning_rate=0.07,max_depth=500,min_child_weight=1.5,n_estimators=500,reg_alpha=0.65,reg_lambda=0.45,subsample=0.95,n_jobs=-1,verbose=True)
    model.fit(train,target)
    #%%
    y_pred = model.predict(test_X)
    mae=metrics.mean_absolute_error(test_y, y_pred)
    print('The Mae for feature selection....')
    print(mae)
    filename = kcc_type[(i)]+'_XGB_model.sav'
    joblib.dump(model, filename)
    print('Trained Model Saved to disk....')
    
    #%%
    thresholds = model.feature_importances_
    sorted_thresholds=np.sort(thresholds)
    #%%
    node_id=np.arange(2601)
    node_IDs = pd.DataFrame(thresholds, index=node_id)
    node_IDs.columns=['Feature_Importance']
    node_IDs.index.name='node_ID'
    #%%
    node_IDs = node_IDs.sort_values('Feature_Importance', ascending=False)
    filtered_nodeIDs=node_IDs.loc[node_IDs['Feature_Importance'] != 0]
    node_ID_list = filtered_nodeIDs.index.tolist()
    filename=kcc_type[(i)]+'.csv'
    print('Saving selected features to disk...')
    filtered_nodeIDs.to_csv(filename)
    #%%
    filtered_point_data=point_data_f[node_ID_list]
    num_points=len(node_ID_list)
    #print('Building Deep Learning Model on Selected Features :', kcc_type[(i)])
    dataset_kcc_f=dataset_kcc_f.values
    dataset_kcc_f=dataset_kcc_f[:,np.newaxis]
    #model_call.dl_model(filtered_point_data.values,dataset_kcc_f,kcc_label,num_points)

















