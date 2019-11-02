import numpy as np
import pandas as pd
import tensorflow as tf
import dfml
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

#Importing Dataset
print('Importing and preprocessing Cloud-of-Point Data')
dataset_0 = pd.read_csv('./Data/car_halo_run1_ydev.csv',header=None)
dataset_1 = pd.read_csv('./Data/car_halo_run2_ydev.csv',header=None)
dataset_2 = pd.read_csv('./Data/car_halo_run3_ydev.csv',header=None)
dataset_3 = pd.read_csv('./Data/car_halo_run4_ydev.csv',header=None)
dataset_4 = pd.read_csv('./Data/car_halo_run5_ydev.csv',header=None)
dataset_5 = pd.read_csv('./Data/car_halo_run6_ydev.csv',header=None)
dataset_6 = pd.read_csv('./Data/car_halo_run7_ydev.csv',header=None)
dataset_7 = pd.read_csv('./Data/car_halo_run8_ydev.csv',header=None)

#Importing Dataset and calling data import function
dataset = pd.concat([dataset_0, dataset_1,dataset_2,dataset_3,dataset_4,dataset_5,dataset_6,dataset_7], ignore_index=True)
input_conv_data,kcc_subset_dump=data_import()
print('Data import and processing completed')

#Splitting to Train and Test Set
split_ratio=0.3

X_train, X_test, y_train, y_test = train_test_split(X_in, Y_out, test_size = split_ratio)

#Calling function to build 3D CNN model
model=dfml.CNN_model_3D()

#Checkpointer to save the best model
checkpointer = ModelCheckpoint('./model/CNN_model_3D.h5', verbose=1, save_best_only='mae')

#Activating Tensorboard for Vizvalization
tensorboard = TensorBoard(log_dir='./logs',histogram_freq=1, write_graph=True, write_images=True)
history=model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=150, batch_size=32,callbacks=[tensorboard,checkpointer])

