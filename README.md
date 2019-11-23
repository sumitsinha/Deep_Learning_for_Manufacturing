# Deep Learning for Manufacturing [WORK IN PROGESS]
Overview
The Deep Learning for Manufacturing Library is built using a Tensorflow and Keras backend to build deep learning models such as 3D CNNs to enable root cause analysis and Quality Control in Manufacturing Systems. The library consists of key datasets and modules to aid model training, building and benchmarking for manufacturing systems. The current emphasis is on multi-stage sheet metal manufacturing systems.
Installation 
The library can be cloned using:
Git clone https://github.com/sumitsinha/Deep_Learning_for_Manufacturing
Datasets
The library consists of the following two key datasets:
1.	3D Cloud of Point data and process parameters for Single Part Car Halo Reinforcement – Obtained due to variations in the Measurement Station
2.	 3D Cloud of Point data and process parameters for Two part assembly for Car Door Inner and Hinge Reinforcement – Obtained due to variations in an Assembly System  

Models
The 3D CNN model termed as PointDevNet has the following layers and Parameters. 
model.add(Conv3D(32,kernel_size=(5,5,5),strides=(2,2,2),activation='relu',input_shape=(voxel_dim,voxel_dim,voxel_dim,deviation_channels)))
model.add(Conv3D(32, kernel_size=(4,4,4),strides=(2,2,2),activation='relu'))
model.add(Conv3D(32, kernel_size=(3,3,3),strides=(1,1,1),activation='relu'))
model.add(MaxPool3D(pool_size=(2,2,2)))
model.add(Flatten())
model.add(Dense(128,kernel_regularizer=regularizers.l2(0.02),activation='relu'))
model.add(Dense(self.output_dimension, activation=final_layer_avt))

The model can be trained with different loss functions depending on system behaviour and application. The key loss functions are:
•	Mean Squared Error
•	Aleatoric loss considering Heteroskedastic variance factor
•	Mixture Density Network output consisting a likelihood function of a Gaussian Mixture Model
