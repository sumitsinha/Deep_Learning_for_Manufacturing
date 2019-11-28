class DLModel:	

	def __init__(self,model_type,output_dimension,optimizer,loss_function,regularizer_coeff,output_type='regression'):
		self.output_dimension=output_dimension
		self.model_type=model_type
		self.optimizer=optimizer
		self.loss_function=loss_function
		self.regularizer_coeff=regularizer_coeff
		self.output_type=output_type

	def cnn_model_3d(self,voxel_dim,deviation_channels):
		
		from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Dropout
		from keras.models import Sequential
		from keras import regularizers

		if(self.output_type=="regression"):
			final_layer_avt='linear'

		if(self.output_type=="classification"):
			final_layer_avt='softmax'

		model = Sequential()
		model.add(Conv3D(32, kernel_size=(5,5,5),strides=(2,2,2),activation='relu',input_shape=(voxel_dim,voxel_dim,voxel_dim,deviation_channels)))
		model.add(Conv3D(32, kernel_size=(4,4,4),strides=(2,2,2),activation='relu'))
		model.add(Conv3D(32, kernel_size=(3,3,3),strides=(1,1,1),activation='relu'))
		model.add(MaxPool3D(pool_size=(2,2,2)))
		model.add(Flatten())
		model.add(Dense(128,kernel_regularizer=regularizers.l2(self.regularizer_coeff),activation='relu'))
		#model.add(Dropout(0.2))
		model.add(Dense(self.output_dimension, activation=final_layer_avt))
		model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['mae'])

		print("3D CNN model succssesfully compiled")
		return model

	def cnn_model_3d_aleatoric(self,voxel_dim=64,deviation_channels=1):

		if(self.model_type=="regression"):
			final_layer_avt='linear'

		if(self.model_type=="classification"):
			final_layer_avt='softmax'

		def myloss(y_true, y_pred):
		    prediction = y_pred[:,0:self.output_dimension]
		    log_variance = y_pred[:,self.output_dimension:self.output_dimension+1]
		    loss = tf.reduce_mean(0.5 * tf.exp(-1 * log_variance) * tf.square(tf.abs(y_true - prediction))+ 0.5 * log_variance)
		    return loss
		
		from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
		from keras.models import Sequential

		model = Sequential()
		model.add(Conv3D(32, kernel_size=(5,5,5),strides=(2,2,2),activation='relu',input_shape=(voxel_dim,voxel_dim,voxel_dim,deviation_channels)))
		model.add(Conv3D(32, kernel_size=(4,4,4),strides=(2,2,2),activation='relu'))
		model.add(Conv3D(32, kernel_size=(3,3,3),strides=(1,1,1),activation='relu'))
		model.add(MaxPool3D(pool_size=(2,2,2)))
		model.add(Flatten())
		model.add(Dense(128,kernel_regularizer=regularizers.l2(0.02),activation='relu'))
		#model.add(Dropout(0.3))
		model.add(Dense(self.output_dimension, activation=final_layer_avt))
		model.compile(loss=myloss, optimizer='adam', metrics=['mae'])

		print("3D CNN model Aleatoric succssesfully compiled")
		return model

	def cnn_model_3d_mdn(self,voxel_dim=64,deviation_channels=1,num_of_mixtures=5):
		
		assert self.model_type=="regression","Mixture Density Network Should be a Regression Model"

		model = Sequential()
		model.add(Conv3D(32, kernel_size=(5,5,5),strides=(2,2,2),activation='relu',input_shape=(voxel_dim,voxel_dim,voxel_dim,deviation_channels)))
		model.add(Conv3D(32, kernel_size=(4,4,4),strides=(2,2,2),activation='relu'))
		model.add(Conv3D(32, kernel_size=(3,3,3),strides=(1,1,1),activation='relu'))
		model.add(MaxPool3D(pool_size=(2,2,2)))
		model.add(Flatten())
		model.add(Dense(128,kernel_regularizer=regularizers.l2(0.02),activation='relu'))
		#model.add(Dropout(0.3))
		model.add(Dense(self.output_dimension, activation=final_layer_avt))
		model.add(mdn.MDN(self.output_dimension, num_of_mixtures))
		model.compile(loss=mdn.get_mixture_loss_func(self.output_dimension,num_of_mixtures), optimizer='adam')

		print("3D CNN Mixture Density Network model succssesfully compiled")
		return model

if (__name__=="__main__"):
	print('Model Build')