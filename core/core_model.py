from keras.layers import Conv3D, MaxPool3D, Flatten, Dense

class dlmf(object):
	
	def __init__(self, output_dimension,model_type='regression'):
		self.output_dimension=output_dimension
		self.model_type=model_type

	def CNN_model_3D(args):
		model = Sequential()
		model.add(Conv3D(32, kernel_size=(5,5,5),strides=(2,2,2),activation='relu',input_shape=(x_dim,y_dim,z_dim,deviation_channels)))
		model.add(Conv3D(32, kernel_size=(4,4,4),strides=(2,2,2),activation='relu'))
		model.add(Conv3D(32, kernel_size=(3,3,3),strides=(1,1,1),activation='relu'))
		model.add(MaxPool3D(pool_size=(2,2,2)))
		model.add(Flatten())
		#model.add(Dropout(0.3))
		model.add(Dense(128,kernel_regularizer=regularizers.l2(0.02),activation='relu'))
		#model.add(Dropout(0.3))
		model.add(Dense(output_dimension, activation='linear'))
		model.compile(loss='mse', optimizer='adam', metrics=['mae'])

		print("3D CNN model succssesfully compiled")
		return model

	if __name__=="__main__"
	
		config_parameters=TrainConfig_params()
		
		x_dim=config.x_dim
		y_dim=config.y_dim
		z_dim=config.z_dim

		model_type=config.model_type