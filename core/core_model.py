""" Containts core classes and methods for intilillzing the deep learning 3D CNN model with different variants of the loss function, inputs are provided from the modelconfig_train.py file"""

class DLModel:	
	""" Deep Learning Model Class

		:param model_type: Type of model to be used for training 3D CNN with MSE loss, 3D CNN with hetreoskedastic aleatoric loss, 3D CNN with a mixture density network (GMM) output
		:type model_type: str (required)

		:param output_dimension: Number of output nodes for the network equal to number of KCCs for the assembly in case MSE is used as loss function
		:type output_dimension: int (required)

		:param optimizer: The optimizer to be used while model training, refer: https://keras.io/optimizers/ for more information
		:type optimizer: keras.optimizer (required) 

		:param loss_function: The loss function to be optimized by training the model, refer: https://keras.io/losses/ for more information
		:type loss_function: keras.losses (required)

		:param regularizer_coeff: The L2 norm regularisation coefficient value used in the fully connected layer of the model, refer: https://keras.io/regularizers/ for more information
		:type regularizer_coeff: float (required)

		:param output_type: The L2 norm regularisation coefficient value, refer: https://keras.io/regularizers/ for more information
		:type output_type: float (required)		

	"""
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

	def cnn_model_3d_tl(self,voxel_dim,deviation_channels):

		from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Dropout, Input
		from keras.models import Model
		from keras import regularizers

		inputs = Input(shape=(None,None,None,3,))
		cnn3d_1=Conv3D(32, kernel_size=(5,5,5),strides=(2,2,2),activation='relu')(inputs)
		cnn3d_2=Conv3D(32, kernel_size=(4,4,4),strides=(2,2,2),activation='relu')(cnn3d_1)
		cnn3d_3=Conv3D(32, kernel_size=(3,3,3),strides=(1,1,1),activation='relu')(cnn3d_2)
		max_pool3d=MaxPool3D(pool_size=(2,2,2))(cnn3d_3)
		pooled_layer=GlobalMaxPooling3D()(max_pool3d)
		dense_1=Dense(128,kernel_regularizer=regularizers.l2(self.regularizer_coeff),activation='relu')(pooled_layer)
		predictions=Dense(self.output_dimension, activation=final_layer_avt)(dense_1)
		model = Model(inputs=inputs, outputs=predictions)

		model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['mae'])
		
		return model

	def cnn_model_3d_aleatoric(self,voxel_dim,deviation_channels):

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

	def cnn_model_3d_mdn(self,voxel_dim,deviation_channels,num_of_mixtures=5):
		
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
	print('Model Deep Learning Class')