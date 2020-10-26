""" Contains core classes and methods for initializing Probabilistic deep learning 3D CNN model with different variants of the loss function, inputs are provided from the modelconfig_train.py file"""

class Multi_Head_DLModel:    
	""" Deep Learning Model Class

		:param model_type: Type of model to be used for training 3D CNN with MSE loss, 3D CNN with hetreoskedastic aleatoric loss, 3D CNN with a mixture density network (GMM) output
		:type model_type: str (required)

		:param output_dimension: Number of output nodes for the network equal to number of KCCs for the assembly in case MSE is used as loss function
		:type output_dimension: int (required)

		:param optimizer: The optimizer to be used while model training, refer: https://keras.io/optimizers/ for more information
		:type optimizer: keras.optimizer (required) 

		:param loss_function: The loss function to be optimized by training the model, refer: https://keras.io/losses/ for more information
		:type loss_function: keras.losses (required)

		:param regularizer_coeff: The L2 norm regularization coefficient value used in the penultimate fully connected layer of the model, refer: https://keras.io/regularizers/ for more information
		:type regularizer_coeff: float (required)

		:param output_type: The output type of the model which can be regression or classification, this is used to define the output layer of the model, defaults to regression (classification: softmax, regression: linear)
		:type output_type: str      

	"""
	def __init__(self,model_type,heads,output_dimension,categorical_kccs,output_type='regression',regularizer_coeff=0.01):
		self.output_dimension=output_dimension
		self.categorical_kccs=categorical_kccs
		self.model_type=model_type
		self.output_type=output_type
		self.heads=heads
		self.regularizer_coeff=regularizer_coeff


	def multi_head_bayes_cnn_model_3d(self,voxel_dim,deviation_channels):
		"""Build the 3D Model using the specified loss function, the inputs are parsed from the assemblyconfig_<case_study_name>.py file

			:param voxel_dim: The voxel dimension of the input, required to build input to the 3D CNN model
			:type voxel_dim: int (required)

			:param voxel_channels: The number of voxel channels in the input structure, required to build input to the 3D CNN model
			:type voxel_channels: int (required)
		"""
		negloglik = lambda y, rv_y: -rv_y.log_prob(y)
		
		aleatoric_std=0.0001
		
		aleatoric_tensor=[aleatoric_std] * self.output_dimension
		#constant aleatoric uncertainty

		import tensorflow as tf
		import tensorflow_probability as tfp
		tfd = tfp.distributions
		
		def _softplus_inverse(x):
  			"""Helper which computes the function inverse of `tf.nn.softplus`."""
  			return tf.math.log(tf.math.expm1(x))
		
		if(self.output_type=="regression"):
			final_layer_avt='linear'

		if(self.output_type=="classification"):
			final_layer_avt='softmax'

		data_in=[None] * self.heads
		conv_3d_1=[None] * self.heads
		conv_3d_2=[None] * self.heads
		conv_3d_3=[None] * self.heads
		max_pool=[None] * self.heads
		flat=[None] * self.heads

		for i in range(self.heads):
			data_in[i]=tf.keras.layers.Input(shape=(voxel_dim,voxel_dim,voxel_dim,deviation_channels))
			conv_3d_1[i]=tfp.layers.Convolution3DFlipout(32, kernel_size=(5,5,5),strides=(2,2,2),activation=tf.nn.relu)(data_in[i])
			conv_3d_2[i]=tfp.layers.Convolution3DFlipout(32, kernel_size=(4,4,4),strides=(2,2,2),activation=tf.nn.relu)(conv_3d_1[i])
			conv_3d_3[i]=tfp.layers.Convolution3DFlipout(32, kernel_size=(3,3,3),strides=(1,1,1),activation=tf.nn.relu)(conv_3d_2[i])
			max_pool[i]=tf.keras.layers.MaxPooling3D(pool_size=[2, 2, 2])(conv_3d_3[i])
			flat[i]=tf.keras.layers.Flatten()(max_pool[i])

		merge = tf.keras.layers.concatenate(flat)

		#hidden_1=tfp.layers.DenseFlipout(128,activation=tf.nn.relu)(merge)
		hidden_2=tfp.layers.DenseFlipout(64,activation=tf.nn.relu)(merge)
		output=tfp.layers.DistributionLambda(lambda t: tfd.MultivariateNormalDiag(loc=t[..., :self.output_dimension], scale_diag=aleatoric_tensor))(hidden_2)

		model=tf.keras.Model(inputs=data_in,outputs=output)

		model.compile(optimizer=tf.keras.optimizers.Adam(),experimental_run_tf_function=False,loss=negloglik,metrics=[tf.keras.metrics.MeanAbsoluteError()])
		print("3D CNN model successfully compiled")

		print(model.summary())
		
		from tensorflow.keras.utils import plot_model
		plot_model(model, to_file='../pre_trained_models/probablistic_bayes_models/pointdevnet_model.png')
		
		return model

	def multi_head_standard_cnn_model_3d(self,voxel_dim,deviation_channels):
		"""Build the 3D Model using the specified loss function, the inputs are parsed from the assemblyconfig_<case_study_name>.py file

			:param voxel_dim: The voxel dimension of the input, required to build input to the 3D CNN model
			:type voxel_dim: int (required)

			:param voxel_channels: The number of voxel channels in the input structure, required to build input to the 3D CNN model
			:type voxel_channels: int (required)
		"""
		negloglik = lambda y, rv_y: -rv_y.log_prob(y)
		
		aleatoric_std=0.0001
		
		aleatoric_tensor=[aleatoric_std] * self.output_dimension
		#constant aleatoric uncertainty

		import tensorflow as tf
		import tensorflow_probability as tfp
		tfd = tfp.distributions
		
		def _softplus_inverse(x):
  			"""Helper which computes the function inverse of `tf.nn.softplus`."""
  			return tf.math.log(tf.math.expm1(x))
		
		if(self.output_type=="regression"):
			final_layer_avt='linear'

		if(self.output_type=="classification"):
			final_layer_avt='softmax'

		data_in=[None] * self.heads
		conv_3d_1=[None] * self.heads
		conv_3d_2=[None] * self.heads
		conv_3d_3=[None] * self.heads
		max_pool=[None] * self.heads
		flat=[None] * self.heads
		conv_3d_dropout_2=[None] * self.heads

		for i in range(self.heads):
			data_in[i]=tf.keras.layers.Input(shape=(voxel_dim,voxel_dim,voxel_dim,deviation_channels))
			conv_3d_1[i]=tf.keras.layers.Convolution3D(32, kernel_size=(5,5,5),strides=(2,2,2),activation=tf.nn.relu)(data_in[i])
			conv_3d_2[i]=tf.keras.layers.Convolution3D(32, kernel_size=(4,4,4),strides=(2,2,2),activation=tf.nn.relu)(conv_3d_1[i])
			conv_3d_dropout_2[i]=tf.keras.layers.Dropout(0.1)(conv_3d_2[i])
			conv_3d_3[i]=tf.keras.layers.Convolution3D(32, kernel_size=(3,3,3),strides=(1,1,1),activation=tf.nn.relu)(conv_3d_dropout_2[i])
			max_pool[i]=tf.keras.layers.MaxPooling3D(pool_size=[2, 2, 2])(conv_3d_3[i])
			flat[i]=tf.keras.layers.Flatten()(max_pool[i])

		merge = tf.keras.layers.concatenate(flat)
		dropout_merge=tf.keras.layers.Dropout(0.2)(merge)
		hidden_1=tf.keras.layers.Dense(64,kernel_regularizer=tf.keras.regularizers.l2(l=self.regularizer_coeff),activation=tf.nn.relu)(dropout_merge)
		hidden_2=tf.keras.layers.Dense(64,kernel_regularizer=tf.keras.regularizers.l2(l=self.regularizer_coeff),activation=tf.nn.relu)(hidden_1)
		
		output_reg=tf.keras.layers.Dense(self.output_dimension-self.categorical_kccs,name='regression_outputs')(hidden_2)
		output_cla=tf.keras.layers.Dense(self.categorical_kccs,activation='sigmoid',name='classification_outputs')(hidden_2)
		
		output=[output_reg,output_cla]

		model=tf.keras.Model(inputs=data_in,outputs=output)
		
		bin_crossentropy=tf.keras.losses.BinaryCrossentropy()
		mse_basic = tf.keras.losses.MeanSquaredError()

		overall_loss_dict={
			"regression_outputs":mse_basic,
			"classification_outputs":bin_crossentropy,
		}

		overall_loss_weights={
			"regression_outputs":1.0,
			"classification_outputs":1.0,
		}

		overall_metrics_dict={
		"regression_outputs":[tf.keras.metrics.MeanAbsoluteError()],
		"classification_outputs":[tf.keras.metrics.CategoricalAccuracy()],
		}

		model.compile(optimizer=tf.keras.optimizers.Adam(),experimental_run_tf_function=False,loss=overall_loss_dict,metrics=overall_metrics_dict,loss_weights=overall_loss_weights)
		print("3D CNN model successfully compiled")

		print(model.summary())
		
		from tensorflow.keras.utils import plot_model
		#plot_model(model, to_file='../pre_trained_models/deterministic_models/pointdevnet_model.png')
		
		return model

	def multi_head_shared_standard_cnn_model_3d(self,voxel_dim,deviation_channels):
		"""Build the 3D Model using the specified loss function, the inputs are parsed from the assemblyconfig_<case_study_name>.py file

			:param voxel_dim: The voxel dimension of the input, required to build input to the 3D CNN model
			:type voxel_dim: int (required)

			:param voxel_channels: The number of voxel channels in the input structure, required to build input to the 3D CNN model
			:type voxel_channels: int (required)
		"""
		negloglik = lambda y, rv_y: -rv_y.log_prob(y)
		
		aleatoric_std=0.0001
		
		aleatoric_tensor=[aleatoric_std] * self.output_dimension
		#constant aleatoric uncertainty

		import tensorflow as tf
		import tensorflow_probability as tfp
		tfd = tfp.distributions
		
		def _softplus_inverse(x):
  			"""Helper which computes the function inverse of `tf.nn.softplus`."""
  			return tf.math.log(tf.math.expm1(x))
		
		if(self.output_type=="regression"):
			final_layer_avt='linear'

		if(self.output_type=="classification"):
			final_layer_avt='softmax'

		data_in=[None] * self.heads
		flat=[None] * self.heads

		for i in range(self.heads):
			data_in[i]=tf.keras.layers.Input(shape=(voxel_dim,voxel_dim,voxel_dim,deviation_channels))
		
		feature_extractor = tf.keras.Sequential()
		feature_extractor.add(tf.keras.layers.Conv3D(32, kernel_size=(5,5,5),strides=(2,2,2),activation='relu',input_shape=(voxel_dim,voxel_dim,voxel_dim,deviation_channels)))
		feature_extractor.add(tf.keras.layers.Conv3D(32, kernel_size=(4,4,4),strides=(2,2,2),activation='relu'))
		feature_extractor.add(tf.keras.layers.Conv3D(32, kernel_size=(3,3,3),strides=(1,1,1),activation='relu'))
		feature_extractor.add(tf.keras.layers.MaxPool3D(pool_size=(2,2,2)))
		feature_extractor.add(tf.keras.layers.Flatten())
		
		for i in range(self.heads):
			flat[i]=feature_extractor(data_in[i])

		if(len(flat)>1):
			merge = tf.keras.layers.concatenate(flat)
		else:
			merge=flat[0]
		
		#hidden_1=tfp.layers.DenseFlipout(128,activation=tf.nn.relu)(merge)
		dropout_merge=tf.keras.layers.Dropout(0.2)(merge)
		hidden_1=tf.keras.layers.Dense(64,kernel_regularizer=tf.keras.regularizers.l2(l=self.regularizer_coeff),activation=tf.nn.relu)(dropout_merge)
		hidden_2=tf.keras.layers.Dense(64,kernel_regularizer=tf.keras.regularizers.l2(l=self.regularizer_coeff),activation=tf.nn.relu)(hidden_1)
		output=tf.keras.layers.Dense(self.output_dimension)(hidden_2)

		model=tf.keras.Model(inputs=data_in,outputs=output)
		
		overall_loss_dict={
			"regression_outputs":mse_basic,
			"classification_outputs":bin_crossentropy,
		}

		overall_loss_weights={
			"regression_outputs":1.0,
			"classification_outputs":1.0,
		}

		overall_metrics_dict={
		"regression_outputs":[tf.keras.metrics.MeanAbsoluteError()],
		"classification_outputs":[tf.keras.metrics.CategoricalAccuracy()],
		}

		model.compile(optimizer=tf.keras.optimizers.Adam(),experimental_run_tf_function=False,loss=tf.keras.losses.MeanSquaredError(),metrics=[tf.keras.metrics.MeanAbsoluteError()])
		print("3D CNN model successfully compiled")

		print(model.summary())
		
		from tensorflow.keras.utils import plot_model
		
		plot_model(model, to_file='../pre_trained_models/probablistic_bayes_models/pointdevnet_model.png')
		
		return model
	
	def multi_head_shared_bayes_cnn_model_3d(self,voxel_dim,deviation_channels):
		"""Build the 3D Model using the specified loss function, the inputs are parsed from the assemblyconfig_<case_study_name>.py file

			:param voxel_dim: The voxel dimension of the input, required to build input to the 3D CNN model
			:type voxel_dim: int (required)

			:param voxel_channels: The number of voxel channels in the input structure, required to build input to the 3D CNN model
			:type voxel_channels: int (required)
		"""
		negloglik = lambda y, rv_y: -rv_y.log_prob(y)
		
		aleatoric_std=0.0001
		
		aleatoric_tensor=[aleatoric_std] * self.output_dimension
		#constant aleatoric uncertainty

		import tensorflow as tf
		import tensorflow_probability as tfp
		tfd = tfp.distributions
		
		def _softplus_inverse(x):
  			"""Helper which computes the function inverse of `tf.nn.softplus`."""
  			return tf.math.log(tf.math.expm1(x))
		
		if(self.output_type=="regression"):
			final_layer_avt='linear'

		if(self.output_type=="classification"):
			final_layer_avt='softmax'

		data_in=[None] * self.heads
		flat=[None] * self.heads

		for i in range(self.heads):
			data_in[i]=tf.keras.layers.Input(shape=(voxel_dim,voxel_dim,voxel_dim,deviation_channels))

		feature_extractor = tf.keras.Sequential()
		feature_extractor.add(tfp.layers.Convolution3DFlipout(32, kernel_size=(5,5,5),strides=(2,2,2),activation='relu',input_shape=(voxel_dim,voxel_dim,voxel_dim,deviation_channels)))
		feature_extractor.add(tfp.layers.Convolution3DFlipout(32, kernel_size=(4,4,4),strides=(2,2,2),activation='relu'))
		feature_extractor.add(tfp.layers.Convolution3DFlipout(32, kernel_size=(3,3,3),strides=(1,1,1),activation='relu'))
		feature_extractor.add(tf.keras.layers.MaxPool3D(pool_size=(2,2,2)))
		feature_extractor.add(tf.keras.layers.Flatten())
		
		for i in range(self.heads):
			flat[i]=feature_extractor(data_in[i])

		merge = tf.keras.layers.concatenate(flat)

		#hidden_1=tfp.layers.DenseFlipout(128,activation=tf.nn.relu)(merge)
		hidden_2=tfp.layers.DenseFlipout(64,activation=tf.nn.relu)(merge)
		output=tfp.layers.DistributionLambda(lambda t: tfd.MultivariateNormalDiag(loc=t[..., :self.output_dimension], scale_diag=aleatoric_tensor))(hidden_2)

		model=tf.keras.Model(inputs=data_in,outputs=output)

		model.compile(optimizer=tf.keras.optimizers.Adam(),experimental_run_tf_function=False,loss=negloglik,metrics=[tf.keras.metrics.MeanAbsoluteError()])
		print("3D CNN model successfully compiled")

		print(model.summary())
		
		from tensorflow.keras.utils import plot_model
		plot_model(model, to_file='../pre_trained_models/probablistic_bayes_models/pointdevnet_model.png')
		
		return model

if (__name__=="__main__"):
	
	print("Prototyping model check...")
	output_dimension=6
	model_type='Probabilistic'
	heads=4
	voxel_dim=64
	deviation_channels=3

	multi_head_model= Multi_Head_DLModel(model_type,heads,output_dimension)
	model=multi_head_model.multi_head_shared_bayes_cnn_model_3d(voxel_dim,deviation_channels)