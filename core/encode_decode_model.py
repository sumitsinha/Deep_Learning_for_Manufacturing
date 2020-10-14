
class Encode_Decode_Model:    
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
	def __init__(self,output_dimension):
		
		self.output_dimension=output_dimension


	def encode_decode_3d(self,filter_root, depth,input_size=(64,64,64,3), n_class=3):
		"""Build the 3D Model using the specified loss function, the inputs are parsed from the assemblyconfig_<case_study_name>.py file

			:param voxel_dim: The voxel dimension of the input, required to build input to the 3D CNN model
			:type voxel_dim: int (required)

			:param voxel_channels: The number of voxel channels in the input structure, required to build input to the 3D CNN model
			:type voxel_channels: int (required)
		"""

		import tensorflow as tf
		import tensorflow_probability as tfp
		import numpy as np
		tfd = tfp.distributions
		
		import tensorflow.keras.backend as K 
		from tensorflow.keras.models import Model
		from tensorflow.keras.layers import Conv3D, MaxPooling3D, Add, BatchNormalization, Input, Activation, Lambda, Concatenate, Flatten, Dense,UpSampling3D,GlobalAveragePooling3D
		from tensorflow.keras.utils import plot_model
		
		"""
		Build UNet model with ResBlock.
		Args:
			filter_root (int): Number of filters to start with in first convolution.
			depth (int): How deep to go in UNet i.e. how many down and up sampling you want to do in the model. 
						Filter root and image size should be multiple of 2^depth.
			n_class (int, optional): How many classes in the output layer. Defaults to 2.
			input_size (tuple, optional): Input image size. Defaults to (256, 256, 1).
			activation (str, optional): activation to use in each convolution. Defaults to 'relu'.
			batch_norm (bool, optional): To use Batch normaliztion or not. Defaults to True.
			final_activation (str, optional): activation for output layer. Defaults to 'softmax'.
		Returns:
			obj: keras model object
		"""
		inputs = Input(input_size)
		x = inputs
		
		# Dictionary for long connections to Up Sampling Layers
		long_connection_store = {}

		Conv = Conv3D
		MaxPooling = MaxPooling3D
		UpSampling = UpSampling3D
		
		activation='relu'
		final_activation='linear'

		# Down sampling
		for i in range(depth):
			out_channel = 2**i * filter_root

			# Residual/Skip connection
			res = Conv(out_channel, kernel_size=1, padding='same', use_bias=False, name="Identity{}_1".format(i))(x)

			# First Conv Block with Conv, BN and activation
			conv1 = Conv(out_channel, kernel_size=3, padding='same', name="Conv{}_1".format(i))(x)
			#if batch_norm:
				#conv1 = BatchNormalization(name="BN{}_1".format(i))(conv1)
			act1 = Activation(activation, name="Act{}_1".format(i))(conv1)

			# Second Conv block with Conv and BN only
			conv2 = Conv(out_channel, kernel_size=3, padding='same', name="Conv{}_2".format(i))(act1)
			#if batch_norm:
				#conv2 = BatchNormalization(name="BN{}_2".format(i))(conv2)

			resconnection = Add(name="Add{}_1".format(i))([res, conv2])

			act2 = Activation(activation, name="Act{}_2".format(i))(resconnection)

			# Max pooling
			if i < depth - 1:
				long_connection_store[str(i)] = act2
				x = MaxPooling(padding='same', name="MaxPooling{}_1".format(i))(act2)
			else:
				x = act2

		feature_vector=Conv(self.output_dimension, 1, padding='same', activation=final_activation, name='Process_Parameter_output')(x)
		process_parameter=GlobalAveragePooling3D()(feature_vector)
		
		#feature_vector=Flatten()(x)
		#process_parameter=Dense(self.output_dimension)(feature_vector)
		
		# Upsampling
		for i in range(depth - 2, -1, -1):
			out_channel = 2**(i) * filter_root

			# long connection from down sampling path.
			long_connection = long_connection_store[str(i)]

			up1 = UpSampling(name="UpSampling{}_1".format(i))(x)
			up_conv1 = Conv(out_channel, 2, activation='relu', padding='same', name="upConvSam{}_1".format(i))(up1)

			#  Concatenate.
			up_conc = Concatenate(axis=-1, name="upConcatenate{}_1".format(i))([up_conv1, long_connection])

			#  Convolutions
			up_conv2 = Conv(out_channel, 3, padding='same', name="upConv{}_1".format(i))(up_conc)

			up_act1 = Activation(activation, name="upAct{}_1".format(i))(up_conv2)

			up_conv2 = Conv(out_channel, 3, padding='same', name="upConv{}_2".format(i))(up_act1)

			# Residual/Skip connection
			res = Conv(out_channel, kernel_size=1, padding='same', use_bias=False, name="upIdentity{}_1".format(i))(up_conc)

			resconnection = Add(name="upAdd{}_1".format(i))([res, up_conv2])

			x = Activation(activation, name="upAct{}_2".format(i))(resconnection)

		# Final convolution
		output = Conv(n_class, 1, padding='same', activation=final_activation, name='output')(x)

		model=Model(inputs, outputs=[process_parameter,output], name='Res-UNet')
		
		print("U-Net Based 3D Encoder Decoder Model Compiled")
		#print(model.summary())
		#plot_model(model,to_file='unet_3D.png',show_shapes=True, show_layer_names=True)
		
		
		mse_basic = tf.keras.losses.MeanSquaredError()
		msle = tf.keras.losses.MeanSquaredLogarithmicError()
		
		def mse_scaled(y_true,y_pred):
			return K.mean(K.square((y_pred - y_true)/10))

		model.compile(optimizer=tf.keras.optimizers.Adam(),experimental_run_tf_function=False,loss=[tf.keras.losses.MeanSquaredError(),mse_scaled],metrics=[tf.keras.losses.MeanSquaredError(),mse_scaled,msle,tf.keras.metrics.MeanAbsoluteError()])
		#print(model.summary())
		return model

	def encode_decode_3d_multi_output(self,filter_root, depth,input_size=(64,64,64,3),output_heads=2, n_class=3):
		"""Build the 3D Model using the specified loss function, the inputs are parsed from the assemblyconfig_<case_study_name>.py file

			:param voxel_dim: The voxel dimension of the input, required to build input to the 3D CNN model
			:type voxel_dim: int (required)

			:param voxel_channels: The number of voxel channels in the input structure, required to build input to the 3D CNN model
			:type voxel_channels: int (required)
		"""

		import tensorflow as tf
		import tensorflow_probability as tfp
		import numpy as np
		tfd = tfp.distributions
		
		import tensorflow.keras.backend as K 
		from tensorflow.keras.models import Model
		from tensorflow.keras.layers import Conv3D, MaxPooling3D, Add, BatchNormalization, Input, Activation, Lambda, Concatenate, Flatten, Dense,UpSampling3D,GlobalAveragePooling3D
		from tensorflow.keras.utils import plot_model
		
		"""
		Build UNet model with ResBlock.
		Args:
			filter_root (int): Number of filters to start with in first convolution.
			depth (int): How deep to go in UNet i.e. how many down and up sampling you want to do in the model. 
						Filter root and image size should be multiple of 2^depth.
			n_class (int, optional): How many classes in the output layer. Defaults to 2.
			input_size (tuple, optional): Input image size. Defaults to (256, 256, 1).
			activation (str, optional): activation to use in each convolution. Defaults to 'relu'.
			batch_norm (bool, optional): To use Batch normaliztion or not. Defaults to True.
			final_activation (str, optional): activation for output layer. Defaults to 'softmax'.
		Returns:
			obj: keras model object
		"""
		inputs = Input(input_size)
		x = inputs
		
		# Dictionary for long connections to Up Sampling Layers
		long_connection_store = {}

		Conv = Conv3D
		MaxPooling = MaxPooling3D
		UpSampling = UpSampling3D
		
		activation='relu'
		final_activation='linear'

		# Down sampling
		for i in range(depth):
			out_channel = 2**i * filter_root

			# Residual/Skip connection
			res = Conv(out_channel, kernel_size=1, padding='same', use_bias=False, name="Identity{}_1".format(i))(x)

			# First Conv Block with Conv, BN and activation
			conv1 = Conv(out_channel, kernel_size=3, padding='same', name="Conv{}_1".format(i))(x)
			#if batch_norm:
				#conv1 = BatchNormalization(name="BN{}_1".format(i))(conv1)
			act1 = Activation(activation, name="Act{}_1".format(i))(conv1)

			# Second Conv block with Conv and BN only
			conv2 = Conv(out_channel, kernel_size=3, padding='same', name="Conv{}_2".format(i))(act1)
			#if batch_norm:
				#conv2 = BatchNormalization(name="BN{}_2".format(i))(conv2)

			resconnection = Add(name="Add{}_1".format(i))([res, conv2])

			act2 = Activation(activation, name="Act{}_2".format(i))(resconnection)

			# Max pooling
			if i < depth - 1:
				long_connection_store[str(i)] = act2
				x = MaxPooling(padding='same', name="MaxPooling{}_1".format(i))(act2)
			else:
				x = act2

		feature_vector=Conv(self.output_dimension, 1, padding='same', activation=final_activation, name='Process_Parameter_output')(x)
		process_parameter=GlobalAveragePooling3D()(feature_vector)
		
		#feature_vector=Flatten()(x)
		#process_parameter=Dense(self.output_dimension)(feature_vector)
		
		# Upsampling
		for i in range(depth - 2, -1, -1):
			out_channel = 2**(i) * filter_root

			# long connection from down sampling path.
			long_connection = long_connection_store[str(i)]

			up1 = UpSampling(name="UpSampling{}_1".format(i))(x)
			up_conv1 = Conv(out_channel, 2, activation='relu', padding='same', name="upConvSam{}_1".format(i))(up1)

			#  Concatenate.
			up_conc = Concatenate(axis=-1, name="upConcatenate{}_1".format(i))([up_conv1, long_connection])

			#  Convolutions
			up_conv2 = Conv(out_channel, 3, padding='same', name="upConv{}_1".format(i))(up_conc)

			up_act1 = Activation(activation, name="upAct{}_1".format(i))(up_conv2)

			up_conv2 = Conv(out_channel, 3, padding='same', name="upConv{}_2".format(i))(up_act1)

			# Residual/Skip connection
			res = Conv(out_channel, kernel_size=1, padding='same', use_bias=False, name="upIdentity{}_1".format(i))(up_conc)

			resconnection = Add(name="upAdd{}_1".format(i))([res, up_conv2])

			x = Activation(activation, name="upAct{}_2".format(i))(resconnection)

		# Final convolution
		mse_basic = tf.keras.losses.MeanSquaredError()
		msle = tf.keras.losses.MeanSquaredLogarithmicError()
		
		def mse_scaled(y_true,y_pred):
			return K.mean(K.square((y_pred - y_true)/10))

		output_list=[]
		output_list.append(process_parameter)
		loss_list=[]
		loss_list.append(mse_basic)

		for i in range(output_heads):
			out_layer_name="output_"+str(i)
			pen_out_layer_name="pen_output_"+str(i)
			pen_out_layer_name2="pen_output2_"+str(i)
			pen_output=Conv(16, 1, padding='same', activation='relu', name=pen_out_layer_name)(x)
			#pen_output_2=Conv(16, 1, padding='same', activation='relu', name=pen_out_layer_name2)(pen_output)
			output = Conv(n_class, 1, padding='same', activation=final_activation, name=out_layer_name)(pen_output)
			output_list.append(output)
			loss_list.append(mse_scaled)

		model=Model(inputs, outputs=output_list, name='Res-UNet')
		
		print("U-Net Based 3D Encoder Decoder Model Compiled")
		#print(model.summary())
		#plot_model(model,to_file='unet_3D_multi_output.png',show_shapes=True, show_layer_names=True)
	
		model.compile(optimizer=tf.keras.optimizers.Adam(),experimental_run_tf_function=False,loss=loss_list,metrics=[tf.keras.losses.MeanSquaredError(),mse_scaled,msle,tf.keras.metrics.MeanAbsoluteError()])
		#print(model.summary())
		return model


	def encode_decode_3d_multi_output_attention(self,filter_root, depth,input_size=(64,64,64,3),output_heads=2, n_class=3):
		"""Build the 3D Model using the specified loss function, the inputs are parsed from the assemblyconfig_<case_study_name>.py file

			:param voxel_dim: The voxel dimension of the input, required to build input to the 3D CNN model
			:type voxel_dim: int (required)

			:param voxel_channels: The number of voxel channels in the input structure, required to build input to the 3D CNN model
			:type voxel_channels: int (required)
		"""

		import tensorflow as tf
		import tensorflow_probability as tfp
		import numpy as np
		tfd = tfp.distributions
		
		import tensorflow.keras.backend as K 
		from tensorflow.keras.models import Model
		from tensorflow.keras.layers import Conv3D, MaxPooling3D, Add, BatchNormalization, Input, Activation, Lambda, Concatenate, Flatten, Dense,UpSampling3D,GlobalAveragePooling3D
		from tensorflow.keras.utils import plot_model
		
		from tensorflow.keras.layers import add, multiply
		"""
		Build UNet model with ResBlock.
		Args:
			filter_root (int): Number of filters to start with in first convolution.
			depth (int): How deep to go in UNet i.e. how many down and up sampling you want to do in the model. 
						Filter root and image size should be multiple of 2^depth.
			n_class (int, optional): How many classes in the output layer. Defaults to 2.
			input_size (tuple, optional): Input image size. Defaults to (256, 256, 1).
			activation (str, optional): activation to use in each convolution. Defaults to 'relu'.
			batch_norm (bool, optional): To use Batch normaliztion or not. Defaults to True.
			final_activation (str, optional): activation for output layer. Defaults to 'softmax'.
		Returns:
			obj: keras model object
		"""
		inputs = Input(input_size)
		x = inputs
		
		# Dictionary for long connections to Up Sampling Layers
		long_connection_store = {}

		Conv = Conv3D
		MaxPooling = MaxPooling3D
		UpSampling = UpSampling3D
		
		activation='relu'
		final_activation='linear'

		def attention_block(x, g, inter_channel):

		    theta_x = Conv(inter_channel, [1,1,1], strides=[1,1,1])(x)
		    phi_g = Conv(inter_channel, [1,1,1], strides=[1,1,1])(g)
		    
		    f = Activation('relu')(add([theta_x, phi_g]))
		    psi_f = Conv(1, [1,1,1], strides=[1,1,1])(f)

		    rate = Activation('sigmoid')(psi_f)

		    att_x = multiply([x, rate])
		    return att_x

		# Down sampling
		for i in range(depth):
			out_channel = 2**i * filter_root

			# Residual/Skip connection
			res = Conv(out_channel, kernel_size=1, padding='same', use_bias=False, name="Identity{}_1".format(i))(x)

			# First Conv Block with Conv, BN and activation
			conv1 = Conv(out_channel, kernel_size=3, padding='same', name="Conv{}_1".format(i))(x)
			#if batch_norm:
				#conv1 = BatchNormalization(name="BN{}_1".format(i))(conv1)
			act1 = Activation(activation, name="Act{}_1".format(i))(conv1)

			# Second Conv block with Conv and BN only
			conv2 = Conv(out_channel, kernel_size=3, padding='same', name="Conv{}_2".format(i))(act1)
			#if batch_norm:
				#conv2 = BatchNormalization(name="BN{}_2".format(i))(conv2)

			resconnection = Add(name="Add{}_1".format(i))([res, conv2])

			act2 = Activation(activation, name="Act{}_2".format(i))(resconnection)

			# Max pooling
			if i < depth - 1:
				long_connection_store[str(i)] = act2
				x = MaxPooling(padding='same', name="MaxPooling{}_1".format(i))(act2)
			else:
				x = act2

		feature_vector=Conv(self.output_dimension, 1, padding='same', activation=final_activation, name='Process_Parameter_output')(x)
		process_parameter=GlobalAveragePooling3D()(feature_vector)
		
		#feature_vector=Flatten()(x)
		#process_parameter=Dense(self.output_dimension)(feature_vector)
		
		# Upsampling
		for i in range(depth - 2, -1, -1):
			out_channel = 2**(i) * filter_root

			# long connection from down sampling path.
			long_connection = long_connection_store[str(i)]

			up1 = UpSampling(name="UpSampling{}_1".format(i))(x)
			up_conv1 = Conv(out_channel, 2, activation='relu', padding='same', name="upConvSam{}_1".format(i))(up1)

			attention_layer = attention_block(x=long_connection, g=up_conv1, inter_channel=out_channel // 4)
			#  Concatenate.
			
			#up_conc = Concatenate(axis=-1, name="upConcatenate{}_1".format(i))([up_conv1, long_connection])
			up_conc = Concatenate(axis=-1, name="upConcatenate{}_1".format(i))([up_conv1, attention_layer])

			#  Convolutions
			up_conv2 = Conv(out_channel, 3, padding='same', name="upConv{}_1".format(i))(up_conc)

			up_act1 = Activation(activation, name="upAct{}_1".format(i))(up_conv2)

			up_conv2 = Conv(out_channel, 3, padding='same', name="upConv{}_2".format(i))(up_act1)

			# Residual/Skip connection
			res = Conv(out_channel, kernel_size=1, padding='same', use_bias=False, name="upIdentity{}_1".format(i))(up_conc)

			resconnection = Add(name="upAdd{}_1".format(i))([res, up_conv2])

			x = Activation(activation, name="upAct{}_2".format(i))(resconnection)

		# Final convolution
		mse_basic = tf.keras.losses.MeanSquaredError()
		msle = tf.keras.losses.MeanSquaredLogarithmicError()
		
		def mse_scaled(y_true,y_pred):
			return K.mean(K.square((y_pred - y_true)*100))

		output_list=[]
		output_list.append(process_parameter)
		loss_list=[]
		loss_list.append(mse_basic)

		for i in range(output_heads):
			out_layer_name="output_"+str(i)
			pen_out_layer_name="pen_output_"+str(i)
			pen_out_layer_name2="pen_output2_"+str(i)
			pen_output=Conv(16, 1, padding='same', activation='relu', name=pen_out_layer_name)(x)
			#pen_output_2=Conv(16, 1, padding='same', activation='relu', name=pen_out_layer_name2)(pen_output)
			output = Conv(n_class, 1, padding='same', activation=final_activation, name=out_layer_name)(pen_output)
			output_list.append(output)
			loss_list.append(mse_scaled)

		model=Model(inputs, outputs=output_list, name='Res-UNet')
		
		print("U-Net Based 3D Encoder Decoder Model Compiled")
		#print(model.summary())
		#plot_model(model,to_file='unet_3D_multi_output_attention.png',show_shapes=True, show_layer_names=True)
	
		model.compile(optimizer=tf.keras.optimizers.Adam(),experimental_run_tf_function=False,loss=loss_list,metrics=[tf.keras.losses.MeanSquaredError(),mse_scaled,tf.keras.metrics.MeanAbsoluteError()])
		#print(model.summary())
		return model

	def encode_decode_3d_multi_output_attention_hybrid(self,filter_root, depth,input_size=(64,64,64,3),categorical_outputs=25,output_heads=2, n_class=3):
		"""Build the 3D Model using the specified loss function, the inputs are parsed from the assemblyconfig_<case_study_name>.py file

			:param voxel_dim: The voxel dimension of the input, required to build input to the 3D CNN model
			:type voxel_dim: int (required)

			:param voxel_channels: The number of voxel channels in the input structure, required to build input to the 3D CNN model
			:type voxel_channels: int (required)
		"""

		import tensorflow as tf
		import tensorflow_probability as tfp
		import numpy as np
		tfd = tfp.distributions
		
		import tensorflow.keras.backend as K 
		from tensorflow.keras.models import Model
		from tensorflow.keras.layers import Conv3D, MaxPooling3D, Add, BatchNormalization, Input, Activation, Lambda, Concatenate, Flatten, Dense,UpSampling3D,GlobalAveragePooling3D
		from tensorflow.keras.utils import plot_model
		
		from tensorflow.keras.layers import add, multiply
		"""
		Build UNet model with ResBlock.
		Args:
			filter_root (int): Number of filters to start with in first convolution.
			depth (int): How deep to go in UNet i.e. how many down and up sampling you want to do in the model. 
						Filter root and image size should be multiple of 2^depth.
			n_class (int, optional): How many classes in the output layer. Defaults to 2.
			input_size (tuple, optional): Input image size. Defaults to (256, 256, 1).
			activation (str, optional): activation to use in each convolution. Defaults to 'relu'.
			batch_norm (bool, optional): To use Batch normaliztion or not. Defaults to True.
			final_activation (str, optional): activation for output layer. Defaults to 'softmax'.
		Returns:
			obj: keras model object
		"""
		bin_crossentropy=tf.keras.losses.BinaryCrossentropy()
		mse_basic = tf.keras.losses.MeanSquaredError()

		#Can also Try
		#scale_factor=4000/64 #number of samples/batchsize
		#kl = sum(model.losses) / scale_factor
		#Annealing of KL divergence
		#bin_crossentropy=tf.keras.losses.BinaryCrossentropy()+kl* K.get_value(kl_alpha)

		overall_loss_dict={
		"regression_outputs":mse_basic,
		"classification_outputs":bin_crossentropy,
		"shape_error_outputs":mse_basic
		}

		overall_loss_weights={
		"regression_outputs":1.0,
		"classification_outputs":1.0,
		"shape_error_outputs":1.0
		}

		overall_metrics_dict={
		"regression_outputs":[tf.keras.metrics.MeanAbsoluteError()],
		"classification_outputs":[tf.keras.metrics.CategoricalAccuracy()],
		"shape_error_outputs":[tf.keras.metrics.MeanAbsoluteError()]
		}

		inputs = Input(input_size)
		x = inputs
		
		# Dictionary for long connections to Up Sampling Layers
		long_connection_store = {}

		Conv = Conv3D
		MaxPooling = MaxPooling3D
		UpSampling = UpSampling3D
		
		activation='relu'
		final_activation='linear'

		def attention_block(x, g, inter_channel):

		    theta_x = Conv(inter_channel, [1,1,1], strides=[1,1,1])(x)
		    phi_g = Conv(inter_channel, [1,1,1], strides=[1,1,1])(g)
		    
		    f = Activation('relu')(add([theta_x, phi_g]))
		    psi_f = Conv(1, [1,1,1], strides=[1,1,1])(f)

		    rate = Activation('sigmoid')(psi_f)

		    att_x = multiply([x, rate])
		    return att_x

		# Down sampling
		for i in range(depth):
			out_channel = 2**i * filter_root

			# Residual/Skip connection
			res = Conv(out_channel, kernel_size=1, padding='same', use_bias=False, name="Identity{}_1".format(i))(x)

			# First Conv Block with Conv, BN and activation
			conv1 = Conv(out_channel, kernel_size=3, padding='same', name="Conv{}_1".format(i))(x)
			#if batch_norm:
				#conv1 = BatchNormalization(name="BN{}_1".format(i))(conv1)
			act1 = Activation(activation, name="Act{}_1".format(i))(conv1)

			# Second Conv block with Conv and BN only
			conv2 = Conv(out_channel, kernel_size=3, padding='same', name="Conv{}_2".format(i))(act1)
			#if batch_norm:
				#conv2 = BatchNormalization(name="BN{}_2".format(i))(conv2)

			resconnection = Add(name="Add{}_1".format(i))([res, conv2])

			act2 = Activation(activation, name="Act{}_2".format(i))(resconnection)

			# Max pooling
			if i < depth - 1:
				long_connection_store[str(i)] = act2
				x = MaxPooling(padding='same', name="MaxPooling{}_1".format(i))(act2)
			else:
				x = act2

		#Regression Outputs
		feature_vector=Conv(self.output_dimension-categorical_outputs, 1, padding='same', activation=final_activation, name='Process_Parameter_output_regression')(x)
		process_parameter_regression=GlobalAveragePooling3D(name='regression_outputs')(feature_vector)
		
		#Classification Outputs
		feature_vector_categorical=Conv(categorical_outputs, 1, padding='same', activation=final_activation, name='Process_Parameter_output_classification')(x)
		process_parameter_cont=GlobalAveragePooling3D()(feature_vector_categorical)
		process_parameter_classification=Activation('sigmoid',name='classification_outputs')(process_parameter_cont)
		
		#feature_vector=Flatten()(x)
		#process_parameter=Dense(self.output_dimension)(feature_vector)
		
		# Upsampling
		for i in range(depth - 2, -1, -1):
			out_channel = 2**(i) * filter_root

			# long connection from down sampling path.
			long_connection = long_connection_store[str(i)]

			up1 = UpSampling(name="UpSampling{}_1".format(i))(x)
			up_conv1 = Conv(out_channel, 2, activation='relu', padding='same', name="upConvSam{}_1".format(i))(up1)

			attention_layer = attention_block(x=long_connection, g=up_conv1, inter_channel=out_channel // 4)
			#  Concatenate.
			
			#up_conc = Concatenate(axis=-1, name="upConcatenate{}_1".format(i))([up_conv1, long_connection])
			up_conc = Concatenate(axis=-1, name="upConcatenate{}_1".format(i))([up_conv1, attention_layer])

			#  Convolutions
			up_conv2 = Conv(out_channel, 3, padding='same', name="upConv{}_1".format(i))(up_conc)

			up_act1 = Activation(activation, name="upAct{}_1".format(i))(up_conv2)

			up_conv2 = Conv(out_channel, 3, padding='same', name="upConv{}_2".format(i))(up_act1)

			# Residual/Skip connection
			res = Conv(out_channel, kernel_size=1, padding='same', use_bias=False, name="upIdentity{}_1".format(i))(up_conc)

			resconnection = Add(name="upAdd{}_1".format(i))([res, up_conv2])

			x = Activation(activation, name="upAct{}_2".format(i))(resconnection)


		output_list=[]
		output_list.append(process_parameter_regression)
		output_list.append(process_parameter_classification)

		output = Conv(n_class*output_heads, 1, padding='same', activation=final_activation, name='shape_error_outputs')(x)
		output_list.append(output)

		# for i in range(output_heads):
		# 	out_layer_name="output_"+str(i)
		# 	pen_out_layer_name="pen_output_"+str(i)
		# 	pen_out_layer_name2="pen_output2_"+str(i)
		# 	pen_output=Conv(16, 1, padding='same', activation='relu', name=pen_out_layer_name)(x)
		# 	#pen_output_2=Conv(16, 1, padding='same', activation='relu', name=pen_out_layer_name2)(pen_output)
		# 	output = Conv(n_class, 1, padding='same', activation=final_activation, name=out_layer_name)(pen_output)
		# 	output_list.append(output)
		# 	loss_list.append(mse_scaled)

		model=Model(inputs, outputs=output_list, name='Res-UNet_Attention_Hybrid')
		
		print("U-Net Based 3D Encoder Decoder Model Compiled")
		#print(model.summary())
		#plot_model(model,to_file='unet_3D_multi_output_attention_hybrid.png',show_shapes=True, show_layer_names=True)
	
		model.compile(optimizer=tf.keras.optimizers.Adam(),experimental_run_tf_function=False,loss=overall_loss_dict,metrics=overall_metrics_dict,loss_weights=overall_loss_weights)
		#print(model.summary())
		return model

	def resnet_3d_cnn_hybrid(self,voxel_dim=64,deviation_channels=3,categoric_outputs=25,w_val=0):

			import numpy as np
			import tensorflow as tf
			import tensorflow.keras.backend as K 
			from tensorflow.keras.models import Model
			from tensorflow.keras import regularizers
			from tensorflow.keras.layers import Conv3D, MaxPooling3D, Add, BatchNormalization, Input, LeakyReLU,Activation, Lambda, Concatenate, Flatten, Dense,UpSampling3D,GlobalAveragePooling3D
			from tensorflow.keras.utils import plot_model

			if(w_val==0):
				w_val=np.zeros(self.output_dimension)
				w_val[:]=1/self.output_dimension


			def weighted_mse(val):
				def loss(yTrue,yPred):

					#val = np.array([0.1,0.1,0.1,0.1,0.1]) 
					w_var = K.variable(value=val, dtype='float32', 
												 name='weight_vec')
					#weight_vec = K.ones_like(yTrue[0,:]) #a simple vector with ones shaped as (60,)
					#idx = K.cumsum(ones) #similar to a 'range(1,61)'

					return K.mean((w_var)*K.square(yTrue-yPred))
				return loss

			input_size=(voxel_dim,voxel_dim,voxel_dim,deviation_channels)
			inputs = Input(input_size)
			x = inputs
			y = Conv3D(32, kernel_size=(4,4,4),strides=(2,2,2), name="conv_block_1")(x)
			res1=y
			
			y = LeakyReLU()(y)
			y = Conv3D(32, kernel_size=(3,3,3),strides=(1,1,1), padding='same',name="conv_block_2")(y)
			y = LeakyReLU()(y)
			y = Conv3D(32, kernel_size=(3,3,3),strides=(1,1,1), padding='same',name="conv_block_3")(y)
			y = Add()([res1, y])
			y = LeakyReLU()(y)
			
			y = Conv3D(32, kernel_size=(3,3,3),strides=(2,2,2), name="conv_block_4")(y)
			res2=y
			y = LeakyReLU()(y)
			
			y = Conv3D(32, kernel_size=(3,3,3),strides=(1,1,1), padding='same',name="conv_block_5")(y)
			y = LeakyReLU()(y)
			
			y = Conv3D(32, kernel_size=(3,3,3),strides=(1,1,1), padding='same',name="conv_block_6")(y)
			y = Add()([res2, y])
			y = LeakyReLU()(y)
			
			y = Conv3D(32, kernel_size=(3,3,3),strides=(2,2,2), name="conv_block_7")(y)
			res3=y
			y = LeakyReLU()(y)
			
			y = Conv3D(32, kernel_size=(3,3,3),strides=(1,1,1),padding='same', name="conv_block_8")(y)
			y = LeakyReLU()(y)
			
			y = Conv3D(32, kernel_size=(3,3,3),strides=(1,1,1),padding='same', name="conv_block_9")(y)
			
			y = Add()([res3, y])
			y = LeakyReLU()(y)
			
			y=Flatten()(y)
			
			y=Dense(128,kernel_regularizer=regularizers.l2(0.01),activation='relu')(y)
			y=Dense(64,kernel_regularizer=regularizers.l2(0.01),activation='relu')(y)
			
			output_regression=Dense(self.output_dimension-categoric_outputs,name="regression_output")(y)
			output_classification=Dense(categoric_outputs,activation="sigmoid",name="classification_output")(y)		
			
			output=[output_regression,output_classification]

			model=Model(inputs, outputs=output, name='Res_3D_CNN_hybrid')
			
			def mse_scaled(y_true,y_pred):
				return K.mean(K.square((y_pred - y_true)))
			
			#loss_regression=tf.keras.losses.MeanSquaredError()
			loss_regression=mse_scaled
			loss_classification=tf.keras.losses.BinaryCrossentropy()
			
			loss_list=[loss_regression,loss_classification]

			model.compile(loss=loss_list, optimizer=tf.keras.optimizers.Adam(),experimental_run_tf_function=False, metrics=[tf.keras.metrics.MeanAbsoluteError(),tf.keras.metrics.Accuracy()])
			
			#plot_model(model,to_file='resnet_3d_cnn_hybrid.png',show_shapes=True, show_layer_names=True)
			#print(model.summary())
			
			return model

if (__name__=="__main__"):
	
	print('Model Summary')
	enc_dec=Encode_Decode_Model(148)

	#model=enc_dec.encode_decode_3d(16,4)
	model=enc_dec.encode_decode_3d_multi_output_attention_hybrid(16,4)