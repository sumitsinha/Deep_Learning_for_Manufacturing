""" Contains core classes and methods for initializing Probabilistic deep learning 3D CNN model with different variants of the loss function, inputs are provided from the modelconfig_train.py file"""

class Bayes_DLModel:    
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
	def __init__(self,model_type,output_dimension,optimizer,loss_function,regularizer_coeff,output_type='regression'):
		self.output_dimension=output_dimension
		self.model_type=model_type
		self.optimizer=optimizer
		self.loss_function=loss_function
		self.regularizer_coeff=regularizer_coeff
		self.output_type=output_type

	def bayes_cnn_model_3d(self,voxel_dim,deviation_channels):
		"""Build the 3D Model using the specified loss function, the inputs are parsed from the assemblyconfig_<case_study_name>.py file

			:param voxel_dim: The voxel dimension of the input, required to build input to the 3D CNN model
			:type voxel_dim: int (required)

			:param voxel_channels: The number of voxel channels in the input structure, required to build input to the 3D CNN model
			:type voxel_channels: int (required)
		"""
		
		negloglik = lambda y, rv_y: -rv_y.log_prob(y)
		
		aleatoric_std=0.001
		aleatoric_tensor=[aleatoric_std] * self.output_dimension
		#constant aleatoric uncertainty

		import tensorflow as tf
		import tensorflow_probability as tfp
		import numpy as np
		tfd = tfp.distributions
		
		def _softplus_inverse(x):
  			"""Helper which computes the function inverse of `tf.nn.softplus`."""
  			return tf.math.log(tf.math.expm1(x))
		kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) / tf.cast(4000, dtype=tf.float32))
		if(self.output_type=="regression"):
			final_layer_avt='linear'

		if(self.output_type=="classification"):
			final_layer_avt='softmax'
		
		c = np.log(np.expm1(1.))

		model = tf.keras.Sequential([
			tf.keras.layers.InputLayer(input_shape=(voxel_dim,voxel_dim,voxel_dim,deviation_channels)),
			tfp.layers.Convolution3DFlipout(32, kernel_size=(5,5,5),kernel_divergence_fn=kl_divergence_function,strides=(2,2,2),activation=tf.nn.relu),
			tfp.layers.Convolution3DFlipout(32, kernel_size=(4,4,4),kernel_divergence_fn=kl_divergence_function,strides=(2,2,2),activation=tf.nn.relu),
			tfp.layers.Convolution3DFlipout(32, kernel_size=(3,3,3),kernel_divergence_fn=kl_divergence_function,strides=(1,1,1),activation=tf.nn.relu),
			tf.keras.layers.MaxPooling3D(pool_size=[2, 2, 2]),
			tf.keras.layers.Flatten(),
			tfp.layers.DenseFlipout(128,activation=tf.nn.relu,kernel_divergence_fn=kl_divergence_function),
			tfp.layers.DenseFlipout(64,kernel_divergence_fn=kl_divergence_function,activation=tf.nn.relu),
			tfp.layers.DenseFlipout(self.output_dimension,kernel_divergence_fn=kl_divergence_function),
			tfp.layers.DistributionLambda(lambda t:tfd.MultivariateNormalDiag(loc=t[..., :self.output_dimension], scale_diag=aleatoric_tensor)),
			])

		#negloglik = lambda y, p_y: -p_y.log_prob(y)
		#experimental_run_tf_function=False
		#tf.keras.optimizers.Adam(lr=0.001)
		model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),experimental_run_tf_function=False,loss=negloglik,metrics=[tf.keras.metrics.MeanAbsoluteError()])
		print("3D CNN model successfully compiled")
		print(model.summary())
		return model

	def bayes_cnn_model_3d_hybrid(self,categorical_kccs,voxel_dim=64,deviation_channels=3):
		"""Build the 3D Model using the specified loss function, the inputs are parsed from the assemblyconfig_<case_study_name>.py file

			:param voxel_dim: The voxel dimension of the input, required to build input to the 3D CNN model
			:type voxel_dim: int (required)

			:param voxel_channels: The number of voxel channels in the input structure, required to build input to the 3D CNN model
			:type voxel_channels: int (required)
		"""
		
		import tensorflow as tf
		import tensorflow_probability as tfp
		import numpy as np
		from tensorflow.keras.models import Model

		from tensorflow.keras.layers import Input
		from tensorflow.keras.utils import plot_model

		#Testing 
		output_dimension=self.output_dimension-categorical_kccs

		#Losses
		negloglik = lambda y, rv_y: -rv_y.log_prob(y)
		bin_crossentropy=tf.keras.losses.BinaryCrossentropy()

		#constant aleatoric uncertainty
		aleatoric_std=0.001
		aleatoric_tensor=[aleatoric_std] * output_dimension
		
		tfd = tfp.distributions
		kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) / (tf.cast(4000, dtype=tf.float32)/tf.cast(64, dtype=tf.float32)))
		
		
		c = np.log(np.expm1(1.))

		input_size=(voxel_dim,voxel_dim,voxel_dim,deviation_channels)
		inputs = Input(input_size)

		x=tfp.layers.Convolution3DFlipout(32, kernel_size=(5,5,5),kernel_divergence_fn=kl_divergence_function,strides=(2,2,2),activation=tf.nn.relu)(inputs)
		x=tfp.layers.Convolution3DFlipout(32, kernel_size=(4,4,4),kernel_divergence_fn=kl_divergence_function,strides=(2,2,2),activation=tf.nn.relu)(x)
		x=tfp.layers.Convolution3DFlipout(32, kernel_size=(3,3,3),kernel_divergence_fn=kl_divergence_function,strides=(1,1,1),activation=tf.nn.relu)(x)
		x=tf.keras.layers.MaxPooling3D(pool_size=[2, 2, 2])(x)
		x=tf.keras.layers.Flatten()(x)
		x=tfp.layers.DenseFlipout(128,activation=tf.nn.relu,kernel_divergence_fn=kl_divergence_function)(x)
		x=tfp.layers.DenseFlipout(64,kernel_divergence_fn=kl_divergence_function,activation=tf.nn.relu)(x)
		
		reg_output=tfp.layers.DenseFlipout(output_dimension,kernel_divergence_fn=kl_divergence_function)(x)
		reg_distrbution=tfp.layers.DistributionLambda(lambda t:tfd.MultivariateNormalDiag(loc=t[..., :output_dimension], scale_diag=aleatoric_tensor),name="regression_outputs")(reg_output)
		
		cla_distrbution=tfp.layers.DenseFlipout(categorical_kccs, kernel_divergence_fn=kl_divergence_function,activation=tf.nn.sigmoid,name="classification_outputs")(x)

		output_list=[]
		output_list.append(reg_distrbution)
		output_list.append(cla_distrbution)
		
		model=Model(inputs, outputs=output_list, name='Hybrid_Bayesian_Model')
		
		model_losses=[]
		model_losses.append(negloglik)
		model_losses.append(bin_crossentropy)

		model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),experimental_run_tf_function=False,loss=model_losses,metrics=[tf.keras.metrics.MeanAbsoluteError(),tf.keras.metrics.Accuracy()])
		print("3D CNN model successfully compiled")
		print(model.summary())
		
		#print(model.summary())
		#plot_model(model,to_file='Bayes_Hybrid.png',show_shapes=True, show_layer_names=True)
		return model

	def bayes_unet_model_3d_hybrid(self,filter_root, depth,categorical_kccs,voxel_dim=64,deviation_channels=3,output_heads=2):
		"""Build the 3D Model using the specified loss function, the inputs are parsed from the assemblyconfig_<case_study_name>.py file

			:param voxel_dim: The voxel dimension of the input, required to build input to the 3D CNN model
			:type voxel_dim: int (required)

			:param voxel_channels: The number of voxel channels in the input structure, required to build input to the 3D CNN model
			:type voxel_channels: int (required)
		"""
		
		import tensorflow as tf
		import tensorflow_probability as tfp
		tfd = tfp.distributions
		import numpy as np
		from tensorflow.keras.models import Model
		import tensorflow.keras.backend as K 
		from tensorflow.keras.models import Model
		from tensorflow.keras.layers import Conv3D, MaxPooling3D, Add, BatchNormalization, Input, Activation, Lambda, Concatenate, Flatten, Dense,UpSampling3D,GlobalAveragePooling3D
		from tensorflow.keras.utils import plot_model
		
		from tensorflow.keras.layers import add, multiply
		from tensorflow.keras.layers import Input
		from tensorflow.keras.utils import plot_model

		#Testing 
		reg_kccs=self.output_dimension-categorical_kccs
		
		#Testing
		#reg_kccs=15-categorical_kccs

		# Probabalistic Losses
		negloglik = lambda y, rv_y: -rv_y.log_prob(y)
		bin_crossentropy=tf.keras.losses.BinaryCrossentropy()
		mse_basic = tf.keras.losses.MeanSquaredError()

		#Can also Try
		#scale_factor=4000/64 #number of samples/batchsize
		#kl = sum(model.losses) / scale_factor
		#Annealing of KL divergence
		#bin_crossentropy=tf.keras.losses.BinaryCrossentropy()+kl* K.get_value(kl_alpha)

		overall_loss_dict={
		"regression_outputs":negloglik,
		"classification_outputs":bin_crossentropy,
		"shape_error_outputs":mse_basic
		}

		overall_loss_weights={
		"regression_outputs":2.0,
		"classification_outputs":2.0,
		"shape_error_outputs":1.0
		}

		overall_metrics_dict={
		"regression_outputs":[tf.keras.metrics.MeanAbsoluteError()],
		"classification_outputs":[tf.keras.metrics.CategoricalAccuracy()],
		"shape_error_outputs":[tf.keras.metrics.MeanAbsoluteError()]
		}

		#constant aleatoric uncertainty
		aleatoric_std=0.001
		aleatoric_std_cop=0.001
		aleatoric_tensor=[aleatoric_std] * reg_kccs
		
		tfd = tfp.distributions
		kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) / (tf.cast(4000, dtype=tf.float32)/tf.cast(64, dtype=tf.float32)))
		
		
		c = np.log(np.expm1(1.))

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

		input_size=(voxel_dim,voxel_dim,voxel_dim,deviation_channels)
		inputs = Input(input_size)
		x = inputs

		# Down sampling
		for i in range(depth):
			out_channel = 2**i * filter_root

			# Residual/Skip connection
			res = tfp.layers.Convolution3DFlipout(out_channel, kernel_size=1, kernel_divergence_fn=kl_divergence_function,padding='same', name="Identity{}_1".format(i))(x)

			# First Conv Block with Conv, BN and activation
			conv1 = tfp.layers.Convolution3DFlipout(out_channel, kernel_size=3, kernel_divergence_fn=kl_divergence_function,padding='same', name="Conv{}_1".format(i))(x)
			#if batch_norm:
				#conv1 = BatchNormalization(name="BN{}_1".format(i))(conv1)
			act1 = Activation(activation, name="Act{}_1".format(i))(conv1)

			# Second Conv block with Conv and BN only
			conv2 = tfp.layers.Convolution3DFlipout(out_channel, kernel_size=3, padding='same', kernel_divergence_fn=kl_divergence_function,name="Conv{}_2".format(i))(act1)
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

		feature_vector_reg=Conv(reg_kccs, 1, padding='same', activation=final_activation, name='Process_Parameter_Reg_output')(x)
		process_parameter_reg=GlobalAveragePooling3D()(feature_vector_reg)
		
		feature_vector_cla=Conv(categorical_kccs, 1, padding='same', activation=final_activation, name='Process_Parameter_Cla_output')(x)
		process_parameter_cla=GlobalAveragePooling3D()(feature_vector_cla)
		
		#feature_categorical=Flatten()(feature_vector)
		#reg_output=tfp.layers.DenseFlipout(output_dimension,kernel_divergence_fn=kl_divergence_function)(process_parameter)
		
		#Process Parameter Outputs
		reg_distrbution=tfp.layers.DistributionLambda(lambda t:tfd.MultivariateNormalDiag(loc=t[..., :reg_kccs], scale_diag=aleatoric_tensor),name="regression_outputs")(process_parameter_reg)
		cla_distrbution=tfp.layers.DenseFlipout(categorical_kccs, kernel_divergence_fn=kl_divergence_function,activation=tf.nn.sigmoid,name="classification_outputs")(process_parameter_cla)

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
		output_list.append(reg_distrbution)
		output_list.append(cla_distrbution)
		
		output = Conv(deviation_channels*output_heads, 1, padding='same', activation=final_activation, name='shape_error_outputs')(x)
		output_list.append(output)
		
		model=Model(inputs, outputs=output_list, name='Hybrid_Bayesian_Model')
		
		#Loss Dictonary Created
		#model_losses=[]
		#model_losses.append(negloglik)
		#model_losses.append(bin_crossentropy)

		model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),experimental_run_tf_function=False,loss=overall_loss_dict,metrics=overall_metrics_dict,loss_weights=overall_loss_weights)
		print("3D CNN model successfully compiled")
		print(model.summary())
		
		#plot_model(model,to_file='Bayes_OSER2.png',show_shapes=True, show_layer_names=True)
		
		return model

if (__name__=="__main__"):
	print('Bayesian Deep Learning Class')
	#bayes_unet_model_3d_hybrid(10,32,4,voxel_dim=64,deviation_channels=3,output_heads=2)