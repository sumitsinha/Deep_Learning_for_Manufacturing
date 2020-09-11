class BaseModelArch:    
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

	
	def base_model_func(self,hp,filter_root,depth,categorical_kccs,voxel_dim=64,deviation_channels=3,output_heads=2):
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
		
		bin_crossentropy=tf.keras.losses.BinaryCrossentropy()
		mse_basic = tf.keras.losses.MeanSquaredError()

		overall_loss_dict={
		"regression_outputs":mse_basic,
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

		#Units for hyper_paramter Tuning
		hp_units = hp.Int('filter_root', min_value = 8, max_value = 32, step = 8)
		filter_root=hp_units
		
		depth = hp.Choice('Depth', values = [2,3])

		hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4]) 
		# Down sampling
		for i in range(depth):
			out_channel = 2**i * filter_root

			# Residual/Skip connection
			res = Conv(out_channel, kernel_size=1, padding='same', name="Identity{}_1".format(i))(x)

			# First Conv Block with Conv, BN and activation
			conv1 = Conv(out_channel, kernel_size=3,padding='same', name="Conv{}_1".format(i))(x)
			#if batch_norm:
				#conv1 = BatchNormalization(name="BN{}_1".format(i))(conv1)
			act1 = Activation(activation, name="Act{}_1".format(i))(conv1)

			# Second Conv block with Conv and BN only
			conv2 = Conv(out_channel, kernel_size=3, padding='same',name="Conv{}_2".format(i))(act1)
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
		

		#Process Parameter Outputs
		reg_distrbution=Activation('linear', name="regression_outputs")(process_parameter_reg)
		cla_distrbution=Activation('sigmoid', name="classification_outputs")(process_parameter_cla)
		
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
		
		model=Model(inputs, outputs=output_list, name='Hybrid_Unet_Model')

		model.compile(optimizer=tf.keras.optimizers.Adam(lr=hp_learning_rate),experimental_run_tf_function=False,loss=overall_loss_dict,metrics=overall_metrics_dict,loss_weights=overall_loss_weights)
		print("3D CNN model successfully compiled")
		#print(model.summary())
		
		#plot_model(model,to_file='Tune_OSER.png',show_shapes=True, show_layer_names=True)
		
		return model