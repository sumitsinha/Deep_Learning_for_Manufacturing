class DL_BM_Arch:    
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

	
	def fcnn(self,depth=3,filter_root=32,output_heads=2,voxel_dim=64,deviation_channels=3):
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

		mse_basic = tf.keras.losses.MeanSquaredError()

		overall_loss_dict={
		"shape_error_outputs":mse_basic
		}

		overall_loss_weights={
		"shape_error_outputs":1.0
		}

		overall_metrics_dict={
		"shape_error_outputs":[tf.keras.metrics.MeanAbsoluteError()]
		}

		Conv = Conv3D
		MaxPooling = MaxPooling3D
		UpSampling = UpSampling3D
		
		activation='relu'
		final_activation='linear'

		input_size=(voxel_dim,voxel_dim,voxel_dim,deviation_channels)
		inputs = Input(input_size)
		x = inputs

		# Down sampling
		for i in range(depth):
			out_channel = 2**i * filter_root

			# First Conv Block with Conv, BN and activation
			conv1 = Conv(out_channel, kernel_size=3,padding='same', name="Conv{}_1".format(i))(x)

			act1 = Activation(activation, name="Act{}_1".format(i))(conv1)

			#DownSampling
			x = MaxPooling(padding='same', name="MaxPooling{}_1".format(i))(act1)

		# Upsampling
		for i in range(depth - 1, -1, -1):
			out_channel = 2**(i) * filter_root

			#Upsampling
			up1 = UpSampling(name="UpSampling{}_1".format(i))(x)
			up_conv1 = Conv(out_channel, 2, activation='relu', padding='same', name="upConvSam{}_1".format(i))(up1)


			#  Convolutions
			up_conv2 = Conv(out_channel, 3, padding='same', name="upConv{}_1".format(i))(up_conv1)

			up_act1 = Activation(activation, name="upAct{}_1".format(i))(up_conv2)

			x = Activation(activation, name="upAct{}_2".format(i))(up_act1)


		output_list=[]
		#output_list.append(reg_distrbution)
		#output_list.append(cla_distrbution)
		
		output = Conv(deviation_channels*output_heads, 1, padding='same', activation=final_activation, name='shape_error_outputs')(x)
		output_list.append(output)
		
		model=Model(inputs, outputs=output_list, name='FCNN')

		model.compile(optimizer=tf.keras.optimizers.Adam(),experimental_run_tf_function=False,loss=overall_loss_dict,metrics=overall_metrics_dict,loss_weights=overall_loss_weights)
		print("3D FCNN model successfully compiled")
		print(model.summary())
		
		plot_model(model,to_file='FCNN.png',show_shapes=True, show_layer_names=True)
		
		return model