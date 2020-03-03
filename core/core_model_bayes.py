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
		import tensorflow as tf
		import tensorflow_probability as tfp
		tfd = tfp.distributions
		import numpy as np
		negloglik = lambda y, rv_y: -rv_y.log_prob(y)
		
		aleatoric_std=0.0001
		
		#constant aleatoric uncertainty

		def _softplus_inverse(x):
  			"""Helper which computes the function inverse of `tf.nn.softplus`."""
  			return tf.math.log(tf.math.expm1(x))
		
		if(self.output_type=="regression"):
			final_layer_avt='linear'

		if(self.output_type=="classification"):
			final_layer_avt='softmax'

		model = tf.keras.Sequential([
			tf.keras.layers.InputLayer(input_shape=(voxel_dim,voxel_dim,voxel_dim,deviation_channels)),
			tfp.layers.Convolution3DFlipout(32, kernel_size=(5,5,5),strides=(2,2,2),activation=tf.nn.relu),
			tfp.layers.Convolution3DFlipout(32, kernel_size=(4,4,4),strides=(2,2,2),activation=tf.nn.relu),
			tfp.layers.Convolution3DFlipout(32, kernel_size=(3,3,3),strides=(1,1,1),activation=tf.nn.relu),
			tf.keras.layers.MaxPooling3D(pool_size=[2, 2, 2]),
			tf.keras.layers.Flatten(),
			tfp.layers.DenseFlipout(128,activation=tf.nn.relu),
			tfp.layers.DenseFlipout(64,activation=tf.nn.relu),
			tfp.layers.DenseFlipout(self.output_dimension),
			tfp.layers.DistributionLambda(lambda t: tfd.MultivariateNormalDiag(loc=t[..., :self.output_dimension], scale_diag=[aleatoric_std,aleatoric_std,aleatoric_std,aleatoric_std,aleatoric_std,aleatoric_std])),
			])

		#negloglik = lambda y, p_y: -p_y.log_prob(y)
		model.compile(optimizer=tf.optimizers.Adam(),experimental_run_tf_function=False, loss=negloglik,metrics=[tf.keras.metrics.MeanAbsoluteError()])
		print("3D CNN model successfully compiled")
		print(model.summary())
		return model


if (__name__=="__main__"):
	print('Bayesian Deep Learning Class')