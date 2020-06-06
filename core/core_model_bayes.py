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

if (__name__=="__main__"):
	print('Bayesian Deep Learning Class')
	#model=bayes_cnn_model_3d_hybrid()