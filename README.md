<a href="https://warwick.ac.uk/fac/sci/wmg/research/digital/dlm/" >
<img align="right" src="http://www.thebiponline.co.uk/bip/wp-content/uploads/2013/08/University-of-Warwick-WMG.png" alt="WMG" width="100">
</a>

# Bayesian Deep Learning for Manufacturing 2.0 (dlmfg)
## Object Shape Error Response (OSER)




[**Digital Lifecycle Management - In Process Quality Improvement (IPQI)**](https://warwick.ac.uk/fac/sci/wmg/research/digital/dlm/projects/ipqi_new)




[![Generic badge](https://img.shields.io/badge/DOI-10.1117/12.2526062-blue.svg)](https://doi.org/10.1117/12.2526062) [![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/) [![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://warwick.ac.uk/fac/sci/wmg/research/digital/dlm/projects/ipqi_new/) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) [![PyPI status](https://img.shields.io/pypi/status/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/) [![Generic badge](https://img.shields.io/badge/Documentation-WIP-red.svg)]() [![Generic badge](https://img.shields.io/badge/PyPI-WIP-red.svg)]() ![](https://img.shields.io/badge/keras-tensorflow-blue.svg)


***
## Overview 
The open source **Bayesian Deep Learning for Manufacturing (dlmfg) Library** is built using a **TensorFlow**, **TensorFlow Probablity** and **Keras** back end to build:  

* Bayesian deep learning models such as **Bayesian 3D Convolutional Neural Network and Bayesian 3D U-net** to enable **root cause analysis** in Manufacturing Systems. 

* Deep reinforcement learning models such as **Deep Deterministic Policy Gradients** to enable **control and correction** in Manufacturing Systems. 


The library can be used across various domains such as assembly systems, stamping, additive manufacturing and milling where the key problem is **Object Shape Error Detection and Estimation**. The library is build using Object Oriented Programming to enable extension and contribution from other related disciplines within the artificial intelligence community as well as the manufacturing community.

## Video 
[**A Video for the work can be found here**](https://github.com/sumitsinha/Deep_Learning_for_Manufacturing/tree/master/resources/video)

## Published Work 
The published work can be found below: 

* [**3D convolutional neural networks to estimate assembly process parameters using 3D point-clouds**](https://www.researchgate.net/publication/333942071_3D_convolutional_neural_networks_to_estimate_assembly_process_parameters_using_3D_point-clouds)

* [**Deep learning enhanced digital twin for closed loop in-process quality improvement**](https://www.sciencedirect.com/science/article/abs/pii/S0007850620301323)


*Two follow up papers are currently under review are expected by Jan 2021*


## Documentation
The complete documentation and other ongoing research can be found here: [**Documentation and Research**](https://sumitsinha.github.io/Deep_Learning_for_Manufacturing/html/index.html).


## Highlights and New Additions

1. [**Bayesian 3D U-Net**](https://github.com/sumitsinha/Deep_Learning_for_Manufacturing/blob/master/core/bayes_unet_hybrid_train.py) model integrating Bayesian layers and attention blocks for uncertainty quantification and superior decoder performance leveraging the *where to look capability* with multi-task capabilities to estimate bot real-valued(regression) and categorical(classification) based values. The Decoder is used to obtain real-valued segmentation maps
2.  [**Deep Reinforcement Learning**](https://github.com/sumitsinha/Deep_Learning_for_Manufacturing/tree/master/deep_reinforcement_learning) using deep deterministic policy gradient (DDPG) and a custom made multi physics manufacturing environment to build agents to correct manufacturing systems
3.  [**Closed Loop Sampling**](https://github.com/sumitsinha/Deep_Learning_for_Manufacturing/blob/master/core/dynamic_adaptive_model_train.py) for faster model training and convergence using epistemic uncertainty of the Bayesian CNN models
4.  [**Matlab Python Integration**](https://github.com/sumitsinha/Deep_Learning_for_Manufacturing/tree/master/cae_simulations) to enable low latency connection between multi-physics manufacturing environments (Matlab) and TensorFlow based DDPG agents
5.  [**Multi-Physics Manufacturing System Simulations**](https://github.com/sumitsinha/Deep_Learning_for_Manufacturing/tree/master/cae_simulations/cae_matlab) to generate custom datasets for various fault scenarios using Variation Response Method (VRM) kernel
6.  [**Uncertainty guided continual learning**](https://github.com/sumitsinha/Deep_Learning_for_Manufacturing/tree/master/continual_learning) to enable life long/incremental training for multiple case studies
7.  [**Automated Model Selection**](https://github.com/sumitsinha/Deep_Learning_for_Manufacturing/tree/master/model_selection) using Keras Tuner that enables hyperparameter optimization and benchmarking for various deep learning architectures
8.  [**Exploratory notebooks**](https://github.com/sumitsinha/Deep_Learning_for_Manufacturing/tree/master/assembly_eda_studies) for various case studies
9. [**3D Gradient-weighted Class Activation Maps**](https://github.com/sumitsinha/Deep_Learning_for_Manufacturing/blob/master/core/u_net_model_deploy_multi_output.py) for interpretability of deep learning models 
10.  [**Datasets for Industrial multi-station case studies**](https://github.com/sumitsinha/Deep_Learning_for_Manufacturing/tree/master/pre_trained_models) for training and benchmarking deep learning models 


## Installation
The library can be cloned using: 


    Git clone https://github.com/sumitsinha/Deep_Learning_for_Manufacturing

## Dataset Download
The datasets can be download by running the download_data.py file within the downloads file. The specifics of the download can be specified in the download\config.py file.

The library consists of the following two key datasets:

1.	[**3D Cloud of Point data with node deviations and process parameters for Single Part Car Halo Reinforcement**](https://sumitsinha.github.io/Deep_Learning_for_Manufacturing/html/case_study_halo.html) – Obtained due to variations in the Measurement Station locators and Stamping Process
2.	[**3D Cloud of Point data with node deviations and process parameters for Two part assembly for Car Door Inner and Hinge Reinforcement**](https://sumitsinha.github.io/Deep_Learning_for_Manufacturing/html/case_study_inner_rf.html) – Obtained due to variations in the Assembly System locators and joining tools.


## Bayesian 3D CNN Model Architecture
Motivated by the recent development of Bayesian Deep Neural Networks Bayesian models considering parameters to be distributions have been build using TensorFlow Probability. The Aleatoric uncertainty have been modelled using Multi-variate normal distributions as outputs while the epistemic distributions have been modelled using distributions on model parameters by using Flip-out layers.

The Bayesian 3D CNN model for single station system has the following layers:
![Bayesian 3D CNN Model](https://github.com/sumitsinha/Deep_Learning_for_Manufacturing/blob/master/model_architecture/bayes_3d_cnn.png)

```python
> negloglik = lambda y, rv_y: -rv_y.log_prob(y)
> model = tf.keras.Sequential([
tf.keras.layers.InputLayer(input_shape=(voxel_dim,voxel_dim,voxel_dim,deviation_channels)), tfp.layers.Convolution3DFlipout(32, kernel_size=(5,5,5),strides=(2,2,2),activation=tf.nn.relu),
tfp.layers.Convolution3DFlipout(32, kernel_size=(4,4,4),strides=(2,2,2),activation=tf.nn.relu),
tfp.layers.Convolution3DFlipout(32, kernel_size=(3,3,3),strides=(1,1,1),activation=tf.nn.relu),
tf.keras.layers.MaxPooling3D(pool_size=[2, 2, 2]),
tf.keras.layers.Flatten(),
tfp.layers.DenseFlipout(128,activation=tf.nn.relu),
tfp.layers.DenseFlipout(64,activation=tf.nn.relu),
tfp.layers.DenseFlipout(self.output_dimension),
tfp.layers.DistributionLambda(lambda t: tfd.MultivariateNormalDiag(loc=t[..., :self.output_dimension], scale_diag=aleatoric_tensor)),])
model.compile(optimizer=tf.keras.optimizers.Adam(),loss=negloglik,metrics=[tf.keras.metrics.MeanAbsoluteError()])

```


## Bayesian 3D U-Net Model Architecture
For scaling to multi-station systems consisting of both categorical and continuous process parameters and prediction of point-clouds (object shape error) in previous stations a 3D - Net Attention based architecture is leveraged.	

![Bayesian 3D CNN Model](https://github.com/sumitsinha/Deep_Learning_for_Manufacturing/blob/master/model_architecture/unet_3d_cnn.png)

**Down-sampling Kernel**
![Bayesian 3D CNN Model](https://github.com/sumitsinha/Deep_Learning_for_Manufacturing/blob/master/model_architecture/down_sample.png)

**Attention based Up-Sampling Kernel**
![Bayesian 3D CNN Model](https://github.com/sumitsinha/Deep_Learning_for_Manufacturing/blob/master/model_architecture/up_sample.png)

```python
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
		cla_distrbution=Activation('sigmoid', name="classification_outputs")(process_parameter_cla)
		#cla_distrbution=tfp.layers.DenseFlipout(categorical_kccs, kernel_divergence_fn=kl_divergence_function,activation=tf.nn.sigmoid,name="classification_outputs")(process_parameter_cla)

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
```


## Verification and Validation
Details of verification and validation of the model on an actual system can be found here: [**Real System Implementation**](https://sumitsinha.github.io/Deep_Learning_for_Manufacturing/html/real_system_implementation.html)

## Benchmarking
Benchmarking of the model is done against various deep learning and machine learning approaches to highlight superiority.

![OSER Benchmarking](https://github.com/sumitsinha/Deep_Learning_for_Manufacturing/blob/master/resources/benchmarking/benchmarking.png)

## Decoder Outputs
The segmentations outputs for the Bayesian 3D U-Net model enables estimation of dimensional quality of products in between stages and stations of the process.

![Real-Valued Segmentation Maps](https://github.com/sumitsinha/Deep_Learning_for_Manufacturing/blob/master/resources/segmentation_maps/decoder.png)

## Model Interpretability
3D Grad-weighted Class Activation Maps (3D Grad-CAMS) for each level of the encoder provides insights into the working of the model and integrate a measure of trust in all estimates.

![Encoder 3D Grad-CAMs](https://github.com/sumitsinha/Deep_Learning_for_Manufacturing/blob/master/resources/segmentation_maps/encoder.png)


##### Please cite work as:

    Sinha, S., Glorieux, E., Franciosa, P., & Ceglarek, D. (2019, June). 3D convolutional neural networks to estimate assembly process parameters using 3D point-clouds. In Multimodal Sensing: Technologies and Applications (Vol. 11059, p. 110590B). International Society for Optics and Photonics.


> @inproceedings{Sinha2019,
> author = {Sinha, Sumit and Glorieux, Emile and Franciosa, Pasquale and Ceglarek, Dariusz},
> booktitle = {Multimodal Sensing: Technologies and Applications},
> doi = {10.1117/12.2526062},
> month = {jun},
> pages = {10},
> publisher = {SPIE},
> title = {{3D convolutional neural networks to estimate assembly process parameters using 3D point-clouds}},
> year = {2019}
}



<a href="https://doi.org/10.1117/12.2526062" >
<img src="https://is3-ssl.mzstatic.com/image/thumb/Purple123/v4/44/96/e5/4496e598-1b99-369a-cefe-cb347e538aa4/AppIcon-0-0-1x_U007emarketing-0-0-0-7-0-0-sRGB-0-0-0-GLES2_U002c0-512MB-85-220-0-0.png/1200x630wa.png" alt="WMG" width="100">
</a>

##### Data generation has been done using a Multi-fidelity CAE simulation software known as VRM :

    Franciosa, P., Palit, A., Gerbino, S., & Ceglarek, D. (2019). A novel hybrid shell element formulation (QUAD+ and TRIA+): A benchmarking and comparative study. Finite Elements in Analysis and Design, 166, 103319.

##### Collaboration:
Please contact [**Sumit Sinha**](https://warwick.ac.uk/fac/sci/wmg/research/digital/dlm/people/sumit/), [**Dr Pasquale Franciosa**](https://warwick.ac.uk/fac/sci/wmg/research/digital/dlm/people/p_franciosa) 
in case of any clarifications or collaborative work with the [**Digital Lifecycle Management**](https://warwick.ac.uk/fac/sci/wmg/research/digital/dlm/) group at [**WMG**](https://warwick.ac.uk/fac/sci/wmg/), [**University of Warwick**](https://warwick.ac.uk/)
