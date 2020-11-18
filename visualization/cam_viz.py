""" Contains classes and methods to visualize the class activation maps of the model for various root causes """

import numpy as np
import tensorflow as tf
#import vis
#import keras
#from vis.utils import utils
#from vis.visualization import visualize_cam
import tensorflow.keras.backend as K

class CamViz:
		
		def __init__(self,model,conv_layer_name):

			self.model=model
			self.conv_layer_name=conv_layer_name

		def get_conv_layer(self,conv_layer_name):

			conv_layer=utils.find_layer_idx(self.model, conv_layer_name)
			return conv_layer

		def grad_cam(self,model_input,kcc_id=0):
			
			final_layer=-1
			grad_top  = visualize_cam(self.model, final_layer,kcc_id, model_input, 
						   penultimate_layer_idx = self.conv_layer,#None,
						   backprop_modifier     = None,
						   grad_modifier         = None)

			return grad_top

		def grad_cam_3d(self,model_input,kcc_id):

			def _watch_layer(layer, tape):
				def decorator(func):
					def wrapper(*args, **kwargs):
						# Store the result of `layer.call` internally.
						layer.result = func(*args, **kwargs)
						# From this point onwards, watch this tensor.
						tape.watch(layer.result)
						# Return the result to continue with the forward pass.
						return layer.result

					return wrapper

				layer.call = decorator(layer.call)
				return layer

			from tensorflow.keras import models
			
			model=self.model
			conv_layer=model.get_layer(self.conv_layer_name)
			#layer_input = self.model.input
			heatmap_model = models.Model([model.inputs], [conv_layer.output, model.output])
			#loss= self.model.layers[-1].output[...,kcc_id]
			#model_prediction = self.model.output[:, kcc_id]
			
			with tf.GradientTape() as gtape:
				#_watch_layer(conv_layer, gtape)
				#print("Check")
				conv_output, predictions = heatmap_model(model_input)
				#loss = predictions[:, np.argmax(abs(predictions[0]))]
				loss = predictions[0][:,kcc_id]
				#loss = predictions[1][:,:,:,:,:]
				#preds = self.model.predict(layer_input)
				#model_prediction = self.model.output[:, np.argmax(abs(preds[0]))]
				grads = gtape.gradient(loss, conv_output)
				#grads = gtape.gradient(model_prediction,cnn_layer.output)

			
			#pooled_grads = K.mean(grads, axis=(0, 1, 2))
			#print(grads.shape)
			#print(conv_output.shape)
			
			#iterate = K.function([layer_input], [pooled_grads, cnn_layer.output[0]])
			#grad_wrt_fmap_fn = K.function([layer_input,K.learning_phase()],[cnn_output,grad_wrt_fmap])

			#fmap_eval, grad_wrt_fmap_eval = iterate(model_input)

			#grad_wrt_fmap_eval /= (np.max(grad_wrt_fmap_eval) + K.epsilon())

			#print(grad_wrt_fmap_eval.shape)
			#alpha_k_c = grad_wrt_fmap_eval.mean(axis=(0,1,2)).reshape((1,1,1,-1))
			#Lc_Grad_CAM = np.maximum(np.sum(fmap_eval*alpha_k_c,axis=-1),0).squeeze()

			#print(alpha_k_c)
			#print(Lc_Grad_CAM)
			fmap_eval=conv_output.numpy()
			grad_wrt_fmap_eval=grads.numpy()
			grad_wrt_fmap_eval /= (np.max(grad_wrt_fmap_eval) + K.epsilon())
			#should feature map be normalized?
			return fmap_eval, grad_wrt_fmap_eval



