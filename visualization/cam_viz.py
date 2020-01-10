""" Contains classes and methods to visualize the class activation maps of the model for various root causes """

import numpy as np
import vis
import keras
from vis.utils import utils
from vis.visualization import visualize_cam
import keras.backend as K

class CamViz:
		
		def __init__(self,model,conv_layer_name):

			self.model=model
			self.conv_layer=self.get_conv_layer(conv_layer_name)

		def get_conv_layer(self,conv_layer_name):

			conv_layer=utils.find_layer_idx(self.model, conv_layer_name)
			return conv_layer

		def grad_cam(self,model_input,kcc_id):
			
			final_layer=-1
			grad_top  = visualize_cam(self.model, final_layer,kcc_id, model_input, 
                           penultimate_layer_idx = self.conv_layer,#None,
                           backprop_modifier     = None,
                           grad_modifier         = None)

			return grad_top

		def grad_cam_3d(self,model_input,kcc_id):

			cnn_output=self.model.layers[self.conv_layer].output
			layer_input = self.model.input

			loss= self.model.layers[-1].output[...,kcc_id]

			grad_wrt_fmap = K.gradients(loss,cnn_output)[0]

			grad_wrt_fmap_fn = K.function([layer_input,K.learning_phase()],
                                  [cnn_output,grad_wrt_fmap])

			fmap_eval, grad_wrt_fmap_eval = grad_wrt_fmap_fn(model_input)

			grad_wrt_fmap_eval /= (np.max(grad_wrt_fmap_eval) + K.epsilon())

			print(grad_wrt_fmap_eval.shape)
			#alpha_k_c = grad_wrt_fmap_eval.mean(axis=(0,1,2)).reshape((1,1,1,-1))
			#Lc_Grad_CAM = np.maximum(np.sum(fmap_eval*alpha_k_c,axis=-1),0).squeeze()

			#print(alpha_k_c)
			#print(Lc_Grad_CAM)

			return fmap_eval, grad_wrt_fmap_eval



