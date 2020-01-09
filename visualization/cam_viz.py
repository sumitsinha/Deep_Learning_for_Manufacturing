""" Contains classes and methods to visualize the class activation maps of the model for various root causes """

import numpy as np
import vis
import keras
from vis.utils import utils
from vis.visualization import visualize_cam

class CamViz:
		
		def __init__(self,model,conv_layer_name):

			self.model=model
			self.conv_layer=get_conv_layer(conv_layer_name)

		def get_conv_layer(self,conv_layer_name):

			conv_layer=utils.find_layer_idx(self.model, conv)conv_layer_name
			return conv_layer

		def grad_cam(self,model_input,kcc_id):
			
			final_layer=-1
			grad_top  = visualize_cam(self.model, final_layer,kcc_id, class_idx, model_input, 
                           penultimate_layer_idx = self.conv_layer,#None,
                           backprop_modifier     = None,
                           grad_modifier         = None)

			return grad_top