""" Contains classes and methods for visualizing the cloud of point data, the KMCs and the voxelized cloud of point data"""

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import plotly as py
import plotly.graph_objs as go
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class CopViz:
	"""Cop Visualization class methods and objects to visualize different forms of COP data  

		:param nominal_cop: nominal_cop [nodes*3]
		:type assembly_system: numpy.array (required)

	"""
	def __init__(self,nominal_cop):

		self.nominal_cop=nominal_cop
		
	def plot_cop(self,plot_file_name):
		"""used to plot the COP data using plotly library

			:param plot_file_name: filename with which the plot is saved 
			:type assembly_system: str (required)

		"""
		nominal_cop=self.nominal_cop

		trace1 = go.Scatter3d(
			x=nominal_cop[:,0],
			y=nominal_cop[:,1],
			z=nominal_cop[:,2],
			mode='markers',
			marker=dict(
				size=2,
				opacity=0.8
			)
		)

		data = [trace1]
		layout = go.Layout(
			scene=dict(aspectmode='cube',aspectratio=dict(x=1, y=1, z=0.95)),
			margin=dict(
				l=0,
				r=0,
				b=0,
				t=0
			)
		)
		fig = go.Figure(data=data, layout=layout)
		py.offline.plot(fig, filename=plot_file_name)

	def get_data_stacks(self,node_id_x,node_id_y,node_id_z):
		"""used to obtain co-ordinates for selected node IDs for each axis

			:param node_id_x: List of KMC node_ids considering x deviations 
			:type node_id_x: list (required)

			:param node_id_y: List of KMC node_ids considering y deviations 
			:type node_id_y: list (required)
			
			:param node_id_z: List of KMC node_ids considering z deviations 
			:type node_id_z: list (required)

			:returns: list of KMCs for three axis
			:rtype: list 

		"""
		
		nominal_cop=self.nominal_cop
		selected_nodes_x=(node_id_x['node_id']-1).tolist()
		selected_nodes_y=(node_id_y['node_id']-1).tolist()
		selected_nodes_z=(node_id_z['node_id']-1).tolist()
		
		selected_points_x=nominal_cop[selected_nodes_x,:]
		selected_points_y=nominal_cop[selected_nodes_y,:]
		selected_points_z=nominal_cop[selected_nodes_z,:]

		return [selected_points_x,selected_points_y,selected_points_z]


	def plot_multiple_stacks(self,stack,plot_path):
		"""used to plot all the KMCs as overlay on cloud of point data

			:param stack: List of list of KMCs for three axis
			:type stack: list (required)

			:param plot_path: plot path to save all the KMCs 
			:type plot_path: str (required)

		"""
		nominal_cop=self.nominal_cop

		trace1 = go.Scatter3d(
			x=nominal_cop[:,0],
			y=nominal_cop[:,1],
			z=nominal_cop[:,2],
			mode='markers',
			marker=dict(
				size=2,
				opacity=0.8
			)
		)


		trace2 = go.Scatter3d(
			
			x=stack[0][:,0],
			y=stack[0][:,1],
			z=stack[0][:,2],
			mode='markers',
			marker=dict(
				color='rgb(120, 0, 0)',
				size=8,
				symbol='circle',
				line=dict(
					color='rgb(204, 204, 204)',
					width=1
				),
				opacity=0.7
			)
		)


		trace3 = go.Scatter3d(
			
			x=stack[1][:,0],
			y=stack[1][:,1],
			z=stack[1][:,2],
			mode='markers',
			marker=dict(
				color='rgb(0, 120, 0)',
				size=8,
				symbol='circle',
				line=dict(
					color='rgb(204, 204, 204)',
					width=1
				),
				opacity=0.7
			)
		)


		trace4 = go.Scatter3d(
			
			x=stack[2][:,0],
			y=stack[2][:,1],
			z=stack[2][:,2],
			mode='markers',
			marker=dict(
				color='rgb(0, 0, 120)',
				size=8,
				symbol='circle',
				line=dict(
					color='rgb(204, 204, 204)',
					width=1
				),
				opacity=0.7
			)
		)

		data = [trace1,trace2,trace3,trace4]
		layout = go.Layout(
			margin=dict(
				l=0,
				r=0,
				b=0,
				t=0
			)
		)
		fig = go.Figure(data=data, layout=layout)
		py.offline.plot(fig, filename=plot_path)

	def plot_voxelized_data(self,voxel_data,component):
		"""used to plot the voxelized cloud-of-point data

			:param voxel_data: voxelized COP data 
			:type voxel_data: numpy.array (required)
			
			:param component: The component of deviation to be considered while plotting
			:type component: int (required)
			
			:param plot_file_name: File name to save plot file 
			:type cop_mapping: str (required)
			
		"""
		
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		#ax.set_aspect('equal')
		ax.voxels(voxel_data[:,:,:,component], edgecolor="k")
		plt.show()
		
	
