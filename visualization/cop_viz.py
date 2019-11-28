from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import plotly as py
import plotly.graph_objs as go

class CopViz():

	def __init__(self,nominal_cop):

		self.nominal_cop=nominal_cop
		
	def plot_cop(self,plot_file_name):
		
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
			margin=dict(
				l=0,
				r=0,
				b=0,
				t=0
			)
		)
		fig = go.Figure(data=data, layout=layout)
		py.offline.plot(fig, filename=plot_file_name)

	def get_data_stacks(self,node_ids):
		
		nominal_cop=self.nominal_cop
		node_ids=node_ids[node_ids.Feature_Importance!=0]
		selected_nodes=node_ids['node_ID'].tolist()
		selected_points=df_nom[node_id_kcc,:]
		return stack


	def plot_multiple_stacks(self,stack):

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
			x=stack[:,0],
			y=stack[:,1],
			z=stack[:,2],
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

		data = [trace1,trace2]
		layout = go.Layout(
			margin=dict(
				l=0,
				r=0,
				b=0,
				t=0
			)
		)
		fig = go.Figure(data=data, layout=layout)
		py.offline.plot(fig, filename='cop_stack_plots')

		def plot_voxelized_data(self,voxel_data,nominal_cop,cop_mapping,plot_file_name):

			#Working with y_deviations
			voxel_dim=len(voxel_cop[:,:,:,1])
			grip_len=np.count_nonzero(voxel_cop)
			grid_cop=np.zeros((grid_len,3))
			
			print('Grid Length')

			for i in range(voxel_dim):
				for j in range(voxel_dim):
					for k in range(voxel_dim):
						if(voxel_data[i,j,k,1]!=0):
							grid_cop[index,0]=i
							grid_cop[index,1]=j
							grid_cop[index,2]=k
					  
			for i in range(grid_len):
				for j in range(len(nominal_cop)):
					if((grid_cop[i,0]==cop_mapping[j,0] and grid_cop[i,1]==cop_mapping[j,1] and grid_cop[i,2]==cop_mapping[j,2])):
						grip_cop_values[i,:]=df_nom[j,:]              

			trace1 = go.Scatter3d(
				x=grip_cop_values[:,0],
				y=grip_cop_values[:,1],
				z=grip_cop_values[:,2],
				mode='markers',
				marker=dict(
					size=5,
					line=dict(
						color='rgba(217, 217, 217, 5)',
						width=0.1
					),
					opacity=1
				)
			)
					

			data = [trace1]
			layout = go.Layout(
				margin=dict(
					l=0,
					r=0,
					b=0,
					t=0
				)
			)
			fig = go.Figure(data=data, layout=layout)
			py.offline.plot(fig, filename=plot_file_name)