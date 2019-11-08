import numpy as np
import pandas as pd

class AssemblySystem:

	def __init__(self,assembly_type,assembly_kccs,assembly_kpis):
		assembly_type=self.assembly_type
		assembly_kccs=self.assembly_kccs
		assembly_kpis=self.assembly_kpis

class PartType(AssemblySystem):

	def __init__(self,assembly_type,assembly_kccs,assembly_kpis,part_name,voxel_dim,point_dim,part_type):
		super().__init__(assembly_type,assembly_kccs,assembly_kpis)
		self.voxel_dim=voxel_dim
		self.voxel_channel=voxel_channel
		self.part_type=part_type

	def get_nominal_cop(self,file_name):
		file_name=self.file_name
		nominal_cop=np.loadtxt(file_name)
		return nominal_cop

	def get_nominal_cop_database(self,conn_str,table_name):
		engine = create_engine(conn_str)
    	squery='select cox,coy,coz from ' + nominal_table
    	df_nom = pd.read_sql_query(squery,con=engine)
    	df_nom=df_nom.values
    	return df_nom

class VRMSimulationModel(PartType):

	def __init__(self,assembly_type,assembly_kccs,assembly_kpis,part_name,voxel_dim,part_type,noise_level,noise_type,convergency_flag=1):
		super().__init__(assembly_type,assembly_kccs,assembly_kpis,part_name,voxel_dim,part_type)
		self.noise_level=noise_level
		self.noise_type=noise_type
		self.convergency_flag=convergency_flag

	def kpi_calculator(self,cop_data,kpi_params)
		#User defined function to calculate KPI from Cloud of Point Data
		kpi=[]
		return kpi