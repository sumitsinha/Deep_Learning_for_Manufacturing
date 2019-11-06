class AssemblySystem:

	def __init__(self,assembly_type,assembly_kccs,assembly_kpis)
		assembly_type=self.assembly_type
		assembly_kccs=self.assembly_kccs
		assembly_kpis=self.assembly_kpis

class PartType(AssemblySystem):

	def __init__(self,assembly_type,assembly_kccs,assembly_kpis,part_name,voxel_dim,part_type="Sheet Metal Part")
		super().__init__(assembly_type,assembly_kccs,assembly_kpis)
		self.part_type=part_name
		self.voxel_dim=voxel_dim
		self.part_type=part_type

	def get_nominal_cop(self)
		file_name=self.part_name+'.csv'
		nominal_cop=np.loadtxt(file_name)
		return nominal_cop