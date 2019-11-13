""" Containts core classes and methods for intilillzing a Assembly System, the inputs are provided in assemblyconfig file in utilities"""

class AssemblySystem:
	"""Assembly System Class

		:param assembly_type: Type of assembly Single-Station/Multi-Station
		:type assembly_system: int (required)

		:param assembly_kccs: Number of KCCs for the assembly
		:type assembly_kccs: int (required)

		:param assembly_kpis: Number of Kpis for the assembly
		:type assembly_kpis: int (required) 
	"""
	def __init__(self,assembly_type,assembly_kccs,assembly_kpis):
		assembly_type=self.assembly_type
		assembly_kccs=self.assembly_kccs
		assembly_kpis=self.assembly_kpis

class PartType(AssemblySystem):
	"""Part System Class, inherits the Assembly System Class, addtional parameters for this class include
		
		:param voxel_dim: Dimension of the voxel
		:type assembly_system: int (required)

		:param voxel_dim: Dimension of the voxel Channel, single channel output - 1 or multi channel - 2,3 (use 1 for deviations in one direction, 2 or 3 if data for multiple deviation directions are present)
		:type assembly_system: int (required)

		:param voxel_dim: Dimension of the voxel
		:type assembly_system: int (required)

		The class containes two functions -  get_nominal_cop and get_nominal_cop_database
	"""
	def __init__(self,assembly_type,assembly_kccs,assembly_kpis,part_name,voxel_dim,point_dim,part_type):
		super().__init__(assembly_type,assembly_kccs,assembly_kpis)
		self.voxel_dim=voxel_dim
		self.voxel_channel=voxel_channel
		self.part_type=part_type

	def get_nominal_cop(self,file_name):
		"""Import nominal cloud-of-point of the assembly from a text/csv file

			:param file_name: Name of the input file
			:type file_name: str (required)

			:returns: numpy array of dim points * 3
			:rtype: np_array [point_dim,3]
		"""
		file_name=self.file_name
		nominal_cop=np.loadtxt(file_name)
		return nominal_cop

	def get_nominal_cop_database(self,conn_str,table_name):
		"""Import nominal cloud-of-point of the assembly from a SQL database assumes the table only contains three coloumns of the noinal COPs in order of the Node IDs		
			
			:param conn_str: Connection String for Database
			:type conn_str: str (required)

			:param table_name: Name of table in the database
			:type table_name: str (required)

			:returns: numpy array of dim points * 3
			:rtype: np_array [point_dim,3]
		"""
		engine = create_engine(conn_str)
		squery ='select * from '+table_name
		df_nom = pd.read_sql_query(squery,con=engine)
		df_nom = df_nom.values
		return df_nom

class VRMSimulationModel(PartType):
	
	"""VRM Simulation Model class inherits the part type class, additional parameters of this class include

		:param noise_level: The level of aritifical noise to be added to simulated data, typically set to 0.1 mm from the measurment system class depending on the scanner
		:type noise_level: float (required)

		:param noise_type: The type of noise to be added, can be gaussin or uniform , for gaussian noise_level is set as standard deviation and mean as zero for uniform the min and max are set -noise_level and +noise_level respectively
		:type noise_type: str (optional)

		:param convergency_flag: Flag to denote if the simulation model had converged while simulating, is set to 1 by default
		:type convergency_flag: int (optional)

		The class containe one function kpi_calculator that needs to be defined by the user depending on the assembly output

	"""
	def __init__(self,assembly_type,assembly_kccs,assembly_kpis,part_name,voxel_dim,part_type,noise_level,noise_type='uniform',convergency_flag=1):
		super().__init__(assembly_type,assembly_kccs,assembly_kpis,part_name,voxel_dim,part_type)
		self.noise_level=noise_level
		self.noise_type=noise_type
		self.convergency_flag=convergency_flag

	def kpi_calculator(self,cop_data,kpi_params=[]):
		""" User defined function to calculate KPI from Cloud of Point Data [KPI]=f(Cop)

			:param cop_data: CoP data for a given sample
			:type cop_data: np_array [point_dim,3] (required)

			:param kpi_params: Various parameters required to calculate the KPI, can be blank if no parameters are required to calculate KPI from CoP
			:type kpi_params: list (optional)

			:returns: list of multivariate KPIs for the given CoP
			:rtype: list

		"""
		
		kpi=[None]*self.assembly_kpis

		#define function here 
		return kpi