""" Contains core classes to initlize the measurment system class """

class MeasurementSystem:
	
	"""Measurement System Class

		:param data_type: Type data obtained from the measurment system, Cloud-of-Point/Image/Point based
		:type assembly_system: str (required)

		:param application: Application of the measuremnt system, 
		:type application: str (required)

		:param system_noise: Noise level of the measurement system
		:type system_noise: float (required) 
	"""
	
	def __init__(self,data_type,application, system_noise):
		self.data_type=data_type
		self.application=application
		self.system_noise=system_noise

class HexagonWlsScanner(MeasurementSystem):
	
	"""Hexagon WLS System Class

		:param data_type: Type data obtained from the measurment system, Cloud-of-Point/Image/Point based
		:type assembly_system: str (required)

		:param application: Application of the measuremnt system, 
		:type application: str (required)

		:param system_noise: Noise level of the measurement system
		:type system_noise: float (required) 
	"""

	def __init__(self,data_type,application, system_noise,part_type,data_format='complete measurement'):
		super(HexagonWlsScanner,self).__init__(data_type,application,system_noise)
		self.part_type=part_type
		self.data_format=data_format