class MeasurementSystem:

	def __init__(self,data_type,application, system_noise):
		self.data_type=data_type
		self.application=application
		self.system_noise=system_noise

class HexagonWlsScanner(MeasurementSystem):
	
	def __init__(self,data_type,application, system_noise,part_type,data_format='complete measurement'):
		super(HexagonWlsScanner,self).__init__(data_type,application,system_noise)
		self.part_type=part_type
		self.data_format=data_format

	def get_data(self):
		inference_data=get_WLS_data();
