class MeasurementSystem:

	def __init__(self,data_type="3D Point Cloud Data",application="In-Line Measurement", system_noise=0.1)
		data_type=self.data_type
		application=self.application
		system_noise=self.system_noise

class HexagonWlsScanner(MeasurementSystem):
	
	def __init__(self,data_type,application, system_noise,part_type,data_format='complete measurement'):
		super().__init__(pata_type,application, system_noise)
		self.part_type=part_type
		self.data_format=data_format

	def get_data(self)
		inference_data=get_WLS_data();
