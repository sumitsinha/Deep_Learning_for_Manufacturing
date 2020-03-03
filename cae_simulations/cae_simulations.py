import sys
import matlab.engine
from assembly_system import VRMSimulationModel

class CAESimulations(VRMSimulationModel):
	
	def __init__(self, simulation_platform,simulation_engine,max_run_length,case_study):
		self.simulation_platform=simulation_platform
		self.simulation_engine=simulation_engine
		self.max_run_length=max_run_length
		self.case_study=case_study
	
	def run_simulations(self,run_id,type_flag='train')

		run_status=0
		if(run_id>self.max_run_length):
			return run_status

		if(run_id<=self.max_run_length):
			print("Initiating Matlab Engine...")
			eng = matlab.engine.start_matlab()
			eng.cd(r'C:\Users\sinha_s\Desktop\VRM - GUI - datagen\Demos',nargout=0)
			print("Initiating CAE simulations for run ID: ",run_id)
			print("Displaying MatLab console output")
			file_identifier=str(run_id)+'_'+type_flag
			if(self.case_study=='inner_rf_assembly')
				eng.main_multi_station_door_hinge(file_identifier,nargout=0)
			print("CAE simulations complete...")
			run_status=1
			return run_status



