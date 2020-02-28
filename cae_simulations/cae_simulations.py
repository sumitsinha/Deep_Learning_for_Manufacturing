import sys
import matlab.engine

class CAESimulations:
	
	def __init__(self, simulation_platform,simulation_engine,max_run_length):
		self.simulation_platform=simulation_platform
		self.simulation_engine=simulation_engine
		self.max_run_length=max_run_length
	
	def run_simulations(self,run_id)

		run_status=0
		if(run_id>self.max_run_length):
			return run_status

		if(run_id<=self.max_run_length):
			print("Initiating Matlab Engine...")
			eng = matlab.engine.start_matlab()
			eng.cd(r'C:\Users\sinha_s\Desktop\VRM - GUI - datagen\Demos',nargout=0)
			print("Initiating CAE simulations for run ID: ",run_id)
			print("Displaying MatLab console output")
			eng.main_multi_station_door_hinge(run_id,nargout=0)
			print("CAE simulations complete...")
			run_status=1
			return run_status



