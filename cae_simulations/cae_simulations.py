import sys

class CAESimulations():
	
	def __init__(self, simulation_platform,simulation_engine,max_run_length,case_study):
		self.simulation_platform=simulation_platform
		self.simulation_engine=simulation_engine
		self.max_run_length=max_run_length
		self.case_study=case_study
	


	def run_simulations(self,run_id,type_flag='train'):
		
		run_status=0
		
		if(run_id>self.max_run_length):
			return run_status

		if(run_id<=self.max_run_length):
			import matlab.engine
			print("Initiating Matlab Engine...")
			
			#Initiating CAE engine within AI environment
			eng = matlab.engine.start_matlab()
			
			#change to absolute path here
			#eng.cd(r'C:\Users\sinha_s\Desktop\VRM - GUI - datagen\Demos',nargout=0)
			#Chnaging to Cross Member Assembly
			eng.cd(r'C:\Users\SINHA_S\Desktop\cross_member_datagen\Demos\Fixture simulation\Multi station\[4] Cross member',nargout=0)

			print("Initiating CAE simulations for run ID: ",run_id)
			print("Displaying MatLab console output")
			
			#file_identifier=str(run_id)+'_'+type_flag
			
			#Running CAE Simulations

			#Inner RF Assembly
			if(self.case_study=='inner_rf_assembly'):
				eng.main_multi_station_door_hinge(run_id,type_flag,nargout=0)

			#Cross Member Assembly
			if(self.case_study=='cross_member_assembly'):
				eng.fixture_simulation_multi_station_cross_member_simulation_dlmfg(run_id,type_flag,nargout=0)
			print("CAE simulations complete...")
			eng.quit()

			run_status=1
			return run_status



