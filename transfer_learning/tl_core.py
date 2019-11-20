import os
import sys
current_path=os.path.dirname(__file__)
parentdir = os.path.dirname(current_path)
sys.path.append("../Vizvalization")
sys.path.append("../utilities")
sys.path.append("../datasets")
sys.path.append("../trained_models")
path_var=os.path.join(os.path.dirname(__file__),"../utilities")
sys.path.append(path_var)
sys.path.insert(0,parentdir)

class TransferLearning:

	def __init__(self, tl_type,tl_base,tl_app):
		self.tl_type=tl_type
		self.tl_base=tl_base
		self.tl_app=tf_app
	
	def get_trained_model(model_name):
		model_path='./trained_models/'+model_name+'.h5'
		transfer_model=load_model(model_path)
		return model

	def tl_mode1(model):
		
		
	def tl_mode2(model):

	def tl_mode3(model):




	return model

if __name__ == '__main__':

	#Parsing from Config File
	data_type=assembly_system['data_type']
	application=assembly_system['application']
	part_type=assembly_system['part_type']
	data_format=assembly_system['data_format']
	assembly_type=assembly_system['assembly_type']
	assembly_kccs=assembly_system['assembly_kccs']	
	assembly_kpis=assembly_system['assembly_kpis']
	voxel_dim=assembly_system['voxel_dim']
	point_dim=assembly_system['point_dim']
	voxel_channels=assembly_system['voxel_channels']
	noise_levels=assembly_system['noise_levels']
	noise_type=assembly_system['noise_type']
	mapping_index=assembly_system['index_conv']
	file_names=assembly_system['data_files']
	
	#Objects of Measurement System, Assembly System, Get Infrence Data
	measurement_system=HexagonWlsScanner(data_type,application, system_noise,part_type,data_format)
	vrm_system=VRMSimulationModel(assembly_type,assembly_kccs,assembly_kpis,voxel_dim,point_dim,voxel_channels,noise_levels,noise_type)
	get_data=GetInferenceData();

	print('Importing and preprocessing Cloud-of-Point Data')
	
	get_train_data=GetTrainData(vrm_system)
	dataset=get_train_data.data_import(file_names)
	point_index=load_mapping_index(mapping_index)
	input_conv_data, kcc_subset_dump=get_train_data.data_convert_voxel(dataset,point_index)

	output_dimension=assembly_kccs
	dl_model=DLModel(output_dimension)
	model=dl_model.CNN_model_3D()

	train_model=TrainModel()
	trained_model,eval_metrics=train_model.run_train_model(model,input_conv_data,kcc_subset_dump)

	print("Model Training Complete..")
	print("The Model Validation Metrics are ")
	print(eval_metrics)

