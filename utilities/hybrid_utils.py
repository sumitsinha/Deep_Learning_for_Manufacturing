
def get_kcc_struct(path='../config/',filename='kcc_config.csv'):
	import pandas as pd
	kcc_df=pd.read_csv(path+filename)
	kcc_struct=[]

	for index,row in kcc_df.iterrows():
		kcc_dict={'kcc_id':row['kcc_id'],'kcc_name':row['kcc_name'],'kcc_type':row['kcc_type'],'kcc_nominal':row['kcc_nominal']
				 ,'kcc_max':row['kcc_max'],'kcc_min':row['kcc_min']
				 }
		kcc_struct.append(kcc_dict)
	
	return kcc_struct


def split_kcc(data):
	
	kcc_struct=get_kcc_struct()

	base_str="tooling_flag_"
	check_str=["tooling_x_","tooling_y_","tooling_z_"]
	tooling_flag_index=[]
	tooling_param_index=[]
	
	for kcc in kcc_struct:
		
		if(base_str in kcc['kcc_name']):
			#id_val=kcc['kcc_name'][-1]
			#id_val=kcc['kcc_name'].substring(kcc['kcc_name'].lastIndexOf('_')+1,len(kcc['kcc_name']))
			id_val=kcc['kcc_name'].rsplit('_', 1)[-1]
			temp_list=[]
			tooling_flag_index.append(kcc['kcc_id'])        
			for kcc_sub in kcc_struct:
				#print(check_str[0]+id_val)
				if(kcc_sub['kcc_name']==check_str[0]+id_val or kcc_sub['kcc_name']==check_str[1]+id_val or kcc_sub['kcc_name']==check_str[2]+id_val):
					temp_list.append(kcc_sub['kcc_id'])
			tooling_param_index.append(temp_list)

	regression_kccs=[]
	categorial_kccs=tooling_flag_index
	
	print("Splitting Contionous and Categorical KCCs")
	for i in range(len(kcc_struct)):

		if(i not in categorial_kccs):
			regression_kccs.append(i)

	if(len(kcc_struct)==len(regression_kccs)+len(categorial_kccs)):
		print("Valid Split")


	#print(regression_kccs)
	#print(categorial_kccs)
	data_classification=data[:,categorial_kccs]
	data_regression=data[:,regression_kccs]

	return data_regression,data_classification
