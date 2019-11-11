import xgboost as xgb
from sklearn importRandomForestRegressor
from sklearn.model_selection import train_test_split

def kmc_model_build(point_data,selected_kcc,kcc_name,save_model=0):
	
	train_X, test_X, train_y, test_y = train_test_split(point_data, selected_kcc, test_size = 0.2)
	train=train_X
    target=train_y
    train.index=range(0,train.shape[0])
    target.index=range(0,train.shape[0])
    
    #%%
    print('KMC Generation for selected :', kcc_name)
    #model=RandomForestRegressor(n_estimators=1000,max_depth=700,n_jobs=-1,verbose=True)
    model=xgb.XGBRegressor(colsample_bytree=0.4,gamma=0.045,learning_rate=0.07,max_depth=500,min_child_weight=1.5,n_estimators=500,reg_alpha=0.65,reg_lambda=0.45,subsample=0.95,n_jobs=-1,verbose=True)
    model.fit(train,target)
    #%%
    y_pred = model.predict(test_X)
    mae=metrics.mean_absolute_error(test_y, y_pred)
    
    print('The MAE for feature selection for: ',kcc_name)
    print(mae)
    
    if(save_model=1)
        filename = kcc_name+'_XGB_model.sav'
        joblib.dump(model, filename)
        print('Trained Model Saved to disk....')
    
    #%%
    thresholds = model.feature_importances_
    sorted_thresholds=np.sort(thresholds)
    #%%
    node_id=np.arange(point_dim)
    node_IDs = pd.DataFrame(thresholds, index=node_id)
    node_IDs.columns=['Feature_Importance']
    node_IDs.index.name='node_ID'
    #%%
    node_IDs = node_IDs.sort_values('Feature_Importance', ascending=False)
    filtered_nodeIDs=node_IDs.loc[node_IDs['Feature_Importance'] != 0]
    node_ID_list = filtered_nodeIDs.index.tolist()
    filename=kcc_name+'.csv'
    print('Saving KMCs to disk...')
    filtered_nodeIDs.to_csv(filename)
    return filtered_nodeIDs

def viz_kmc(kcc_name,copviz):    
    filename=kcc_name+'.csv'
    node_ids = pd.read_csv(filename)
    stack=copviz.get_data_stacks(node_ids)
    copviz.plot_multiple_stacks(stack)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Arguments to initiate Measurement System Class and Assembly System Class")
    parser.add_argument("-D", "--data_type", help = "Example: 3D Point Cloud Data", required = False, default = "3D Point Cloud Data")
    parser.add_argument("-A", "--application", help = "Example: Inline Root Cause Analysis", required = False, default = "Inline Root Cause Analysis")
    parser.add_argument("-P", "--part_type", help = "Example: Door Inner and Hinge Assembly", required = False, default = "Door Inner and Hinge Assembly")
    parser.add_argument("-F", "--data_format", help = "Example: Complete vs Partial Data", required = False, default = "Complete")
	parser.add_argument("-S", "--assembly_type", help = "Example: Multi-Stage vs Single-Stage", required = False, default = "Single-Stage")
    parser.add_argument("-C", "--assembly_kccs", help = "Number of KCCs for the Assembly", required = False, default =15,type=int )
    parser.add_argument("-I", "--assembly_kpis	", help = "Number of KPIs for the Assembly", required = False, default = 6,type=int)
    parser.add_argument("-V", "--voxel_dim", help = "The Granularity of Voxels - 32 64 128", required = False, default = 64,type=int)
    parser.add_argument("-P", "--point_dim", help = "Number of key Nodes", required = True, type=int)
    parser.add_argument("-C", "--voxel_channels", help = "Number of Channels - 1 or 3", required = False, default = 1,type=int)
    parser.add_argument("-N", "--noise_levels", help = "Amount of Artificial Noise to add while training", required = False, default = 0.1,type=float)
    parser.add_argument("-T", "--noise_type", help = "Type of noise to be added uniform/Gaussian default uniform", required = False, default = "uniform")
	argument = parser.parse_args()
	
	data_type=argument.data_type
	application=argument.application
	part_type=argument.part_type
	data_format=argument.data_format
	assembly_type=argument.assembly_type	
	assembly_kccs=argument.assembly_kccs	
	assembly_kpis=argument.assembly_kpis
	voxel_dim=argument.voxel_dim
	point_dim=argument.point_dim
	voxel_channels=argument.voxel_channels
	noise_levels=argument.noise_levels
	noise_type=argument.noise_type

	#Objects of Measurement System and Assembly System
	measurement_system=HexagonWlsScanner(data_type,application, system_noise,part_type,data_format)
	vrm_system=VRMSimulationModel(assembly_type,assembly_kccs,assembly_kpis,voxel_dim,point_dim,voxel_channels,noise_levels,noise_type)

	print('Importing and preprocessing Cloud-of-Point Data')
	
	file_names=['car_halo_run1_ydev.csv','car_halo_run2_ydev.csv','car_halo_run3_ydev.csv','car_halo_run4_ydev.csv','car_halo_run5_ydev.csv']
	get_train_data=GetTrainData(vrm_system)
	dataset=get_train_data.data_import(file_names)

	kcc_id=[]
    kmc_list=[]
	point_data=dataset[:, 0:point_dim]

    print('Generating KMC for all KCCs')
	for i in range(kcc_dim):
		kcc_name="KCC_"+str(i+1)
		kcc_id.append(kcc_name)
    	selected_kcc=dataset[:,point_dim:point_dim+i]
        kmc_list[i].append(kmc_model_build(point_data,selected_kcc,kcc_name))

    plot_kmc=1;
    kmc_tplot="KCC_1"
    filename=+'.csv'
    if(plot_kmc==1):
        print("Plotting KMCs for: ",kmc_tplot)
        print(print)
        copviz=CopViz(vrm_system)
        viz_kmc(filename,copviz)






