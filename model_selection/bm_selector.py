# Prepare Benchmarking Input Output depending on type of Inputs and Outputs
# 1 Only Regression
# 2 Only Classification
# 3 Only Shape Error
# 4 Regression + Classification
# 5 Regression + OSE
# 6 Classification + OSE
# 7 Regression + Classification + OSE

def get_bm_io(option_num,kcc_regression=0,kcc_classification=0,shape_error=0):
	
	def one():
		Y_out_list=[]
		Y_out_list.append(kcc_regression)
		
		return Y_out_list

	def two():
		Y_out_list=[]
		Y_out_list.append(kcc_classification)

		return Y_out_list

	def three():
		Y_out_list=[]
		Y_out_list.append(shape_error)
		
		return Y_out_list

	def four():
		Y_out_list=[]
		Y_out_list.append(kcc_regression)
		Y_out_list.append(kcc_classification)    

		return Y_out_list

	def five():
		Y_out_list=[]
		Y_out_list.append(kcc_regression)
		Y_out_list.append(shape_error)

		return Y_out_list

	def six():
		Y_out_list=[]
		Y_out_list.append(kcc_classification)
		Y_out_list.append(shape_error)
		
		return Y_out_list

	def seven():
		Y_out_list=[]
		Y_out_list.append(kcc_regression)
		Y_out_list.append(kcc_classification)   
		Y_out_list.append(shape_error)
		
		return Y_out_list

	options = {
		   1 : one,
		   2 : two,
		   3 : three,
		   4 : four,
		   5 : five,
		   6 : six,
		   7 : seven,
		}

	return options[option_num]()