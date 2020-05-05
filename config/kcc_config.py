"""The Process Parameter/KCC configuration file consists of parameters of each KCC of the assembly system. This is required for sampling/active learning strategy. Each KCC is appended as a dictionary in a list of KCCs. Each KCC has the following parameters
                
        :param kcc_struct[index]['kcc_id']: Unigue identifier for the KCC
        :type kcc_struct[index]['kcc_id']: int (required)

        :param kcc_struct[index]['kcc_name']: The name/description of the KCC
        :type kcc_struct[index]['kcc_name']: str (required)

        :param kcc_struct[index]['kcc_id']: Unigue identifier for the KCC
        :type kcc_struct[index]['kcc_id']: int (required)

        :param kcc_struct[index]['kcc_type']: The type of the KCC, see below for more details
        :type kcc_struct[index]['kcc_type']: int (required)

        :param kcc_struct[index]['kcc_nominal']: The nominal value of the KCC
        :type kcc_struct[index]['kcc_nominal']: float (required)

        :param kcc_struct[index]['kcc_max']: The maximum value of the KCC
        :type kcc_struct[index]['kcc_max']: float (required)

        :param kcc_struct[index]['kcc_min']: The minimum value of the KCC
        :type kcc_struct[index]['kcc_min']: float (required)
"""

kcc_struct=[]

kcc_struct.append({'kcc_id':0,
                  'kcc_name':'pv3_1',
                 'kcc_type':1,
                 'kcc_nominal':0,
                 'kcc_max':2,
                 'kcc_min':-2   
        })

kcc_struct.append({'kcc_id':1,
                  'kcc_name':'pv3_2',
                 'kcc_type':1,
                 'kcc_nominal':0,
                 'kcc_max':2,
                 'kcc_min':-2   
        })

kcc_struct.append({'kcc_id':2,
                  'kcc_name':'pv4_3',
                 'kcc_type':1,
                 'kcc_nominal':0,
                 'kcc_max':2,
                 'kcc_min':-2   
        })

kcc_struct.append({'kcc_id':3,
                  'kcc_name':'pv4_4',
                 'kcc_type':1,
                 'kcc_nominal':0,
                 'kcc_max':2,
                 'kcc_min':-2   
        })

kcc_struct.append({'kcc_id':4,
                  'kcc_name':'rot_pinhole',
                 'kcc_type':1,
                 'kcc_nominal':0,
                 'kcc_max':1,
                 'kcc_min':-1   
        })

kcc_struct.append({'kcc_id':5,
                  'kcc_name':'clamp_1',
                 'kcc_type':1,
                 'kcc_nominal':0,
                 'kcc_max':2,
                 'kcc_min':-2   
        })

kcc_struct.append({'kcc_id':6,
                  'kcc_name':'clamp_2',
                 'kcc_type':1,
                 'kcc_nominal':0,
                 'kcc_max':2,
                 'kcc_min':-2   
        })
kcc_struct.append({'kcc_id':7,
                  'kcc_name':'clamp_3',
                 'kcc_type':1,
                 'kcc_nominal':0,
                 'kcc_max':3,
                 'kcc_min':-3   
        })
kcc_struct.append({'kcc_id':8,
                  'kcc_name':'clamp_4',
                 'kcc_type':1,
                 'kcc_nominal':0,
                 'kcc_max':2,
                 'kcc_min':-2   
        })
kcc_struct.append({'kcc_id':9,
                  'kcc_name':'clamp_5',
                 'kcc_type':1,
                 'kcc_nominal':0,
                 'kcc_max':2,
                 'kcc_min':-2   
        })
kcc_struct.append({'kcc_id':10,
                  'kcc_name':'clamp_6',
                 'kcc_type':1,
                 'kcc_nominal':0,
                 'kcc_max':2,
                 'kcc_min':-2   
        })
kcc_struct.append({'kcc_id':11,
                  'kcc_name':'clamp_7',
                 'kcc_type':1,
                 'kcc_nominal':0,
                 'kcc_max':2,
                 'kcc_min':-2   
        })
