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
                  'kcc_name':'pinslot_z_rot',
                 'kcc_type':1,
                 'kcc_nominal':0,
                 'kcc_max':0.3,
                 'kcc_min':-0.2   
        })

kcc_struct.append({'kcc_id':1,
                  'kcc_name':'pinhole_x',
                 'kcc_type':1,
                 'kcc_nominal':0,
                 'kcc_max':0.8,
                 'kcc_min':-0.8   
        })

kcc_struct.append({'kcc_id':2,
                  'kcc_name':'pinhole_y',
                 'kcc_type':1,
                 'kcc_nominal':0,
                 'kcc_max':1,
                 'kcc_min':-0.5   
        })

kcc_struct.append({'kcc_id':3,
                  'kcc_name':'clamp_m1_z',
                 'kcc_type':1,
                 'kcc_nominal':0,
                 'kcc_max':1,
                 'kcc_min':-1   
        })

kcc_struct.append({'kcc_id':4,
                  'kcc_name':'clamp_m2_z',
                 'kcc_type':1,
                 'kcc_nominal':0,
                 'kcc_max':1,
                 'kcc_min':-0.8   
        })

kcc_struct.append({'kcc_id':5,
                  'kcc_name':'clamp_m3_z',
                 'kcc_type':1,
                 'kcc_nominal':0,
                 'kcc_max':1.5,
                 'kcc_min':-1.5   
        })

