#Config File to parametrize the KCCs of the system

kcc_struct=[]

kcc_struct.append({'kcc_id':0,
                  'kcc_name':'pinhole_x',
                 'kcc_type':1,
                 'kcc_nominal':0,
                 'kcc_max':1.5,
                 'kcc_min':-1.5   
        })

kcc_struct.append({'kcc_id':1,
                  'kcc_name':'pinhole_z',
                 'kcc_type':1,
                 'kcc_nominal':0,
                 'kcc_max':1.5,
                 'kcc_min':-1.5   
        })

kcc_struct.append({'kcc_id':2,
                  'kcc_name':'pinslot_z',
                 'kcc_type':1,
                 'kcc_nominal':0,
                 'kcc_max':1,
                 'kcc_min':-1   
        })
