#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Bendlets as an intermediate data representation
#Trasnferable and Interpretable
#Bendlets as Graph
#Inductive bias about the assembly process


# In[2]:


#Importing Required Modules
from scipy.io import loadmat
import tensorflow as tf
import sonnet as snt
import graph_nets as gn
from tqdm import tqdm

print("TensorFlow version {}".format(tf.__version__))
print("Sonnet version {}".format(snt.__version__))

from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sonnet as snt
import tensorflow as tf

import stellargraph as sg
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
K.clear_session()

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"]="0" # Nvidia Quadro GV100
#os.environ["CUDA_VISIBLE_DEVICES"]="1" # Nvidia Quadro M2000

#Adding Path to various Modules
sys.path.append("../core")
sys.path.append("../visualization")
sys.path.append("../utilities")
sys.path.append("../datasets")
sys.path.append("../trained_models")
sys.path.append("../config")

#Importing Config files
import assembly_config as config
import model_config as cftrain
import hybrid_utils as hy_util

#Importing required modules from the package
from measurement_system import HexagonWlsScanner
from assembly_system import VRMSimulationModel
from wls400a_system import GetInferenceData
from data_import import GetTrainData
from encode_decode_model import Encode_Decode_Model
from training_viz import TrainViz
from metrics_eval import MetricsEval


# In[3]:


#Extracting Mesh Connectivity

adj_mat=loadmat('./resources/adj_mat.mat')
adj_csr=adj_mat['adj_mat'].tocsr()
adj_csr_n = adj_csr.nonzero()
sparserows = adj_csr_n [0]
sparsecols = adj_csr_n [1]

edges = pd.DataFrame(
    {"source":sparserows , "target": sparsecols}
)
edges


# In[4]:


print('Parsing from Assembly Config File....')

data_type=config.assembly_system['data_type']
application=config.assembly_system['application']
part_type=config.assembly_system['part_type']
part_name=config.assembly_system['part_name']
data_format=config.assembly_system['data_format']
assembly_type=config.assembly_system['assembly_type']
assembly_kccs=config.assembly_system['assembly_kccs']
assembly_kpis=config.assembly_system['assembly_kpis']
voxel_dim=config.assembly_system['voxel_dim']
point_dim=config.assembly_system['point_dim']
voxel_channels=config.assembly_system['voxel_channels']
noise_type=config.assembly_system['noise_type']
mapping_index=config.assembly_system['mapping_index']

system_noise=config.assembly_system['system_noise']
aritifical_noise=config.assembly_system['aritifical_noise']
data_folder=config.assembly_system['data_folder']
kcc_folder=config.assembly_system['kcc_folder']
kcc_files=config.assembly_system['kcc_files']
test_kcc_files=config.assembly_system['test_kcc_files']

print('Parsing Complete....')


# In[5]:


#added for hybrid model
categorical_kccs=config.assembly_system['categorical_kccs']

print('Parsing from Training Config File')

model_type=cftrain.model_parameters['model_type']
output_type=cftrain.model_parameters['output_type']
batch_size=cftrain.model_parameters['batch_size']
epocs=cftrain.model_parameters['epocs']
split_ratio=cftrain.model_parameters['split_ratio']
optimizer=cftrain.model_parameters['optimizer']
loss_func=cftrain.model_parameters['loss_func']
regularizer_coeff=cftrain.model_parameters['regularizer_coeff']
activate_tensorboard=cftrain.model_parameters['activate_tensorboard']


# In[6]:


print('Creating file Structure....')

bn_model_name='bendlets'

folder_name=part_type
train_path='../trained_models/'+part_type
pathlib.Path(train_path).mkdir(parents=True, exist_ok=True) 

train_path=train_path+'/'+bn_model_name
pathlib.Path(train_path).mkdir(parents=True, exist_ok=True) 

model_path=train_path+'/models'
pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)

logs_path=train_path+'/logs'
pathlib.Path(logs_path).mkdir(parents=True, exist_ok=True)

plots_path=train_path+'/plots'
pathlib.Path(plots_path).mkdir(parents=True, exist_ok=True)


# In[7]:


#Objects of Measurement System, Assembly System, Get Inference Data
print('Initializing the Assembly System and Measurement System....')
measurement_system=HexagonWlsScanner(data_type,application,system_noise,part_type,data_format)
vrm_system=VRMSimulationModel(assembly_type,assembly_kccs,assembly_kpis,part_name,part_type,voxel_dim,voxel_channels,point_dim,aritifical_noise)
get_data=GetTrainData()

kcc_sublist=cftrain.encode_decode_params['kcc_sublist']
output_heads=cftrain.encode_decode_params['output_heads']
encode_decode_multi_output_construct=config.encode_decode_multi_output_construct

if(output_heads==len(encode_decode_multi_output_construct)):
	print("Valid Output Stages and heads")
else:
	print("Inconsistent model setting")

print("KCC sub-listing: ",kcc_sublist)

#Check for KCC sub-listing
if(kcc_sublist!=0):
	output_dimension=len(kcc_sublist)
else:
	output_dimension=assembly_kccs

print("Process Parameter Dimension: ",output_dimension)

input_size=(voxel_dim,voxel_dim,voxel_dim,voxel_channels)

model_depth=cftrain.encode_decode_params['model_depth']
inital_filter_dim=cftrain.encode_decode_params['inital_filter_dim']


# In[8]:


#importing file names for model input
input_file_names_x=config.encode_decode_construct['input_data_files_x']
input_file_names_y=config.encode_decode_construct['input_data_files_y']
input_file_names_z=config.encode_decode_construct['input_data_files_z']

input_dataset=[]
input_dataset.append(get_data.data_import(input_file_names_x,data_folder))
input_dataset.append(get_data.data_import(input_file_names_y,data_folder))
input_dataset.append(get_data.data_import(input_file_names_z,data_folder))

kcc_dataset=get_data.data_import(kcc_files,kcc_folder)

if(kcc_sublist!=0):
	print("Sub-setting Process Parameters: ",kcc_sublist)
	kcc_dataset=kcc_dataset.iloc[:,kcc_sublist]
	test_kcc_dataset=test_kcc_dataset[:,kcc_sublist]
else:
	print("Using all Process Parameter")


# In[9]:


#Creating Feature Matrix
pp_dim=0
feature_dim=voxel_channels+pp_dim
feature_matrix=np.zeros((len(input_dataset[0]),point_dim,feature_dim))

#print(feature_matrix.shape)
for i in range(voxel_channels):
    feature_matrix[:,:,i]=input_dataset[i].values[:,0:point_dim]

print(feature_matrix.shape)


# In[10]:


#Creating feature matrix for one graph
def create_node_df(feature_matrix):
    node_data = pd.DataFrame(
    {"x_dev": feature_matrix[:,0], "y_dev": feature_matrix[:,1],"z_dev": feature_matrix[:,2]})
    
    return node_data

#node_df=create_node_df(feature_matrix[0,:,:])
#node_df


# In[11]:


#create Stellar Graph Instance
from stellargraph import StellarGraph

graphs_x=[]

for i in tqdm(range(len(feature_matrix))):
    #edges are constant
    #Node features are to be extracted
    node_df=create_node_df(feature_matrix[i,:,:])
    graph_mesh = StellarGraph(node_df,edges)
    
    graphs_x.append(graph_mesh)


# In[12]:


graph_labels=kcc_dataset


# In[13]:


#Add code for scaling all outputs


# In[14]:


G = nx.Graph()
for i in range(len(sparsecols)):
    G.add_edge(sparserows[i], sparsecols[i])


# In[15]:


#Plotting Graph for Testing
from matplotlib import pylab
import networkx as nx

def save_graph(graph,file_name):
    #initialze Figure
    plt.figure(num=None, figsize=(20, 20), dpi=80)
    plt.axis('off')
    fig = plt.figure(1)
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph,pos)
    nx.draw_networkx_edges(graph,pos)
    nx.draw_networkx_labels(graph,pos)

    cut = 1.00
    xmax = cut * max(xx for xx, yy in pos.values())
    ymax = cut * max(yy for xx, yy in pos.values())
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)

    plt.savefig(file_name,bbox_inches="tight")
    pylab.close()
    del fig

#Assuming that the graph g has nodes and edges entered
#save_graph(G,"cross_member.pdf")


# In[16]:


import stellargraph as sg
from sklearn import model_selection
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import GCNSupervisedGraphClassification
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping




#Create Deep Learning Model
def create_graph_classification_model(generator):
    gc_model = GCNSupervisedGraphClassification(
        layer_sizes=[64, 64],
        activations=["relu", "relu"],
        generator=generator,
        dropout=0.1,
    )
    x_inp, x_out = gc_model.in_out_tensors()
    predictions = Dense(units=64, activation="relu")(x_out)
    predictions = Dense(units=64, activation="relu")(predictions)
    predictions = Dense(units=12, activation="linear")(predictions)

    # Let's create the Keras model and prepare it for training
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(), loss="mse", metrics=["mae"])

    return model

#Create Deep Learning Model
def create_deep_graph_model(generator):
    
    from stellargraph.layer import DeepGraphCNN
    from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
    
    k = 35  # the number of rows for the output tensor
    layer_sizes = [32, 32, 32, 1]

    dgcnn_model = DeepGraphCNN(
        layer_sizes=layer_sizes,
        activations=["tanh", "tanh", "tanh", "tanh"],
        k=k,
        bias=False,
        generator=generator,
    )
    
    x_inp, x_out = dgcnn_model.in_out_tensors()

    x_out = Conv1D(filters=16, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)
    x_out = MaxPool1D(pool_size=2)(x_out)

    x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)

    x_out = Flatten()(x_out)

    x_out = Dense(units=128, activation="relu")(x_out)
    x_out = Dropout(rate=0.2)(x_out)

    predictions = Dense(units=12, activation="linear")(x_out)

    # Let's create the Keras model and prepare it for training
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(lr=0.0001), loss="mse", metrics=["mae"])

    return model


#Split Dataset
train_graphs, test_graphs = model_selection.train_test_split(graph_labels, train_size=0.8, test_size=None)

#Create generators
gen = PaddedGraphGenerator(graphs=graphs)

train_gen = gen.flow(
    list(train_graphs.index - 1),
    targets=train_graphs.values,
    batch_size=50,
    symmetric_normalization=False,
)

test_gen = gen.flow(
    list(test_graphs.index - 1),
    targets=test_graphs.values,
    batch_size=1,
    symmetric_normalization=False,
)

epochs = 200  # maximum number of training epochs
#batch_size=10

checkpointer = tf.keras.callbacks.ModelCheckpoint(model_path, verbose=1, save_best_only='loss',save_weights_only=True,restore_best_weights=True)

history = model.fit(train_gen, epochs=epochs, ,verbose=1, validation_data=test_gen, shuffle=True,callbacks=[checkpointer])

print("Training Complete!")





