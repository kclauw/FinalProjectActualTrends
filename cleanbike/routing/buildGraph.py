import pandas as pd
from util import to_Cartesian,geo_to_cartesian,distToKM,kmToDIST
from scipy import spatial
from .routing.util import *
from .routing.buildGraph import *
#Read data
all_data = pd.read_csv('./data/processed_data.csv', encoding='cp1252',sep=",")


#CREATE GRAPH OF NODES WITH DISTANCE AND POLLUTION AS WEIGHTS
all_data_array = all_data.as_matrix()
coord = all_data[['latitude','longitude']].as_matrix()
#coord = all_data[['long_wgs84','lat_wgs84']].as_matrix()
#graph.add_nodes([i for i in range(len(all_data))])
edges_dataset=[]
nodes_dataset=[]

df_nodes = pd.read_csv('./data/nodes_dataset20.csv', encoding='cp1252',sep=",")
df_edges = pd.read_csv('./data/edges_dataset20.csv', encoding='cp1252',sep=",")

for index, row in df_nodes.iterrows():
    node = int(row['node'])
    x = row['x']
    y = row['y']
    pm = row['pm']
    graph.add_node(node,x,y,pm)

for index, row in df_edges.iterrows():
    source = row['source']
    target = row['target']
    distance= row['distance']
    duration = row['duration']
    pm = row['pm']
    print(source,target)