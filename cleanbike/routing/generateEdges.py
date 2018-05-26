
from util import *
import pandas as pd
import os 
from graph import Graph
import re
import json
from algorithms import multi_objective_dijkstra,backpropagateroutes
import numpy as np
from route import distance_duration
import random
import math
 
def distance_on_unit_sphere(lat1, long1, lat2, long2):
 
    # Convert latitude and longitude to
    # spherical coordinates in radians.
    degrees_to_radians = math.pi/180.0
     
    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians
     
    # theta = longitude
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians
     
    # Compute spherical distance from spherical coordinates.
     
    # For two locations in spherical coordinates
    # (1, theta, phi) and (1, theta', phi')
    # cosine( arc length ) =
    # sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length
     
    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
    math.cos(phi1)*math.cos(phi2))
    arc = math.acos( cos )
     
    # Remember to multiply arc by the radius of the earth
    # in your favorite set of units to get length.
    return arc


from math import sin, cos, sqrt, atan2, radians

def distance_calc(x1,y1,x2,y2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(x1)
    lon1 = radians(y1)
    lat2 = radians(x2)
    lon2 = radians(y2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

#BUILD GRAPH
graph = Graph()
dir_path = os.path.dirname(os.path.realpath(__file__))

#Read data
all_data = pd.read_csv(dir_path + '/data/processed_data.csv', encoding='cp1252',sep=",")


df_nodes = pd.read_csv(dir_path + '/data/nodes_dataset.csv', encoding='cp1252',sep=",")
df_edges = pd.read_csv(dir_path + '/data/edges_dataset.csv', encoding='cp1252',sep=",")

#Build Nodes
for index, row in df_nodes.iterrows():
    node = row['node']
    x = round(row['x'],5)
    y = round(row['y'],5)
    if(x == 50.82221):
      print("test",x,y)
    pm = row['pm']
    graph.add_node(int(node),x,y,pm)



distance_matrix = []
edges_dataset = []
edges_dataset2 = []

empty = pd.read_csv(dir_path + '/data/empty.csv', encoding='cp1252',sep=",",names=['source','target','distance','duration','pm'])

for source_node in graph.nodes:
    x_1 = source_node.x
    y_1 = source_node.y
    temp = []
    for neighbor_node in graph.nodes:
        x_2 = neighbor_node.x
        y_2 = neighbor_node.y
        if x_1 != x_2 and y_1 != y_2:
            temp.append(distance_calc(x_1,y_1,x_2,y_2))
        else:
            temp.append(0.0)
    print("source",source_node.nr)
    np_array = np.array(temp)
    sorted_array = np.argsort(np_array)
    n_shortest_distance = sorted_array[1:10]
    
    for shortest in n_shortest_distance:
        target_node = graph.nodes[shortest]
        #distance, duration = distance_duration((x_1,y_1),(target_node.x,target_node.y))

        #edges_dataset.append([source, int(target_node.nr), distance, duration,target_node.pollution])
        edges_dataset2.append([int(source_node.nr), int(target_node.nr), np_array[shortest], 0,target_node.pollution])

    distance_matrix.append(temp)

    #temp_edges = pd.DataFrame(edges_dataset,index=None,columns=['source','target','distance','duration','pm'])
    #temp_edges = df_edges.append(temp_edges)
    #temp_edges.to_csv('./data/edges_dataset_new.csv', sep=',',index=None,columns=['source','target','distance','duration','pm'])



temp_edges = pd.DataFrame(edges_dataset2,index=None,columns=['source','target','distance','duration','pm'])
temp_edges.to_csv('./data/edges_dataset.csv', sep=',',index=None,columns=['source','target','distance','duration','pm'])


#multi_objective_dijkstra(graph,10,350)
#routes = backpropagateroutes(graph,10,12)
#print(routes)