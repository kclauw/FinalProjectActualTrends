import pandas as pd
from scipy import spatial
from util import to_Cartesian,geo_to_cartesian,distToKM,kmToDIST
import geopandas as gpd
import numpy as np
from geopandas.tools import sjoin
import folium
from folium.plugins import MarkerCluster
import shapely
from shapely.geometry import Point
import unicodedata
import pysal as ps
from shapely import geometry
import sys
from folium import plugins
import string
import folium
from shapely import wkt, geometry
import json
from pprint import pprint
from openrouteservice import client, places
from openrouteservice import distance_matrix
from geopandas import GeoDataFrame
from shapely.geometry import Point
from openrouteservice import geocoding
from algorithms import multi_objective_dijkstra
from folium.plugins import MarkerCluster
from folium import IFrame

import re
from route import name,path,distance_duration,distanceMatrix

#Read data
all_data = pd.read_csv('./data/processed_data.csv', encoding='cp1252',sep=",")

#Convert geo-coordinates to cartesian
all_data['x'], all_data['y'], all_data['z'] = geo_to_cartesian(all_data['lat_wgs84'],all_data['long_wgs84'])

#Save to file
all_data.to_csv('./data/final_dataset.csv')

baseStationCoordinates = all_data[['lat_wgs84', 'long_wgs84']].as_matrix()


ulb = [50.813724000000001,4.3842050000000006]

north = [50.860652000000002, 4.3581160000000008]
addresses=[]

#TODO : Can use a reverse geo coder based on google to retrieve the station names (when working with new data)
#Reverse geo code the address name


#locationlist = zip(all_data['long_wgs84'],all_data['lat_wgs84'])
locationlist = [[all_data['long_wgs84'][i],all_data['lat_wgs84'][i]] for i in range(len(all_data)) ]

#Create SF basemap specifying map center, zoom level, and using the default OpenStreetMap tiles
map = folium.Map(location=[50.821658, 4.394886], zoom_start=15)
points=[]
#Folium bugs when using numpy -> use a list of lists as an alternative
baseStationCoordinates = [[all_data['lat_wgs84'][i],all_data['long_wgs84'][i],all_data['pm2.5'][i]] for i in range(len(all_data['latitude']))]
locationlist = [[all_data['lat_wgs84'][i],all_data['long_wgs84'][i]] for i in range(len(all_data))]


#Create empty lists to contain the point coordinates and the point pop-up information
coords, popups = [], [] 

map.save('routing.html')


for point in range(0, len(all_data)):
    #t='{0},({1}): {2}'.format(area_of_circle, points, estimate(radius, points))
    address=all_data['address'][point].split('-')[0].capitalize()
    x=all_data['latitude'][point]
    y=all_data['longitude'][point]
    #print(address,x,y)
    pm=all_data['pm2.5'][point]
    text = u"<strong>{0}</strong><br>Lat: {1:.3f}<br>Long: {2:.3f}".format(address, x, y)
    popup = folium.Popup(text, parse_html=True)
    coords.append([x,y])
    popups.append(IFrame(text, width = 300, height = 100))




#Create a Folium feature group for this layer, since we will be displaying multiple layers
pt_lyr = folium.FeatureGroup(name = 'pt_lyr')

#Add the clustered points of crime locations and popups to this layer
pt_lyr.add_children(MarkerCluster(locations = coords, popups =popups))

#Add this point layer to the map object
map.add_children(pt_lyr)  



#Add the pollution of each base station to the map
max_amount = float(all_data['pm2.5'].max())
map.add_child(plugins.HeatMap(baseStationCoordinates,
                                 radius=15,
                                 min_opacity=0.2,
                                 max_val=max_amount,
                                 blur=15, 
                                 max_zoom=10))

map.save('routing.html')



#CREATE GRAPH BETWEEN STATION
from graph import Graph
graph = Graph()
#convert the 30 km to cartesian coordinates distance
#dist = kmToDIST(1)

# create the KD-tree using the 3D cartesian coordinates
coordinates = list(zip(all_data['x'], all_data['y'], all_data['z']))
tree = spatial.cKDTree(coordinates)
# convert the 30 km to cartesian coordinates distance

# get all the points within 30 km from the reference point
#dist = kmToDIST(3)
#ix = tree.query_ball_point((x_ref, y_ref, z_ref), dist)
# get all the points within 30 km from the reference point
#ix = tree.query_ball_point((x_ref, y_ref, z_ref), dist)
#CREATE DISTANCE MATRIX

#CREATE GRAPH OF NODES WITH DISTANCE AND POLLUTION AS WEIGHTS
all_data_array = all_data.as_matrix()
coord = all_data[['latitude','longitude']].as_matrix()
#coord = all_data[['long_wgs84','lat_wgs84']].as_matrix()
#graph.add_nodes([i for i in range(len(all_data))])
edges_dataset=[]
nodes_dataset=[]

df_nodes = pd.read_csv('./data/nodes_dataset20.csv', encoding='cp1252',sep=",")
df_edges = pd.read_csv('./data/edges_dataset20.csv', encoding='cp1252',sep=",")


for source in range(len(df_nodes),len(all_data)):
    #VISUALIZATION ON MAP
    print("source",source)
    address=all_data['address'][source].split('-')[0].capitalize()
    x=all_data['lat_wgs84'][source]
    y=all_data['long_wgs84'][source]
    pm=all_data['pm2.5'][source]
    nodes_dataset.append([source,x,y,pm])
    #name = all_data['name'][source]
    print("Node : ",source,x,y,address)
    x_ref, y_ref, z_ref = to_Cartesian(x,y)
    # get the cartesian distances from the 10 closest points
    dist, ix = tree.query((x_ref, y_ref, z_ref), 20)
    graph.add_node(source,x,y,pm)
    #Find the node that is not yet going to another node 
    print("ix",ix,dist)
    #Add neighbors to graph
    for index in ix:
        if index != source:
            x_target = coord[index][0]
            y_target = coord[index][1]
            distance, duration = distance_duration((x,y),(x_target,y_target))
            address=all_data['address'][index].split('-')[0].capitalize()
            pm_target=all_data['pm2.5'][index]
            #pm_target=all_data['pm2.5'][index]
            #pm_target=all_data.loc[[index]]['pm2.5']
            #print("address",address,"source",source,"target",index,"distance",distance,"pm",pm_target)
            graph.add_edge(source,index,distance,duration,pm_target)
            #folium.PolyLine([[x, y], [x_target, y_target]]).add_to(map)
            #print("target",[x,y], [x_target, y_target])
            edges_dataset.append([source,index,distance,duration,pm_target])
            print([source,index,distance,duration,pm_target])
    #Append data to CSV
    print(edges_dataset)
    temp_nodes = pd.DataFrame(nodes_dataset,index=None,columns=['node','x','y','pm'])
    temp_nodes = df_nodes.append(temp_nodes)
    temp_nodes.to_csv('./data/nodes_dataset20.csv', sep=',',index=None,columns=['node','x','y','pm'])
    temp_edges = pd.DataFrame(edges_dataset,index=None,columns=['source','target','distance','duration','pm'])
    temp_edges = df_edges.append(temp_edges)
    temp_edges.to_csv('./data/edges_dataset20.csv', sep=',',index=None,columns=['source','target','distance','duration','pm'])

""""
        
print('done')
map

map.save('routing_final.html')


ulb = [50.813724000000001,4.3842050000000006]

north = [50.860652000000002, 4.3581160000000008]
etterbeek = [50.822206999999999, 4.3893250000000004]

initial = graph.find_node(north)

final = graph.find_node(etterbeek)


noord = 27


etterbeek = 190
jaargetijden=263
petite_suide=268
cemitiere=109
vub = 39
ulb = 309
print(noord,etterbeek)
#multi_objective_dijkstra(graph,0,5)
print(graph.nodes[etterbeek].coordinates)
etterbeek_coord = tuple(graph.nodes[etterbeek].coordinates)
jaargetijden_coord = tuple(graph.nodes[jaargetijden].coordinates)
petite_suide_coord = tuple(graph.nodes[petite_suide].coordinates)
cemitiere_coord = tuple(graph.nodes[cemitiere].coordinates)
ulb_coord = tuple(graph.nodes[ulb].coordinates)


#FIND THE ROUTES AND ADD THEM TO MAP
folium.PolyLine(path([etterbeek_coord,jaargetijden_coord])).add_to(map)
folium.PolyLine(path([jaargetijden_coord,petite_suide_coord])).add_to(map)
folium.PolyLine(path([petite_suide_coord,ulb_coord])).add_to(map)
folium.PolyLine(path([etterbeek_coord,cemitiere_coord])).add_to(map)
folium.PolyLine(path([cemitiere_coord,ulb_coord])).add_to(map)
folium.PolyLine(path([etterbeek_coord,petite_suide_coord])).add_to(map)
folium.PolyLine(path([petite_suide_coord,ulb_coord])).add_to(map)
map.save('routes_final.html')  
"""