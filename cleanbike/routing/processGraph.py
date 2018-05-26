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
from folium.plugins import MarkerCluster
from folium import IFrame
from graph import Graph
from algorithms import multi_objective_dijkstra,backpropagateroutes
from route import name,path,distance_duration,distanceMatrix


#Read data
all_data = pd.read_csv('./data/processed_data.csv', encoding='cp1252',sep=",")

#Convert geo-coordinates to cartesian
all_data['x'], all_data['y'], all_data['z'] = geo_to_cartesian(all_data['lat_wgs84'],all_data['long_wgs84'])

#Save to file
all_data.to_csv('./data/final_dataset.csv')

stationArr = all_data[['lat_wgs84', 'long_wgs84']].as_matrix()


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
stationArr = [[all_data['lat_wgs84'][i],all_data['long_wgs84'][i],all_data['pm2.5'][i]] for i in range(len(all_data['latitude']))]
locationlist = [[all_data['lat_wgs84'][i],all_data['long_wgs84'][i]] for i in range(len(all_data))]


#Create empty lists to contain the point coordinates and the point pop-up information
coords, popups = [], [] 

map.save('routing_test123.html')


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



#Add the pollution of each
max_amount = float(all_data['pm2.5'].max())
map.add_child(plugins.HeatMap(stationArr,
                                 radius=15,
                                 min_opacity=0.2,
                                 max_val=max_amount,
                                 blur=15, 
                                 max_zoom=10))

map.save('routing_test123.html')


g = Graph()
#Read data
nodes = pd.read_csv('./data/nodes_dataset.csv', encoding='cp1252',sep=",")
edges = pd.read_csv('./data/edges_dataset.csv', encoding='cp1252',sep=",")

#Add nodes to graph
for i in range(len(nodes)):
  x=nodes['x'][i]
  y=nodes['y'][i]
  n=nodes['node'][i]
  pm=nodes['pm'][i]
  g.add_node(n,x,y,pm)

#Add edges to graph
#Add nodes to graph

for i in range(len(edges)):
  source=edges['source'][i]
  target=edges['target'][i]
  distance=edges['distance'][i]
  duration=edges['duration'][i]
  pm=edges['pm'][i]
  g.add_edge(source,target,distance,duration,pm)

vub = 39
ulb = 309


def create_route(source,target):
  colors = ['red','green','blue','orange','yellow']
  multi_objective_dijkstra(g,0,target)
  routes = backpropagateroutes(g, source,target)
  
  for route in routes:
	  print("route",route)
	  r = []
	  for i in range(1,len(route)):
	    source_nr = route[i-1]
	    target_nr = route[i]
	    source = g.get_node(source_nr)
	    target = g.get_node(target_nr)
	    #p = path(tuple([source.x,source.y]),tuple([target.x,target.y]))
	    folium.PolyLine([source,target],colors=colors[0]).add_to(map)
	   

  #folium.PolyLine(path([route[i-1],route])).add_to(map)
#create_route(vub, ulb)
map.save('TEST.html')
