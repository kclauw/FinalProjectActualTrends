from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from django.shortcuts import render_to_response
from django.shortcuts import render
from .routing.util import *
import pandas as pd
import os
from .routing.graph import Graph
from .routing.route import path
import re
import json


#BUILD GRAPH
graph = Graph()
dir_path = os.path.dirname(os.path.realpath(__file__))

#Read data
all_data = pd.read_csv(dir_path + '/routing/data/processed_data.csv', encoding='cp1252',sep=",")


df_nodes = pd.read_csv(dir_path + '/routing/data/nodes_dataset.csv', encoding='cp1252',sep=",")
df_edges = pd.read_csv(dir_path + '/routing/data/edges_dataset.csv', encoding='cp1252',sep=",")

#Build Nodes
for index, row in df_nodes.iterrows():
    node = row['node']
    x = round(row['x'],5)
    y = round(row['y'],5)
    if(x == 50.82221):
      print("test",x,y)
    pm = row['pm']
    graph.add_node(int(node),x,y,pm)

#Build Edges
for index, row in df_edges.iterrows():
    source = row['source']
    target = row['target']
    distance= row['distance']
    duration = row['duration']
    pm = row['pm']
    n = graph.get_node(int(target))
    graph.add_edge(source, int(target), distance, duration,n.pollution)



def dijkstra(graph, initial,target):
  visited = {initial: 0}
  path = {}

  nodes = set(graph.nodes)

  while nodes and min_node != target:
      min_node = None
      for node in nodes:
          if node in visited:

              if min_node is None:
                  min_node = node
              elif visited[node] < visited[min_node]:
                  min_node = node
      print(node)
      if min_node is None:
            break

      print(min_node)
      nodes.remove(min_node)
      current_weight = visited[min_node]

      for edge in graph.edges[min_node]:
          weight = current_weight + graph.distances[(min_node, edge)]
          if edge not in visited or weight < visited[edge]:
              visited[edge] = weight
              path[edge] = min_node


  return visited, path

def multi_objective_dijkstra(graph, initial,destination):
  #Visited nodes and distances so far
  visited = {initial:(0,0)}
  paretoFront = {(0,0) : (0,0)}
  T = []

  path = {}

  #Initialize label of initial node



  originLabel = [[0,0],None,None]

  #graph.get_node(initial).add_temp_label(originLabel)
  #graph.get_node(initial).add_perm_label(originLabel)


  tempLabels = {initial : originLabel}
  visited = {initial: 0}
  #Lexicographical order corresponds to the lowest first objectives
  lexicographicalOrder = {initial : 0}

  it=0
  visited = {initial : 0}
  currentNode = graph.get_node(initial)
  currentNode.add_perm_label(originLabel)

  ALL_ROUTES = []
  while tempLabels:


    #Find the lowest node according to a lexicographical order in the temporary set
    source = min(lexicographicalOrder, key=lexicographicalOrder.get)

    currentLabel = tempLabels[source]
    currentNode = graph.get_node(source)


    #Move label from temporal to perm and remove = make final !
    currentNode.add_perm_label(currentLabel)
    h = 0
    if len(currentNode.permLabels) > 0 :
      h = currentNode.permLabels.index(currentLabel)
    del lexicographicalOrder[source]
    del tempLabels[source]

    #if source == destination:
    #  break

    #Mark all the neighbors of the currentNode
    for neighbor in graph.edges[source]:


      neighborNode = graph.get_node(neighbor)

      #Get transition rewards
      distance,duration, pollution = graph.get_rewards(source, neighbor)

      sourceDistance, sourcePollution = currentNode.permLabels[h][0]
      newReward = [sourceDistance+distance,sourcePollution+pollution]
      newLabel = [newReward,source,h]
      dominatedNodes = []


      #Determine if the solution is optimal by the temporary archive

      dominated2 = False


      #Determine dominance in permant archive of node
      for label in neighborNode.permLabels:
        target = neighborNode.permLabels[0]
        dominated2 = dominates(label[0],newReward)
        #print("neighbor",neighbor,sourceDistance,sourcePollution,"new_reward",newReward,"neighbors",neighborNode.permLabels,dominated2)
        if dominated2:
          break

      if not dominated2:

        ALL_ROUTES.append([source,neighbor])

        #Store the label of vertex as temporary
        tempLabels[neighbor] = newLabel
        lexicographicalOrder[neighbor] = newLabel[0][0]
        neighborNode.add_perm_label(newLabel)

        #Delete all the temporary labels dominated by label
        #for dom in dominatedNodes:
          #del lexicographicalOrder[dom]
          #del tempLabels[dom]

    it += 1


  return ALL_ROUTES
#we are using 2 types of routes because there is a chance the google API is running out of requests.
#This is far from clean code (this should probably be fixed inside the route file)...
def prepare_route(graph,route_nodes):
  route=[]
  route2=[]
  n = len(route_nodes)
  for i in range(n):
    current_node = graph.get_node(route_nodes[i])
    route.append([current_node.x,current_node.y])
    if i < (n - 1) :
        next_node =  graph.get_node(route_nodes[i+1])
        origins = tuple([current_node.x,current_node.y])
        destinations = tuple([next_node.x,next_node.y])
        #THIS PART ADDS THE ROUTES ACCORDING TO THE GOOGLE API
        temp = path(origins,destinations)
        if temp != None :
            for r in temp:
                route.append([r[0],r[1]])
                #print("Temp ",[r[0],r[1]])

        route2.append(temp)
  return route

def backpropagateroutes(graph, initial,final):
  routes = []
  initial_node = graph.get_node(final)

  for current_node in initial_node.permLabels:
    a=[initial_node.nr]
    previous_node = current_node[1]
    h = current_node[2]
    print("initial",initial_node.nr)
    #print("current",previous_node)
    a.append(previous_node)
    while previous_node != initial and previous_node != None :
      neighbor = graph.get_node(previous_node)
      previous_node = neighbor.permLabels[h][1]
      #print("next",previous_node)
      h = neighbor.permLabels[h][2]
      a.append(previous_node)
    #print(a[::-1])

    if previous_node != None:
      routes.append(a[::-1])
      #break
    print("\n")
  return routes


def index(request):
    return render(request, 'cleanbike/index.html')


@csrf_exempt
def route(request):

    #Extract initial coordintes
    x_1, y_1 = re.sub('[!@#$A-Za-z()]', '', request.POST['firstbox']).split(',')
    x_2, y_2 = re.sub('[!@#$A-Za-z()]', '', request.POST['secondbox']).split(',')

    #Find nodes of initial starting coordintes
    source = graph.find_node(x_1,y_1)
    target = graph.find_node(x_2,y_2)


    ALL_ROUTES = multi_objective_dijkstra(graph, source, target)
    routes = backpropagateroutes(graph, source, target)

    #Find the coordinates of each route
    coordinate_routes = []
    for r in routes:
      coordinate_routes.append(prepare_route(graph,r))


    return render(request, 'cleanbike/routes.html',{'content' : coordinate_routes})
