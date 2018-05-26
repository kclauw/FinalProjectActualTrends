
import random
import heapq
import collections
import functools
import numpy as np
from collections import deque
from util import dominates
from graph import Graph

def is_pareto_efficient(costs):
    """
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)  # Remove dominated points
    return is_efficient

def multi_objective_dijkstra(graph, initial,target):
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
  while tempLabels and it < 2:

    
    #Find the lowest node according to a lexicographical order in the temporary set
    source = min(lexicographicalOrder, key=lexicographicalOrder.get)
  
    print("Node",source)

    currentLabel = tempLabels[source]
    currentNode = graph.get_node(source)
  
    #Move label from temporal to perm and remove = make final ! 
    currentNode.add_perm_label(currentLabel)
    h = 0
    if len(currentNode.permLabels) > 0 :
      h = currentNode.permLabels.index(currentLabel)

    del lexicographicalOrder[source]
    del tempLabels[source]

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
      dominated = False
     
      for label in tempLabels:
        target = tempLabels[label][0]
        #print("neighbor",neighbor,newReward,target,isParetoDominated(newReward,target))
        if dominates(newReward,target) :
          dominatedNodes.append(label)
      
      dominated = len(dominatedNodes) > 0
      #print("neighbor",neighbor,newReward,dominated)
     
        

     
      
      #Determine dominance in permant archive of node
      for label in neighborNode.permLabels:
        target = neighborNode.permLabels[0]
        dominated = dominates(newReward,label[0])
        if dominated:
          break
      
      if not dominated:
        #Store the label of vertex as temporary
        tempLabels[neighbor] = newLabel
        lexicographicalOrder[neighbor] = newLabel[0][0]
        #neighborNode.add_perm_label(newLabel)
        dom = []
        #print("neighbor",neighbor,neighborNode.permLabels)
        #for label in tempLabels:
          #print("label",newReward,tempLabels[label][0],dominates(newReward,tempLabels[label][0]))
          #if dominates(newReward,tempLabels[label][0]):
           # dom.append(label)
        #for domi in dom:
        #  del lexicographicalOrder[domi]
        #  del tempLabels[domi]
        #print("adding",neighbor,newLabel,tempLabels)

      #Delete all the temporary labels dominated by label
      for dom in dominatedNodes:
        del lexicographicalOrder[dom]
        del tempLabels[dom]


    it += 1
    #print("after",tempLabels)
    #Remove current from temporary labels
  #print("\n")
  #for node in range(len(graph.nodes)):
    #print("node",node," ",graph.nodes[node].permLabels)
      #tempLabels[neighborNode.nr] = newLabel
      #lexicographicalOrder[neighborNode.nr] = newReward[0]


      #Determine dominance of solution in contrast to the permanent archive
      #for solution in neighborNode.permLabels:
      #if solution == None:

        
          #lexicographicalOrder


        #isDominated = paretoDominated(newReward,solution[0])
        #print(isDominated)

      #Determine dominance

      #Store as temporary

      #Remove the dominated points
   
 







   



  return visited, path


def dijkstra(graph, initial):
  visited = {initial: 0}
  path = {}

  nodes = set(graph.nodes)

  while nodes: 
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
      print("next",previous_node)
      h = neighbor.permLabels[h][2]
      a.append(previous_node)
    #print(a[::-1])
    
    if previous_node != None:
      routes.append(a[::-1])
      #break
    print("\n")
  return routes

