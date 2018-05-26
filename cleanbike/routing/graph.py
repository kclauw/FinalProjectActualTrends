from collections import defaultdict

import pandas as pd
import numpy as np

class Node:
   def __init__(self,nr,x,y,pollution,initial=None):
    self.nr = nr
    self.tempLabels = []
    self.permLabels = []
    self.previousReward = initial
    self.x = x
    self.y = y
    self.pollution = pollution



   def add_temp_label(self,label):
    """
    Add a label to the temporary list of vectorial rewards and return the position
    """
    self.tempLabels.append(label)
    return len(self.tempLabels) - 1


   def add_perm_label(self,label):
    """
    Add a label to the permanent list of vectorial rewards and return the position
    """
    if label not in self.permLabels:
        self.permLabels.append(label)








    

class Graph:
  def __init__(self):
    self.nodes = []
    self.edges = defaultdict(list)
    self.objectives = {}
    self.paretoFront = {}

  def get_node(self,nr):
    return self.nodes[nr]

  def add_nodes(self, value):
      for i in value:
          self.nodes.append(Node(i))

  def add_node(self,value,x,y,pollution):
      self.nodes.append(Node(value,x,y,pollution))
  #TODO : Nodes should be stored in a dictionary according to the geocoordinates
  def find_node(self,x,y):
    for node in self.nodes:
      if abs(float(node.x) - float(x)) <= 0.00001:
        return node.nr
    return -1


  def add_edge(self, from_node, to_node, distance,duration,pollution):
    
    #self.edges[to_node].append(from_node) # dict to neighbour nodes
    #if not (to_node,from_node) in self.objectives and not (from_node,to_node) in self.objectives:
    self.edges[from_node].append(to_node)
    self.objectives[(from_node, to_node)] = [distance,duration,pollution] # dict for distance
    #self.objectives[(to_node, from_node)] = [distance,pollution]

  def add_edge_pareto(self, from_node, to_node, distance,pollution):
    self.edges[from_node].append(to_node)
    self.edges[to_node].append(from_node) # dict to neighbour nodes
    self.objectives[(from_node, to_node)] = [distance,duration,pollution] # dict for distance
    #self.objectives[(to_node, from_node)] = [distance,pollution]

  def get_rewards(self,from_node,to_node):
    return self.objectives[(from_node, to_node)]


  def get_distance(self,from_node,to_node):
    return self.objectives[(from_node, to_node)][0] 

  def get_pollution(self,from_node,to_node):
    return self.objectives[(from_node, to_node)][1] 

  def isParetoOptimal(self,distance,pollution):
    for edge in ParetoFront:
        print(edge)



  def merge(self,archive1,archive2):
    print("arch1",archive1)
    print("arch2",archive2)
    undominated = []
    for sol1 in archive1:
        dom = False
        for sol2 in archive2:
            
            dom = paretoDominated(sol1,sol2)
            print("sol",sol1,"sol2",sol2,"dom",dom)
            if dom:
                break

    return archive1

#Read data
#nodes = pd.read_csv('./data/nodes_dataset.csv', encoding='cp1252',sep=",")

#for i in range(len(nodes)):
  #x=nodes['x'][i]
  #y=nodes['y'][i]
  #n=nodes['node'][i]
  #pm=nodes['pm'][i]

    