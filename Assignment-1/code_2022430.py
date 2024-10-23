import numpy as np
import pickle

# General Notes:
# - Update the provided file name (code_<RollNumber>.py) as per the instructions.
# - Do not change the function name, number of parameters or the sequence of parameters.
# - The expected output for each function is a path (list of node names)
# - Ensure that the returned path includes both the start node and the goal node, in the correct order.
# - If no valid path exists between the start and goal nodes, the function should return None.


# Algorithm: Iterative Deepening Search (IDS)

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]
from collections import deque
def bfs_check_path(adj_matrix, start_node, goal_node):
    n = len(adj_matrix)
    visited = set()
    que = deque([start_node])
    visited.add(start_node)
    while que:
        currNode = que.popleft()
        if currNode == goal_node:
            return True
        for i in range(n):
            if adj_matrix[currNode][i] > 0 and i not in visited:
                visited.add(i)
                que.append(i)
    return False
def helpDiameter(v, adj_matrix, n):
    dist = [-1]*n
    que1 = deque([(v, 0)]) 
    dist[v] = 0
    while que1:
        currNode, currDist = que1.popleft()
        for i in range(n):
            if adj_matrix[currNode][i] > 0 and dist[i] == -1:
                dist[i] = currDist + 1
                que1.append((i, currDist + 1))
    return max(dist)

def getDiameter(adj_matrix, start_node):
    n = len(adj_matrix)
    return helpDiameter(start_node, adj_matrix, n)

def bfsForDfs(n, adj_matrix, currNode, goalNode, depth_left, path, visited):
    if currNode == goalNode:
        return path
    if depth_left == 0:
        return None
    visited.add(currNode)
    for i in range(n):
        if adj_matrix[currNode][i] > 0 and i not in visited:
            newPath = bfsForDfs(n, adj_matrix, i, goalNode, depth_left - 1, path + [i], visited)
            if newPath:
                return newPath
    visited.remove(currNode)
    return None

def get_ids_path(adj_matrix, start_node, goal_node):
    if not bfs_check_path(adj_matrix, start_node, goal_node):
        return None    
    n = len(adj_matrix)
    diam = getDiameter(adj_matrix, start_node)
    bfsLimit = int(0.7 * diam)
    for d in range(diam + 1):
        visited = set()
        que = deque([(start_node, 0, [start_node])])
        while que:
            currNode, depth, path = que.popleft()
            if currNode == goal_node:
                return path
            if depth >= bfsLimit:
                newVisited = visited.copy()
                dfsPath = bfsForDfs(n, adj_matrix, currNode, goal_node, d - depth, path, newVisited)
                if dfsPath:
                    return dfsPath
            else:
                for i in range(n):
                    if adj_matrix[currNode][i] > 0 and i not in visited:
                        visited.add(i)
                        que.append((i, depth + 1, path + [i]))
    
    return None

  

# Algorithm: Bi-Directional Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

from collections import deque
def get_bidirectional_search_path(adj_matrix, start_node, goal_node):
  if start_node == goal_node:
    return [start_node]
  n = len(adj_matrix)
  visited1 = set() ; visited2 = set()
  que1 = deque() ; que2 = deque()
  que1.append(start_node) ; que2.append(goal_node)
  visited1.add(start_node) ; visited2.add(goal_node)
  parent1 = {} ; parent2 = {}
  parent1[start_node] = -1 ; parent2[goal_node] = -1
  while len(que1) > 0 and len(que2) > 0 :
    while len(que1) > 0 :  
      curr1 = que1.popleft() 
      for i in range(n):
        if adj_matrix[curr1][i] > 0 and i not in visited1:
          que1.append(i) ; visited1.add(i) ; parent1[i] = curr1
          if i in visited2 :
            path = [] ; temp = i
            while temp != -1:
              path.append(temp)
              temp = parent1[temp]
            path.reverse()
            temp = visited2.intersection(parent1.keys()).pop()
            temp = parent2[temp]
            while temp != -1:
              path.append(temp)
              temp = parent2[temp]
            return path
    while len(que2) > 0 : 
        curr2 = que2.popleft()
        for i in range(n):
          if adj_matrix[i][curr2] > 0 and i not in visited2:
            que2.append(i) ; visited2.add(i) ; parent2[i] = curr2
            if i in visited1:
              path = [] ; temp = i
              while temp != -1:
                path.append(temp)
                temp = parent2[temp]
              path.reverse()
              temp = visited1.intersection(parent2.keys()).pop()
              temp = parent1[temp]
              while temp != -1:
                path.append(temp)
                temp = parent1[temp]
              return path
  return None





#c part 
def allNodesCheckIDSandBDP(adj_matrix):
  with open("C_part.txt", "w") as f:
    for i in range(len(adj_matrix)):
      for j in range(len(adj_matrix)):
        f.write(f'For node {i} to {j} : \n')
        path1 = get_ids_path(adj_matrix, i, j)
        path2 = get_bidirectional_search_path(adj_matrix, i, j)
        f.write(f'IDS Path : {path1}\n')
        f.write(f'Bi-Directional Path : {path2}\n')

import time
import tracemalloc
def allNodesTimeIDSandBDP(adj_matrix):
  # dp = floydwarshall(adj_matrix)
  startTime1 = time.perf_counter()
  for i in range(len(adj_matrix)):
    for j in range(len(adj_matrix)):
      # if dp[i][j] == float('inf'):
      #   continue   
      path1 = get_ids_path(adj_matrix, i, j)  
  endTime1 = time.perf_counter()
  print(f'Time taken for IDS : {endTime1 - startTime1:.6f} seconds')
  
  startTime2 = time.perf_counter()
  for i in range(len(adj_matrix)):
    for j in range(len(adj_matrix)):
      path2 = get_bidirectional_search_path(adj_matrix, i, j) 
  endTime2 = time.perf_counter()
  print(f'Time taken for Bi-Directional : {endTime2 - startTime2:.6f} seconds')
  
IDSMem = [] ; BidirectionalMem = [] ; AstarMem = [] ; BiAstarMem = []
def allNodesMemIDSandBDP(adj_matrix):
  maxPeak1 = 0
  for i in range(len(adj_matrix)):
    for j in range(len(adj_matrix)):
      tracemalloc.start()
      path1 = get_ids_path(adj_matrix, i, j)  
      current1mem , peak1mem = tracemalloc.get_traced_memory()
      IDSMem.append(peak1mem)
      tracemalloc.stop()
      maxPeak1 += peak1mem
  print(f'Total memory used by IDS is: {maxPeak1/1024} KB')
  

  maxPeak2 = 0
  for i in range(len(adj_matrix)):
    for j in range(len(adj_matrix)):
      tracemalloc.start()
      path2 = get_bidirectional_search_path(adj_matrix, i, j) 
      current2mem , peak2mem = tracemalloc.get_traced_memory()
      BidirectionalMem.append(peak2mem)
      tracemalloc.stop()
      maxPeak2 += peak2mem
  print(f'Total memory used by Bidirectional search is: {maxPeak2/1024} KB')
  
  
# Algorithm: A* Search Algorithm

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 28, 10, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 27, 9, 8, 5, 97, 28, 10, 12]

def heuristic(startNode , currNode ,endNode , node_attributes):
  distUW =  ((node_attributes[startNode]['x'] - node_attributes[currNode]['x']) ** 2 + (node_attributes[startNode]['y'] - node_attributes[currNode]['y']) ** 2)**(1/2)
  distWV =  ((node_attributes[endNode]['x'] - node_attributes[currNode]['x']) ** 2 + (node_attributes[endNode]['y'] - node_attributes[currNode]['y']) ** 2)**(1/2)
  return distUW + distWV

import heapq
def get_astar_search_path(adj_matrix, node_attributes, start_node, goal_node):
  n = len(adj_matrix)
  # print(heuristic(1,2,3,node_attributes))
  pq = []
  heapq.heappush(pq, (heuristic(start_node , start_node , goal_node , node_attributes)  , start_node , [start_node]))
  dist = [float('inf')] * n
  dist[start_node] = 0
  visited  = set()
  while len(pq) != 0 :
    currentDist , currNode , currPath = heapq.heappop(pq)
    if currNode == goal_node:
      return currPath
    if currNode not in visited:
      visited.add(currNode)
      for i in range(n):
        if adj_matrix[currNode][i] > 0:
          tempScore = dist[currNode] + adj_matrix[currNode][i]
          if tempScore < dist[i] :
            dist[i] = tempScore
            heapq.heappush(pq , (tempScore + heuristic(start_node , i ,goal_node , node_attributes) , i , currPath + [i]))
            
  return None 


# Algorithm: Bi-Directional Heuristic Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 34, 33, 11, 32, 31, 3, 5, 97, 28, 10, 12]


def getCost(list1 , adj_matrix):
  if list1 == None:
    return 0
  xost = 0 
  for i in range(len(list1) - 1):
    xost += adj_matrix[list1[i]][list1[i+1]]
  return xost
  


def get_bidirectional_heuristic_search_path(adj_matrix, node_attributes, start_node, goal_node):
  if start_node == goal_node:
    return [start_node]
  n = len(adj_matrix)
  pq1 = [] ; pq2 = []
  dist1 = [float('inf')]*n ; dist2 = [float('inf')]*n
  parent1 = {} ; parent2 = {}
  dist1[start_node] = 0 ; dist2[goal_node] = 0
  heapq.heappush(pq1 , (heuristic(start_node , start_node , goal_node ,node_attributes) , start_node))
  heapq.heappush(pq2 , (heuristic(goal_node , goal_node , start_node , node_attributes) , goal_node))
  parent1[start_node] = -1 ; parent2[goal_node] = -1
  visited1 = set() ; visited2 = set()
  # sumVal = 0
  minCost = float('inf') ; minPath = []
  while len(pq1)!=0 and len(pq2)!=0:
    # if len(pq1)!=0 :
      currval1temp , currNode1temp = pq1[0] ; 
      currval2temp , currNode2temp = pq2[0] ; 
      if currval1temp < currval2temp : 
        if len(pq1) != 0 :
          currVal1 , currNode1 = heapq.heappop(pq1)
          visited1.add(currNode1)
          for i in range(n):
            if adj_matrix[currNode1][i] >0 and i not in visited1:
              temp1 = dist1[currNode1] + adj_matrix[currNode1][i]
              if temp1 < dist1[i]:
                dist1[i] = temp1
                heapq.heappush(pq1 , (temp1 + heuristic(start_node , i , goal_node , node_attributes) , i)) 
                parent1[i] = currNode1    
              
              if i in visited2 :
                res = []
                x = i
                while x!= -1 : 
                  res.append(x)
                  x = parent1.get(x , -1)
                res.reverse()
                x = parent2.get(i , -1)
                while x != -1 :
                  res.append(x)
                  x = parent2.get(x , -1)
                costPath = getCost(res , adj_matrix)
                if costPath < minCost:
                  # print(costPath)
                  minPath = res ; minCost =  costPath
              
    # if len(pq2) != 0:
      else:
        if len(pq2) != 0 : 
          currVal2 , currNode2 = heapq.heappop(pq2)
          visited2.add(currNode2)
          for i in range(n):
            if adj_matrix[i][currNode2] > 0 and i not in visited2 :
              temp2 = dist2[currNode2] + adj_matrix[i][currNode2]
              if temp2 < dist2[i]:
                dist2[i] = temp2
                heapq.heappush(pq2 , (temp2 + heuristic(goal_node , i , start_node , node_attributes) , i))
                parent2[i] = currNode2
              if i in visited1 :
                res = []
                x = i
                while x!= -1:
                  res.append(x)
                  x = parent2.get(x , -1)
                res.reverse()
                x = parent1.get(i , -1)
                while x != -1:
                  res.append(x)
                  x = parent1.get(x , -1)
                res = res[::-1]
                costPath = getCost(res , adj_matrix)
                if costPath < minCost:
                  # print(costPath)
                  minPath = res ; minCost =  costPath
  if minPath != []:
    return minPath
  return None          
      
#e part
def allNodesCheckASPandBSH(adj_matrix , node_attributes):
  with open("E_part.txt", "w") as f:  
    for i in range(len(adj_matrix)):
      for j in range(len(adj_matrix)):
        f.write(f'For node {i} to {j} : \n')
        path1 = get_astar_search_path(adj_matrix,node_attributes ,i, j)
        path2 = get_bidirectional_heuristic_search_path(adj_matrix,node_attributes, i, j)
        f.write(f'A* Path : {path1}\n')
        f.write(f'Bi-Directional A* Path : {path2}\n')

import time
import tracemalloc
def allNodesTimeASPandBSH(adj_matrix , node_attributes):
  startTime1 = time.perf_counter()
  for i in range(len(adj_matrix)):
    for j in range(len(adj_matrix)):
      path1 = get_astar_search_path(adj_matrix,node_attributes ,  i, j)  
  endTime1 = time.perf_counter()
  print(f'Time taken for A* : {endTime1 - startTime1:.6f} seconds')
  
  startTime2 = time.perf_counter()
  for i in range(len(adj_matrix)):
    for j in range(len(adj_matrix)):
      path2 = get_bidirectional_heuristic_search_path(adj_matrix,node_attributes,  i, j) 
  endTime2 = time.perf_counter()
  print(f'Time taken for Bi-Directional A* : {endTime2 - startTime2:.6f} seconds')
  
  
def allNodesMemASPandBSH(adj_matrix , node_attributes):
  maxPeak1 = 0
  for i in range(len(adj_matrix)):
    for j in range(len(adj_matrix)):
      tracemalloc.start()
      path1 = get_astar_search_path(adj_matrix, node_attributes,i, j)  
      current1mem , peak1mem = tracemalloc.get_traced_memory()
      AstarMem.append(peak1mem)
      tracemalloc.stop()
      maxPeak1 += peak1mem   
  print(f'Total memory used by A* is: {maxPeak1/1024} KB')
  

  maxPeak2 = 0
  for i in range(len(adj_matrix)):
    for j in range(len(adj_matrix)):
      tracemalloc.start()
      path2 = get_bidirectional_heuristic_search_path(adj_matrix, node_attributes , i, j) 
      current2mem , peak2mem = tracemalloc.get_traced_memory()
      BiAstarMem.append(peak2mem)
      maxPeak2 += peak2mem
      tracemalloc.stop()
  print(f'Total memory used by Bidirectional A* is: {maxPeak2/1024} KB')



# Bonus Problem
 
# Input:
# - adj_matrix: A 2D list or numpy array representing the adjacency matrix of the graph.

# Return:
# - A list of tuples where each tuple (u, v) represents an edge between nodes u and v.
#   These are the vulnerable roads whose removal would disconnect parts of the graph.

# Note:
# - The graph is undirected, so if an edge (u, v) is vulnerable, then (v, u) should not be repeated in the output list.
# - If the input graph has no vulnerable roads, return an empty list [].
# def isedge(node , visited , parent , ):

def checkEdge(v , parent , cnt , adj_matrix , cntIncreasing , backedge , res , n):
  cnt += 1 
  backedge[v] = cnt ;  cntIncreasing[v] = cnt 
  for i in range(n):
    if adj_matrix[v][i] == 0:
      continue
    if i == parent :
      continue
    if cntIncreasing[i] == -1:
      checkEdge(i , v , cnt , adj_matrix , cntIncreasing , backedge , res , n)
      
      if backedge[i] > cntIncreasing[v] :
        res.append((v , i))
      backedge[v] = min(backedge[v] , backedge[i])
    else:
      backedge[v] = min(backedge[v] , cntIncreasing[i])
  
  


def bonus_problem(adj_matrix):
  n = len(adj_matrix)
  adj_matrixUndirected = [[0 for _ in range(n)] for _ in range(n)]
  for i in range(n):
    for j in range(n):
      if adj_matrix[i][j] > 0 or adj_matrix[j][i] > 0:
        adj_matrixUndirected[i][j] = 1; adj_matrixUndirected[j][i] = 1
  
  cnt = 0
  res = [] ; cntIncreasing = [-1]*n ; backedge = [-1]*n
  for i in range(n):
    if cntIncreasing[i] == -1 : 
      checkEdge(i , -1 , cnt , adj_matrixUndirected , cntIncreasing , backedge , res , n)
      
  # print(len(res))
  return res

adj_matrix = np.load('IIIT_Delhi.npy')
with open('IIIT_Delhi.pkl', 'rb') as f:
  node_attributes = pickle.load(f)
# allNodesCheckIDSandBDP(adj_matrix) ; 
# allNodesTimeIDSandBDP(adj_matrix) ; 
# allNodesMemIDSandBDP(adj_matrix)
# allNodesCheckASPandBSH(adj_matrix , node_attributes) ;
# allNodesTimeASPandBSH(adj_matrix , node_attributes) ;
# allNodesMemASPandBSH(adj_matrix , node_attributes)


# print("from book implementation(cost):" , getCost(get_bidirectional_heuristic_search_path(adj_matrix , node_attributes , start_node , end_node) , adj_matrix))
# print("from what answer is given:(cost)" , getCost([4 , 34 , 33 ,11 , 32 , 31 , 3 , 5,  97 , 28 , 10 , 12]  , adj_matrix) )


#----------------------------------------------------------------------------------------------------------------------------

# (f) part

import re
import matplotlib.pyplot as plt
def scatterPlotPathCost():
  with open("C_part.txt" , 'r') as file:
    strFile = file.read()

  pattern = re.compile(r'IDS Path : (.+?)\nBi-Directional Path : (.+?)(?:\n|$)')

  paths = pattern.findall(strFile)
  with open("E_part.txt" , 'r') as file1:
    strFile1 = file1.read()

  pattern1 = re.compile(r'A\* Path : (.+?)\nBi-Directional A\* Path : (.+?)(?:\n|$)')

  paths1 = pattern1.findall(strFile1)

  
  
  def getPath(path):
    return eval(path) if path != 'None' else None

  idsPaths = [getPath(ids) for ids , bi in paths]
  biPaths = [getPath(bi) for ids , bi in paths]
  AstartPaths = [getPath(astar) for astar , biastar in paths1]
  biAstarPaths = [getPath(biastar) for astar , biastar in paths1]
  idsCosts = [getCost(path , adj_matrix) for path in idsPaths]
  biCosts = [getCost(path , adj_matrix) for path in biPaths]
  astarCosts = [getCost(path , adj_matrix) for path in AstartPaths]
  biAstarCosts = [getCost(path , adj_matrix) for path in biAstarPaths]

  plt.figure(figsize=(12, 10))
  plt.subplot(2 , 2 , 1)
  plt.scatter(list(range(len(idsCosts))), idsCosts, color='blue', label='IDS Path Cost', alpha=0.6)
  plt.title('IDS Path Cost')
  plt.xlabel('Node Pair Index')
  plt.ylabel('Path Cost')
  plt.legend()
  plt.subplot(2 , 2 , 2)
  plt.scatter(list(range(len(biCosts))), biCosts, color='green', label='Bi-Directional Path Cost', alpha=0.6)
  plt.title('Bi-Directional Path Cost')
  plt.xlabel('Node Pair Index')
  plt.ylabel('Path Cost')
  plt.legend()
  plt.subplot(2 , 2 , 3)
  plt.scatter(list(range(len(astarCosts))), astarCosts, color='red', label='Astar', alpha=0.6)
  plt.title('Astar Path Cost')
  plt.xlabel('Node Pair Index')
  plt.ylabel('Path Cost')
  plt.legend()
  plt.subplot(2 , 2 , 4)
  plt.scatter(list(range(len(biAstarCosts))), biAstarCosts, color='black', label='Bi-Directional Path Cost', alpha=0.6)
  plt.title('Bi-Directional Astar Path Cost')
  plt.xlabel('Node Pair Index')
  plt.ylabel('Path Cost')
  plt.legend()
  
  plt.tight_layout()
  plt.show()



def TimePlot(adj_matrix):
  
  ids_times = []
  bidirectional_times = []
  astar_times = []
  bidirectional_heuristic_times = []
  
  for i in range(len(adj_matrix)):
    for j in range(len(adj_matrix)):
      startTime1 = time.perf_counter()
      path1 = get_ids_path(adj_matrix, i, j)  
      endTime1 = time.perf_counter()
      ids_times.append(endTime1 - startTime1)
  for i in range(len(adj_matrix)):
    for j in range(len(adj_matrix)):
      startTime2 = time.perf_counter()
      path2 = get_bidirectional_search_path(adj_matrix, i, j) 
      endTime2 = time.perf_counter()
      bidirectional_times.append(endTime2 - startTime2) 
  
  for i in range(len(adj_matrix)):
    for j in range(len(adj_matrix)):
      startTime3 = time.perf_counter()
      path1 = get_astar_search_path(adj_matrix , node_attributes , i , j)
      endTime3 = time.perf_counter()
      astar_times.append(endTime3 - startTime3)
      
  for i in range(len(adj_matrix)):
    for j in range(len(adj_matrix)):
      startTime4 = time.perf_counter()      
      path2 = get_bidirectional_heuristic_search_path(adj_matrix , node_attributes , i , j)
      endTime4 = time.perf_counter()
      bidirectional_heuristic_times.append(endTime4 - startTime4)

  plt.figure(figsize=(12, 10))
  plt.subplot(2 , 2 , 1)
  plt.scatter(list(range(len(ids_times))), ids_times, color='blue', label='IDS time', alpha=0.6)
  plt.title('IDS times')
  plt.xlabel('Node Pair Index')
  plt.ylabel('times')
  plt.legend()
  plt.subplot(2 , 2 , 2)
  plt.scatter(list(range(len(ids_times))), bidirectional_times, color='green', label='Bi-Directional time', alpha=0.6)
  plt.title('Bi-Directional times')
  plt.xlabel('Node Pair Index')
  plt.ylabel('times')
  plt.legend()
  plt.subplot(2 , 2 , 3)
  plt.scatter(list(range(len(ids_times))), astar_times, color='red', label='Astar time', alpha=0.6)
  plt.title('Astar times')
  plt.xlabel('Node Pair Index')
  plt.ylabel('times')
  plt.legend()
  plt.subplot(2 , 2 , 4)
  plt.scatter(list(range(len(ids_times))), bidirectional_heuristic_times, color='black', label='Bi-Directional time', alpha=0.6)
  plt.title('Bi-Directional Astar times')
  plt.xlabel('Node Pair Index')
  plt.ylabel('times')
  plt.legend()
  
  plt.tight_layout()
  plt.show()


def MemoryPlot(adj_matrix):
  # ids_mem = []
  # bidirectional_mem = []
  # astar_mem = []
  # bidirectional_heuristic_mem = []
  # for i in range(len(adj_matrix)):
  #   for j in range(len(adj_matrix)):
  #     tracemalloc.start()
  #     path1 = get_ids_path(adj_matrix, i, j)  
  #     current1mem , peak1mem = tracemalloc.get_traced_memory()
  #     tracemalloc.stop()
  #     ids_mem.append(peak1mem)
  # for i in range(len(adj_matrix)):
  #   for j in range(len(adj_matrix)):
  #     tracemalloc.start()
  #     path2 = get_bidirectional_search_path(adj_matrix, i, j) 
  #     current2mem , peak2mem = tracemalloc.get_traced_memory()
  #     tracemalloc.stop()
  #     bidirectional_mem.append(peak2mem) 
  
  # for i in range(len(adj_matrix)):
  #   for j in range(len(adj_matrix)):
  #     tracemalloc.start()
  #     path1 = get_astar_search_path(adj_matrix , node_attributes , i , j)
  #     current3mem , peak3mem = tracemalloc.get_traced_memory()
  #     tracemalloc.stop()
  #     astar_mem.append(peak3mem)      
  # for i in range(len(adj_matrix)):
  #   for j in range(len(adj_matrix)):
  #     tracemalloc.start()
  #     path2 = get_bidirectional_heuristic_search_path(adj_matrix , node_attributes , i , j)
  #     current4mem , peak4mem = tracemalloc.get_traced_memory()
  #     tracemalloc.stop()
  #     bidirectional_heuristic_mem.append(peak4mem)
      
      
  plt.figure(figsize=(12, 10))
  plt.subplot(2 , 2 , 1)
  plt.scatter(list(range(len(IDSMem))),IDSMem, color='blue', label='IDS Path Memory', alpha=0.6)
  plt.title('IDS Memory')
  plt.xlabel('Node Pair Index')
  plt.ylabel('Memory')
  plt.legend()
  plt.subplot(2 , 2 , 2)
  plt.scatter(list(range(len(IDSMem))), BidirectionalMem, color='green', label='Bi-Directional Memory', alpha=0.6)
  plt.title('Bi-Directional Memory')
  plt.xlabel('Node Pair Index')
  plt.ylabel('Memory')
  plt.legend()
  plt.subplot(2 , 2 , 3)
  plt.scatter(list(range(len(IDSMem))), AstarMem, color='red', label='Astar Memory', alpha=0.6)
  plt.title('Astar Memory')
  plt.xlabel('Node Pair Index')
  plt.ylabel('Memory')
  plt.legend()
  plt.subplot(2 , 2 , 4)
  plt.scatter(list(range(len(IDSMem))), BiAstarMem, color='black', label='Bi-Directional Memory', alpha=0.6)
  plt.title('Bi-Directional Astar Memory')
  plt.xlabel('Node Pair Index')
  plt.ylabel('Memory')
  plt.legend()
  
  plt.tight_layout()
  plt.show()
  
# scatterPlotPathCost()
# TimePlot(adj_matrix)
# MemoryPlot(adj_matrix)
  



if __name__ == "__main__":
  adj_matrix = np.load('IIIT_Delhi.npy')
  with open('IIIT_Delhi.pkl', 'rb') as f:
    node_attributes = pickle.load(f)
  start_node = int(input("Enter the start node: "))
  end_node = int(input("Enter the end node: ")) 
  print(f'Iterative Deepening Search Path: {get_ids_path(adj_matrix,start_node,end_node)}')
  print(f'Bidirectional Search Path: {get_bidirectional_search_path(adj_matrix,start_node,end_node)}')
  print(f'A* Path: {get_astar_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bidirectional Heuristic Search Path: {get_bidirectional_heuristic_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bonus Problem: {bonus_problem(adj_matrix)}')    