graph1 = {
'a': {'b': 1, 'c': 4},
'b': {'c': 1, 'd': 2, 'e': 2},
'c': {},
'd': {'b': 1, 'c': 5},
'e': {'d': 3}
}
graph2 = {
'A': {'B': 5, 'D': 10},
'B': {'A': 5, 'D': 6, 'C': 2},
'D': {'A': 10,'B': 6,'C':2, 'E': 5 },
'C': {'B': 2, 'E': 4, 'F': 8,'D':2},
'E': {'D': 5, 'C': 4, 'F': 3},
'F': {'E': 3, 'C': 8}
}
graph3= {
'B': {'A': 5, 'D': 1, 'G': 2},
'A': {'B': 5, 'D': 3, 'E': 12, 'F':5},
'D': {'B': 1, 'G': 1, 'E': 1, 'A': 3},
'G': {'B': 2, 'D': 1, 'C': 2},
'C': {'G': 2, 'E': 1, 'F': 16},
'E': {'A': 12, 'D': 1, 'C': 1, 'F': 2},
'F': {'A': 5, 'E': 2, 'C': 16}
}
graph4= {
'1':{'5':2,'4':6,'3':3,'2':9},
'2':{'1':9,'3':2,'7':10,'6':5},
'3':{'1':3,'2':2,'4':3,'7':5,'8':11,'9':7},
'4':{'1':6,'3':3,'5':7,'9':1,'10':2,'11':7},
'5':{'1':2,'4':7,'11':5},
'6':{'2':5,'7':7,'12':9},
'7':{'2':10,'3':5,'6':7,'8':3,'9':12,'12':10,'13':4},
'8':{'3':11,'9':4,'7':3,'12':8,'13':3,'14':12},
'9':{'3':7,'4':1,'7':12,'8':4,'10':14,'14':10,'15':11},
'10':{'4':2,'9':14,'11':6,'15':12,'16':13},
'11':{'4':7,'5':5,'10':6,'16':17},
'12':{'6':9,'7':10,'8':8,'13':1,'17':2},
'13':{'7':4,'8':3,'12':1,'14':15,'18':9},
'14':{'8':12,'9':10,'13':15,'15':6,'18':10,'20':8},
'15':{'9':11,'10':12,'14':6,'16':2,'19':3,'20':13},
'16':{'10':13,'11':17,'15':2,'19':5},
'17':{'12':2,'18':3},
'18':{'13':9,'14':10,'17':3,'20':7},
'19':{'10':2,'15':3,'16':5,'20':3},
'20':{'14':8,'15':13,'18':7,'19':3}
}


#Implementation of Distance vector routing
import pdb
import time

from prettytable.prettytable import PrettyTable
#Find time to run the code
start_time = time.time()
# Step 1: For each node prepare the destination and predecessor
def initialize(graph, source):
 d = {} # Stands for destination
 p = {} # Stands for next hop
 for node in graph:
  d [node] = float ('Inf') # initialization to infinity
  p [node] = None
 d [source] = 0 # For the source we know how to reach
 return d, p
 
#Step 2: Run the Bellman-Ford algorithm
def bellman_ford (graph, source):
 countloop=0
 d, p = initialize (graph, source)
 for i in range(len(graph)-1): #Run this until is converges
  for u in graph:
   for v in graph[u]: #For each neighbor of u
    if d[v] > d[u] + graph[u][v]:
    # Record this lower distance
      d[v] = d[u] + graph[u][v]
      p[v] = u
    countloop=countloop+1
    
# Step 3: check for negative-weight cycles
 for u in graph:
  for v in graph[u]:
   assert d[v] <= d[u] + graph[u][v]
 return d,p,countloop

"""
The DIJKSTRA algorithm for Link state routing
"""

#Find time to run the code
import time
start_time = time.time()
def dijkstra(graph,source):
 #Find all nodes as dictionary keys
 nodes=graph.keys();
 #let unvisted is a dictionary that sores unvisited nodes and cost from   source
 unvisited = {node: float("inf") for node in nodes}
 #Let visited is a dictionary contains the visited nodes
 visited = {}
 current =source
 currentCost = 0
 unvisited[current] = currentCost
 #To find the next hop, p is used
 p = {} # Stands for next hop
 for node in graph:
  p[node] = None
 countloop=0
 #continue the loop until unvisited is empty
 while True:
  #Find the neighbours and cost from current nodes
  for neighbour, cost in graph[current].items():
   countloop=countloop+1
   if neighbour not in unvisited: continue
   #Find the new cost
   newCost = currentCost + cost
   #Update cost if current cost> newcost
   if unvisited[neighbour] > newCost:
     unvisited[neighbour] = newCost
     p[neighbour]=current
  visited[current] = currentCost

  #delete the current node from unvisited list
  del unvisited[current]
  if not unvisited: break
  #sort the unvisited list to find the node with min cost
  candidates = [node for node in unvisited.items() if node[1]]
  current, currentCost = sorted(candidates, key = lambda x: x[1])[0]
 return (visited,p,countloop)


#MAIN MODULE

#Call the bellman_ford function with source node as A for graph 3


from prettytable import PrettyTable

d, p,countloop = bellman_ford(graph4, '1')
#print the cost and next hop

print(1* "\n")
head="!!!!!!!!!!!! ROUTING TABLE AT SOURCE NODE A !!!!!!!!!!!!!"
headt=head.center(155)
print(headt)
print(1* "\n")


table4=PrettyTable(["d","c","n"])


for i,j,k in zip(d.keys(),d.values(),p.values()):
 table4.add_row([i,j,k])
 table4.add_row(["","",""])
print(table4)
print('')
print("( Distance vector routing:Time to run = %s seconds)" % (time.time() - start_time))
print("Distance vector routing:No of loop iterations:= ",countloop)
#Call the DIJKSTRA function with source node as A for graph3
d,p,countloop=dijkstra(graph4, '1')
#print the cost and next hop

print(2* "\n")

table4l=PrettyTable(["d","c","n"])

for i,j,k in zip(d.keys(),d.values(),p.values()):
  table4l.add_row([i,j,k])
  table4l.add_row(["","",""])
print(table4l)
print(" ")
print("( Link state routing :Time to run = %s seconds) " % (time.time() - start_time))
print("Link state routing:No of loop iterations:= ",countloop)

#run with different graphs
LOOP1=[]

print(2* "\n")

dvr="---DISTANCE VECTOR ROUTING PERFORMANCE---"
dvrp=dvr.center(155)
print(dvrp)
print(2 *"\n")
print("________")
print("")
print("---Distance vector routing performace in graph1---")
d, p,countloop = bellman_ford(graph1, 'a')
#print the cost and next hop

table1=PrettyTable(["d","c","n"])
for i,j,k in zip(d.keys(),d.values(),p.values()):
 table1.add_row([i,j,k])
 table1.add_row(["","",""])

print(table1)
print(" ")
print("No of loop iterations:= ",countloop)
LOOP1.append(countloop)

print(2* "\n")
print("________")
print("")

print("---Distance vector routing performace in graph2---")
d, p,countloop = bellman_ford(graph2, 'A')
#print the cost and next hop
table2=PrettyTable(["d","c","n"])
for i,j,k in zip(d.keys(),d.values(),p.values()):
 table2.add_row([i,j,k])
 table2.add_row(["","",""])
print(table2)
print(" ")
print("No of loop iterations:= ",countloop)
LOOP1.append(countloop)
print(2* "\n")
print("________")
print("---Distance vector routing performace in graph3---")
print("")
d, p,countloop = bellman_ford(graph3, 'A')
#print the cost and next hop
table3=PrettyTable(["d","c","n"])
for i,j,k in zip(d.keys(),d.values(),p.values()):
 table3.add_row([i,j,k])
 table3.add_row(["","",""])
print(table3)
print(" ")
print("No of loop iterations:= ",countloop)
LOOP1.append(countloop)
print(2* "\n")
print("________")
print("")
print("---Distance vector routing performace in graph4---")
d, p,countloop = bellman_ford(graph4, '1')

print(table4)
print(" ")
print("No of loop iterations:= ",countloop)
LOOP1.append(countloop)
LOOP2=[]

print(2 * "\n")

lsr="---LINK STATE ROUTING PERFORMANCE---"
lsrp=lsr.center(155)
print(lsrp)
print("\n")
print("________")
print("")
print ("---Link state routing performace in graph1:---")

d,p,countloop=dijkstra(graph1, 'a')

table1l=PrettyTable(["d","c","n"])
for i,j,k in zip(d.keys(),d.values(),p.values()):

 table1l.add_row([i,j,k])
 table1l.add_row(["","",""])

print(table1l)
print("")
print("No of loop iterations:= ",countloop)
LOOP2.append(countloop)
print("________")
print ("---Link State routing performace in graph2:---")
d, p,countloop =dijkstra(graph2, 'A')
table2l=PrettyTable(["d","c","n"])
for i,j,k in zip(d.keys(),d.values(),p.values()):
  table2l.add_row([i,j,k])
  table2l.add_row(["","",""])

print(table2l) 

print(" ")
print("No of loop iterations:= ",countloop)
LOOP2.append(countloop)
#print the cost and next hop
print("________")
print("")
print ("---Link state routing performace in graph3:---")
d, p,countloop =dijkstra(graph3, 'A')
table3l=PrettyTable(["d","c","n"])
for i,j,k in zip(d.keys(),d.values(),p.values()):
 table3l.add_row([i,j,k])
 table3l.add_row(["","",""])

print(table3l)
print(" ")
print("No of loop iterations:= ",countloop)
LOOP2.append(countloop)
print("________")
print("")
print ("---Link state routing performance in graph4:---")
d, p,countloop = dijkstra(graph4, '1')

print(table4l)
print(" ")
print("No of loop iterations:= ",countloop)
LOOP2.append(countloop)
print("________")
print ("No of Loops by Distance Vector routing and Link state routing for respective graphs are")
print(" ")
print ("Distance vector :",LOOP1)
print("")
print ("Link state :",LOOP2)
print("________")


#PLOT THE GRAPH TO COMPARE THE PERFORMANCE
import numpy as np
import matplotlib.pyplot as plt
N = 4
ind = np.arange(N) # the x locations for the groups
width = 0.30 # the width of the bars
fig = plt.figure()
ax = fig.add_subplot(111)
yvals = [4, 9, 2,3]
rects1 = ax.bar(ind, LOOP1, width, color='teal',label='Distance Vector')
zvals = [1,2,3,4]
rects2 = ax.bar(ind+width, LOOP2, width, color='gold',label='Link state')
plt.title("Performance Comparision of Distance Vector VS link state")
ax.set(facecolor="ghostwhite")
ax.set_ylabel('No of Loops')
ax.set_xlabel('(No of nodes-No of links)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('5-8', '6-18', '7-24','20-94') )
ax.legend(loc='upper left')
def autolabel(rects):
 for rect in rects:
  h = rect.get_height()
  ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),ha='center', va='bottom')
autolabel(rects1)
autolabel(rects2)
plt.show()


#Dvr vs Lsr Complexity comparison
import random
import math
import matplotlib.pyplot as plt
"""
The comparison of time complexity
"""
n=random.sample(range(5000), 30)
# N=no of nodes in sorted order
N=sorted(n)
#No of links=No of nodes
L=N
B=[];D=[];
#Calculate complexity
for l,n in zip(L,N):
 B.append(l*n);
 D.append(l+n*(math.log( n )));

plt.plot(N,B,'r.-',label="Distance vector")
plt.plot(N,D,'b.-',label="Link state")
plt.title("Complexity curve of Distance vector vs Link state routing")
plt.xlabel("No of nodes=No of Links")
plt.ylabel("Time complexity")
plt.legend()
plt.grid(b=True, which='major',color='grey',linestyle='-',linewidth='0.25')
plt.minorticks_on()
plt.grid(b=True,which='minor',color='grey',linestyle='--',linewidth='0.25')
plt.show()