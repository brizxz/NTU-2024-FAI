# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Shang-Tse Chen (stchen@csie.ntu.edu.tw) on 03/03/2022

"""
This is the main entry point for HW1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

import sys
import heapq
from queue import PriorityQueue
import copy


def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    
    start = maze.getStart() #get start of Maze
    startP = start #set a startPoint in queue
    dot = maze.getObjectives()[0] # get the single dot position
    visited = {} # a visited dictionary
    queue = [] # queue list for bfs
    queue.append(startP)
    visited[startP] = True
    path= {} # set the path to record each vertex came from
    path[startP] = None # set the start Point as the key, the value is none
    while queue:
        startP = queue.pop(0)
        if(startP == dot): # early exit if start point = dot
            break
        neighbor = maze.getNeighbors(startP[0],startP[1])
        for i in neighbor:
            if maze.isValidMove(i[0],i[1]) and (not i in visited): # check the neighbor if is visited and is valid move or not.
                visited[i] = True
                path[i] = startP  #set the neighbor as the key, the current vertex as the value in order to avoid multiple neighbor that cause the updating dict value
                queue.append(i)

   
    path_to_findDot = get_path_toDot(start,dot,path)
    return path_to_findDot # return the path

def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = maze.getStart()
    startP = start 
    dot = maze.getObjectives()[0] 
    queue = []
    heapq.heappush(queue, (0,startP))
    path = {} 
    path[startP] = None
    cost = {} 
    cost[startP] = 0
    
    while queue:
        startP = heapq.heappop(queue)[1]
        if(startP == dot): 
            break
        neighbor = maze.getNeighbors(startP[0],startP[1]) 
        for i in neighbor:
            if maze.isValidMove(i[0],i[1]): 
                newCost = cost[startP]+1  
                if (not i in cost) or newCost<cost[i]: 
                    cost[i] = newCost 
                    priority = newCost + ManhattanDistance(i, dot) # f = g + h
                    heapq.heappush(queue, (priority,i)) # set the queue priority depending upon the f = g+h
                    path[i] = startP 

    path_to_findDot = get_path_toDot(start,dot,path) 
    return path_to_findDot 

def ManhattanDistance(current, dot):
    """
    Use the Manhattan distance from the current postion to the dot position as the heuristic function.
    @param current: current position
    @param dot: The dot position
    @return Manhattan distance
    """
    return abs(current[0]-dot[0]) + abs(current[1]-dot[1])

def get_dis(maze, start):
    from collections import deque
    n, m = maze.getDimensions()
    dis = [[-1 for i in range(m)] for j in range(n)]
    q = deque([start])
    dis[start[0]][start[1]] = 0
    while q:
        nx, ny = q.popleft()
        for (x, y) in maze.getNeighbors(nx, ny):
            if dis[x][y] == -1:
                dis[x][y] = dis[nx][ny] + 1
                q.append((x, y))
    return dis

class UnionFind:
    def __init__(self, n):
        self.lead = list(range(n))
        
    # disjoint set
    def find(self, x):
        if self.lead[x] == x: return x
        else: 
            self.lead[x] = self.find(self.lead[x])
            return self.lead[x]
    
    def same(self, x, y):
        return self.find(x) == self.find(y)

    def union(self, x, y):
        x = self.find(x); y = self.find(y)
        self.lead[x] = y

def astar_mst(maze):
    foods = maze.getObjectives()
    k = len(foods)
    dis_from_foods = [get_dis(maze, food) for food in foods]

    def kruskal(edges):
        edges = sorted(edges, key=lambda x: x[2])
        dsu = UnionFind(k + 1)
        mst = 0
        for u, v, w in edges:
            if not dsu.same(u, v):
                mst += w
                dsu.union(u, v)
        return mst

    mst_dp = dict()
    class Node:
        def __init__(self, x, y, food_ids, g, parent=None):
            self.x = x; self.y = y
            self.food_ids = food_ids
            self.g = g; self.h = self._get_h()
            self.parent = parent

        @property
        def f(self):
            return self.g + self.h

        def goal(self):
            return self.food_ids == 0

        def check_eat(self):
            for i in range(k):
                if (self.x, self.y) == foods[i] and (self.food_ids >> i & 1):
                    if self.food_ids & (1 << i):
                        self.food_ids &= ~(1 << i)
                    else:
                        self.food_ids |= (1 << i)
                        break
        
        def _get_h(self):
            self.check_eat()
            if self.food_ids == 0:
                return 0

            mst = mst_dp.get(self.food_ids)
            if not mst:
                edges = []
                for i in range(k):
                    if not (self.food_ids >> i & 1): continue
                    for j in range(k):
                        if not (self.food_ids >> j & 1): continue
                        w = dis_from_foods[i][foods[j][0]][foods[j][1]]
                        edges.append((i, j, w))
                mst = kruskal(edges)
                mst_dp[self.food_ids] = mst

            mi = float('inf')
            for i in range(k):
                if not (self.food_ids >> i & 1): continue
                mi = min(mi, dis_from_foods[i][self.x][self.y])
            mst += mi
            return mst

        def __eq__(self, other):
            return self.x == other.x and self.y == other.y and self.food_ids == other.food_ids
        
        def __lt__(self, other):
            return self.f < other.f
        
        def __hash__(self) -> int:
            return (self.x, self.y, self.food_ids).__hash__()
        
    start = maze.getStart()
    start = Node(x=start[0], y=start[1], food_ids=(1 << k) - 1, g=0)
    pq = []
    from collections import defaultdict
    vis = defaultdict(lambda: False)
    dis = defaultdict(lambda: float('inf'))
    
    heapq.heappush(pq, start)
    dis[start] = start.g
    while pq:
        now = heapq.heappop(pq)
        if vis[now]: continue
        vis[now] = True
        if now.goal():
            path = []
            while now:
                path.append((now.x, now.y))
                now = now.parent
            return path[::-1]
        
        for (x, y) in maze.getNeighbors(now.x, now.y):
            child = Node(x = x, y = y, food_ids = now.food_ids, g = now.g + 1, parent = now)
            if not vis[child] and child.g < dis[child]:
                dis[child] = child.g
                heapq.heappush(pq, child)
    return []

def get_path_toDot(start, dot, path):
    path_to_findDot = []  
    path_to_findDot.append(dot) 
    nexts = dot
    while(nexts != start): # find the path from dot to the start point
        path_to_findDot.append(path[nexts])
        nexts = path[nexts]
    return path_to_findDot[::-1] # return the reversed path

def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    ##call multi function to find the 4 corners dot
    return astar_multi1(maze)

def astar_multi(maze):
    return astar_mst(maze)

def astar_multi1(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    if len(maze.getObjectives()) == 1:
        return astar(maze)
    start = maze.getStart()
    dots = maze.getObjectives()
    startPosition = maze.getStart()
    dots.insert(0, start) #put start into the goals
    edge = {}
    myMst = {}
    getPath = []
    goals = []
    notVisited = {}
    visited = {}
    queue = [] #a queue to get current path
    for i in dots:
        for j in dots:
            if i != j:
                path=[] #build a path of MST
                q = []
                visitedList = {}
                cost = ManhattanDistance(i,j)
                heapq.heappush(q,((cost,[i])))
                while q:
                    path = heapq.heappop(q)[1]
                    startP = path[-1]
                    if startP not in visitedList: #check startp if is visited or not
                        startCost = ManhattanDistance(startP,j)+len(path)/1.5-1 #heuristic
                        visitedList[startP] = startCost #update the visited list
                        if startP == j:
                            break
                        neighbors = maze.getNeighbors(startP[0],startP[1]) 
                        for neighbor in neighbors:
                            newCost = ManhattanDistance(neighbor,j)+len(path)/1.5-1
                            if neighbor not in visitedList: #check the condition
                                heapq.heappush(q,(newCost,path+[neighbor]))
                            elif visitedList[neighbor]>newCost:
                                visitedList[neighbor] = newCost
                                heapq.heappush(q,(newCost,path+[neighbor]))

                edge[(i, j)] = path
                myMst[(i, j)] = len(path)
    
    mstCost = PrimMst(dots, myMst) #get the prim's MST cost
    startNode = HeuristicNode(startPosition, 0, mstCost) #put the start Node into the Node class
    #startNode.notVisited = dots
    startNode.notVisited = maze.getObjectives()# re-get the objectives, previous lists of dots has been updated

    heapq.heappush(queue,startNode) #put start Node into the queue
    notVisited[start] = len(startNode.notVisited) #set notVisited dict

    while len(dots):
        current = heapq.heappop(queue)
        if not current.notVisited: #check current.notvisited
            break
        for next in current.notVisited:
            newCost = current.cost + myMst[(current.position, next)]
            nextNode = HeuristicNode(next, newCost, 0)
            nextNode.previous = current #link the node
            nextNode.notVisited = copy.copy(current.notVisited) #copy the current to nextNode
            if next in nextNode.notVisited:
               nextNode.notVisited.remove(next)
            notVisited[next] = len(nextNode.notVisited)
            visited[next] = 0 #flag itself as visited
            mstCost = PrimMst(current.notVisited, myMst) #use mst to get cost
            nextNode.totalCost = newCost + mstCost #total = mst + current cost + edge cost
            if len(dots):
                nextNode.totalCost += len(nextNode.notVisited)
            heapq.heappush(queue,nextNode)
    
    while current:
        goals.append(current.position) #build the list of goals through the previous node
        current = current.previous
    for i in range(len(goals)-1):
        getPath += edge[(goals[i],goals[i+1])][:-1] #getting path and get rid of the repeaded one
    getPath.append(start)
    return getPath[::-1] #return reverse one


class HeuristicNode:
    def __init__(self, position, cost,totalCost):
        self.position = position # a tuple of position
        self.cost = cost #get the heuristic data of the Node
        self.previous = None
        self.notVisited = []
        self.totalCost = totalCost # The total cost that will be mst + manhan distance
        
    def __eq__(self,other): # check equal
        return self.position == other.position
    def __lt__(self,other): #check less than
        return self.totalCost < other.totalCost
    def __hash__(self):
        return hash(str(self.totalCost))
    def __str__(self): # return the string
        return "The Position is "+ str(self.position)+ " The totalCost is "+str(self.totalCost) + " The Not Visited list is "+str(self.notVisited)


def NewHeuristic(start, dots):
    minD = sys.maxsize
    for dot in dots:
        distance = ManhattanDistance(start, dot) #use manhattan distance to get the h
        minD = min(minD,distance)
    return minD
        
def PrimMst(dots,path):
    ## finding MST by using Prim's algorithm
    start = dots[0] #get the first goal
    MstPath = []
    mstCost = 0
    visited = {}
    visited[start] = True
    while len(dots) > len(visited):
        queue = []
        for i in visited:
            for j in dots:
                if visited.get(j) != True:
                    edge = (i,j) #edge will be from a tuple position to another tuple position
                    cost = path[edge]-1 #make it for medium search
                    heapq.heappush(queue,((cost,edge)))
        newEdge = heapq.heappop(queue)
        MstPath.append(newEdge[1])
        mstCost+=newEdge[0]
        edgeGoal = newEdge[1][1] #get the edge goal
        visited[edgeGoal] = True
    return mstCost

def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.
    Use Dijkstra algorithm
    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = maze.getStart()
    dots = maze.getObjectives()
    path_to_findDot = []
    while dots:
        queue = [] # build a priorityQueue for a star search
        node = HeuristicNode(start,NewHeuristic(start,dots),0) #set a node for start points and put with the heuristic value
        heapq.heappush(queue,node)
        path = {}
        visited = []
        while queue:
            startP = heapq.heappop(queue)
            startPPosition = startP.position #get the position tuple of the start point
            visited.append(startPPosition)
            if startPPosition in dots:
                startTarget = startPPosition
                break
            neighbors = maze.getNeighbors(startPPosition[0], startPPosition[1]) #get the neighbor
            for neighbor in neighbors:
                if (not neighbor in visited) and maze.isValidMove(neighbor[0],neighbor[1]): #check neighbor
                    path[neighbor] = startPPosition
                    heapq.heappush(queue,HeuristicNode(neighbor,len(get_path_toDot(start,startPPosition,path)),0))#put into the queue depending upon the g+h
        pathMap = get_path_toDot(start,startTarget,path) #get the path be getting the target
        newStart = pathMap[-1] #get the new Start point depending upon the previous goal
        if len(dots) > 1:
            pathMap.remove(newStart)
        path_to_findDot.extend(pathMap) #extend the whole path
        dots.remove(newStart)
        start = newStart
    return path_to_findDot