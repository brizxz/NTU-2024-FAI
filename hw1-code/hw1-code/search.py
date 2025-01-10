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
# to the poss of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

from collections import deque
import heapq
import itertools
import copy
import sys

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)

def get_path(start, dot, path):
    res_path = []  # path to find the dot
    res_path.append(dot) # append the dot into the list
    nexts = dot
    while(nexts != start): # find the path from dot to the start point
        res_path.append(path[nexts])
        nexts = path[nexts]

    res_path.reverse()
    return res_path # return the reversed path


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    dot_pos = maze.getObjectives()[0]
    visted = {}; path = {}
    queue = deque([start])
    visted[start] = True
    path[start] = None

    while queue:
        nodenow = queue.popleft()
        if nodenow == dot_pos:
            break
        
        nei = maze.getNeighbors(nodenow[0],nodenow[1])
        for neighbor in nei:
            if neighbor not in visted and maze.isValidMove(neighbor[0],neighbor[1]):
                visted[neighbor] = True
                path[neighbor] = nodenow
                queue.append(neighbor)
    return get_path(start, dot_pos, path)


def heuristic(node, goal):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    dot_pos = maze.getObjectives()[0]
    visited = set()
    open_list = [(0, start, [start])]

    while open_list:
        _, node, path = heapq.heappop(open_list)
        
        if node == dot_pos:
            return path
        
        if node not in visited:
            visited.add(node)

        nei = maze.getNeighbors(node[0], node[1])
        for neighbor in nei:
            if neighbor not in visited and maze.isValidMove(neighbor[0], neighbor[1]):
                new_path = path + [neighbor]
                g = len(new_path)
                h = heuristic(neighbor, dot_pos)
                f = g + h
                heapq.heappush(open_list, (f, neighbor, new_path))
        
    return []

def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    start = maze.getStart()
    dots_pos = maze.getObjectives()
    min_length = float('inf')
    res = []
    permutations_list = [[list(perm)] for perm in itertools.permutations(dots_pos)]
    for perm in permutations_list:
        perm = perm[0]
        total_length = 0
        temp_start = start
        temp_visited = set()
        cur_path = []
        for dot_pos in perm:
            temp_visited.clear()
            open_list = [(0, temp_start, [temp_start])]
            while open_list:
                _, node, path = heapq.heappop(open_list)
                if node == dot_pos:
                    cur_path = cur_path + path
                    cur_path = cur_path[:-1]
                    total_length += len(path) - 1
                    temp_start = node
                    temp_visited.add(node)
                    break
                if node not in temp_visited:
                    temp_visited.add(node)
                    nei = maze.getNeighbors(node[0], node[1])
                    for neighbor in nei:
                        if neighbor not in temp_visited and maze.isValidMove(neighbor[0], neighbor[1]):
                            new_path = path + [neighbor]
                            g = len(new_path)
                            h = heuristic(neighbor, dot_pos)
                            f = g + h
                            heapq.heappush(open_list, (f, neighbor, new_path))
        if total_length < min_length:
            min_length = total_length
            res = cur_path
    for dot in dots_pos:
        if dot not in res:
            res = res + [dot]
    return res

class graph_node:
    def __init__(self, pos, cost,totalCost):
        self.pos = pos # a tuple of pos
        self.cost = cost #get the heuristic data of the Node
        self.previous = None
        self.notVisited = []
        self.totalCost = totalCost # The total cost that will be mst + manhan distance
        
    def __eq__(self,other): # check equal
        return self.pos == other.pos
    def __lt__(self,other): #check less than
        return self.totalCost < other.totalCost

def build_graph(maze, dots):
    mst_graph = {}
    edge = {}
    for source in dots:
        for dest in dots:
            if source != dest:
                path=[] #build a path of mst
                q = []
                visitedList = {}
                cost = heuristic(source,dest)
                heapq.heappush(q,((cost,[source])))
                while q:
                    path = heapq.heappop(q)[1]
                    start_p = path[-1]
                    if start_p not in visitedList:
                        startCost = heuristic(start_p,dest) + len(path)/1.5 - 1 
                        visitedList[start_p] = startCost
                        if start_p == dest:
                            break

                        neighbors = maze.getNeighbors(start_p[0], start_p[1]) 
                        for neighbor in neighbors:
                            newCost = heuristic(neighbor,dest) + len(path)/1.5 - 1
                            if neighbor not in visitedList:
                                heapq.heappush(q,(newCost, path + [neighbor]))
                            elif visitedList[neighbor] > newCost: # insert if newcost is small or equal 
                                visitedList[neighbor] = newCost
                                heapq.heappush(q,(newCost, path + [neighbor]))

                edge[(source, dest)] = path
                mst_graph[(source, dest)] = len(path)
    return mst_graph, edge

def Prim_mst(dots, path):
    start = dots[0]
    mst_path = []
    mstCost = 0
    visited = set([start])  # 使用集合來追蹤訪問過的點
    edges = []

    while len(dots) > len(visited):
        for source in visited:
            for dest in dots:
                if dest not in visited:
                    edge = (source, dest)
                    cost = path[edge] - 1
                    heapq.heappush(edges, (cost, edge))

        while True:
            newEdge = heapq.heappop(edges)
            edgeGoal = newEdge[1][1]
            if edgeGoal not in visited:
                break

        mst_path.append(newEdge[1])
        mstCost += newEdge[0]
        visited.add(edgeGoal)

    return mstCost


def astar_multi(maze):
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
    start_buf = maze.getStart()
    dots.insert(0, start) # put start into the goals

    mst_graph, edge = build_graph(maze, dots)
    
    getPath = []
    goals = []
    notVisited = {}
    visited = {}
    queue = [] # current path

    mstCost = Prim_mst(dots, mst_graph) # get mst cost
    startNode = graph_node(start_buf, 0, mstCost)
    startNode.notVisited = maze.getObjectives()
    heapq.heappush(queue,startNode)
    notVisited[start] = len(startNode.notVisited) 

    while len(dots):
        current = heapq.heappop(queue)
        if not current.notVisited: 
            break

        for next in current.notVisited:
            newCost = current.cost + mst_graph[(current.pos, next)]
            nextNode = graph_node(next, newCost, 0)
            nextNode.previous = current 
            nextNode.notVisited = copy.copy(current.notVisited) 

            if next in nextNode.notVisited:
               nextNode.notVisited.remove(next)
            notVisited[next] = len(nextNode.notVisited)
            visited[next] = 0 
            mstCost = Prim_mst(current.notVisited, mst_graph) #use mst to get cost
            nextNode.totalCost = newCost + mstCost #total = mst + current cost + edge cost
            if len(dots):
                nextNode.totalCost += len(nextNode.notVisited)
            heapq.heappush(queue,nextNode)
    
    while current:
        goals.append(current.pos) 
        current = current.previous
    for i in range(len(goals) - 1):
        getPath += edge[(goals[i], goals[i+1])][:-1] #getting path and get rid of the repeaded one
    getPath.append(start)
    getPath.reverse()
    return getPath #return reverse one

def fast_heuristic(start, dots):
    min_dis = float("inf")
    for dot in dots:
        distance = heuristic(start, dot) 
        min_dis = min(min_dis, distance)
    return min_dis

def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.
    Use Dijkstra algorithm
    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = maze.getStart()
    dots = maze.getObjectives()
    res_path = []

    while dots:
        queue = []
        node = graph_node(start, fast_heuristic(start, dots), 0)
        heapq.heappush(queue, node)
        path = {}
        visited = []
        while queue:
            start_p = heapq.heappop(queue)
            start_pos = start_p.pos
            visited.append(start_pos)
            if start_pos in dots:
                startTarget = start_pos
                break
            neighbors = maze.getNeighbors(start_pos[0], start_pos[1]) 
            for neighbor in neighbors:
                if (not neighbor in visited) and maze.isValidMove(neighbor[0],neighbor[1]): 
                    path[neighbor] = start_pos
                    heapq.heappush(queue,graph_node(neighbor, len(get_path(start,start_pos,path)), 0))
        pathMap = get_path(start, startTarget, path)
        newStart = pathMap[-1] 
        if len(dots) > 1:
            pathMap.remove(newStart)
        res_path.extend(pathMap) 
        dots.remove(newStart)
        start = newStart
    return res_path