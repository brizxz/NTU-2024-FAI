import itertools
import os
import time

import pandas as pd

from maze import Maze
from search import search


def execute(filename, searchMethod):
    maze = Maze(filename)
    t1 = time.time()
    path = search(maze, searchMethod)
    total_time = time.time()-t1  # time in seconds
    statesExplored = maze.getStatesExplored()
    print("Results")
    print("Path Length:", len(path))
    print("States Explored:", statesExplored)
    print("Total time", total_time, "seconds")
    if "p4/" in filename:
        assert maze.isValidPath(path) in ['Last position is not goal', 'Unnecessary path detected', 'Valid',]
    else: 
        assert maze.isValidPath(path) == 'Valid'
    return len(path), statesExplored, total_time


maze_dir = "./maps_full"
# maze_files = {}
# for subdir in os.listdir(maze_dir):
#     subdir_path = os.path.join(maze_dir, subdir)
#     if os.path.isdir(subdir_path):
#         maze_files[subdir] = []
#         for filename in os.listdir(subdir_path):
#             maze_files[subdir].append(os.path.join(subdir_path, filename))

res = {}

# for method in methods:
#     if method == "astar" or method == "bfs": #P1
#         maze_file = maze_files["single"]
#     elif method == "astar_corner": #P2
#         maze_file = maze_files["corner"]
#     elif method == "astar_multi": #P3
#         maze_file = maze_files['astar_multi']
#     elif method == "fast": #P4
#         maze_file = maze_files["multi"]

#     for maze in maze_file:
#         print(f"Method: {method} - Maze: {maze}")
#         path_len, statesExplored, total_time = execute(maze, method)
#         res[(method, maze)] = (path_len, statesExplored, total_time)

for part_idx in range(1, 5):
    subdir_path = os.path.join(maze_dir, "p" + str(part_idx))
    if part_idx == 1:
        methods = ["astar", "bfs"]
    elif part_idx == 2:
        methods = ["astar_corner"]
    elif part_idx == 3:
        methods = ["astar_multi"]
    elif part_idx == 4:
        methods = ["fast"]

    for maze_file in os.listdir(subdir_path):
        maze_filename = os.path.join(subdir_path, maze_file)
        for method in methods:
            print(f"Method: {method} - Maze: {maze_filename}")
            path_len, statesExplored, total_time = execute(maze_filename, method)
            res[(method, maze_filename)] = (path_len, statesExplored, total_time)


df = pd.DataFrame.from_dict(res, orient='index', columns=['Path Length', 'States Explored', 'Total Time'])
df[['Method', 'Maze File']] = pd.DataFrame(df.index.tolist(), index=df.index)
df.reset_index(drop=True, inplace=True)
df.to_csv('output.csv', index=False)

print("Save Done!")
#print(df)