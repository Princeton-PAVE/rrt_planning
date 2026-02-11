from maze2d import Maze2DEnv
from planner import InformedRRTStar
import numpy as np
import cv2
from time import perf_counter
#environment is simulation
def main():
    """
    maze_map = [
        [0,0,0,0,0,0,0,0,0,0],
        [0,1,1,0,0,0,1,1,1,0],
        [0,0,0,0,1,0,0,0,1,0],
        [0,1,1,1,1,1,0,0,1,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
    ]
    """
    """
    maze_map = np.zeros((10, 10), dtype=np.uint8) #no walls to make sure RRT* is working
    maze_map[3,3] = 1
    """
    
    #(0,0) is top left
    
    maze_map = np.array([
        [0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,1,1,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0,0],
    ])
    
    MAZE_SIZE = 1000 # length width
    MAX_STEP_SIZE = 0.025 # units are in proportion of MAZE_SIZE
    GOAL_SIZE = 1
    
    maze_map = maze_map.astype(np.uint8)
    maze_map = cv2.resize(maze_map, (MAZE_SIZE, MAZE_SIZE), interpolation=cv2.INTER_NEAREST)

    #maze_map = [[0] * 100 for _ in range(10)]
    
    for r in range(len(maze_map)):
        for c in range(len(maze_map[0])):
            if maze_map[r][c]:
                # print((r,c))
                pass
    
    
    #(y,x)
    env = Maze2DEnv(maze_map)
    start = (500, 20)
    goal = (600, 800)
    planner = InformedRRTStar(env, start, goal)
    
    N = 10
    t0 = perf_counter()
    for i in range(N):
        # print("run", i)
        path, nodes = planner.calculate_path(
            step_size=MAZE_SIZE*MAX_STEP_SIZE, 
            radius=0.5, max_iter=1000, goal_thresh=0.5)
    t1 = perf_counter()
    
    print(f"Found in {(t1-t0)/N:.4f}")
    # print(f"Path distance: {nodes[-1].cost}")
    # print("path:", path)
    
    if path is not None:
        env.pos, env.goal = path[0], path[-1]
        env.render_maze(path=path, nodes=nodes, save_path="rrt_star_result.png")
    else:
        # still useful to visualize the explored tree
        env.pos, env.goal = start, goal
        env.render_maze(nodes=nodes, save_path="rrt_star_result.png")
            
    

if __name__ == "__main__":
    main()