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
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,1,0,0,0],
        [0,0,0,0,0,1,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0],
        [0,0,1,0,0,1,0,0,0,0],
        [0,0,1,0,0,1,0,0,0,0],
        [0,0,1,0,0,0,1,0,0,0],
        [0,0,1,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
    ])
    
    MAZE_SIZE = 1000 # length width
    start = (0.500 * MAZE_SIZE, 0.020 * MAZE_SIZE) # (y, x)
    goal = (0.500 * MAZE_SIZE, 0.800 * MAZE_SIZE)
    
    MAX_STEP_SIZE = 0.04 # units are in proportion of MAZE_SIZE
    REWIRE_SIZE = 0.025
    MAX_ITERATIONS = 1000
    GOAL_SIZE = 0.05
    REBUILD_EVERY = 10
    GOAL_SAMPLE_RATE = 0.1
    
    
    maze_map = maze_map.astype(np.uint8)
    maze_map = cv2.resize(maze_map, (MAZE_SIZE, MAZE_SIZE), interpolation=cv2.INTER_NEAREST) 
    
    #(y,x)
    env = Maze2DEnv(maze_map)
    planner = InformedRRTStar(env, start, goal)
    
    N_TRIALS = 30
    t0 = perf_counter()
    for i in range(N_TRIALS):
        # print("run", i)
        path, all_sampled = planner.calculate_path(
            step_size=MAX_STEP_SIZE * MAZE_SIZE,
            radius=REWIRE_SIZE * MAZE_SIZE,
            max_iter=MAX_ITERATIONS,
            goal_thresh=GOAL_SIZE * MAZE_SIZE,
            rebuild_every=REBUILD_EVERY,
            goal_sample_rate=GOAL_SAMPLE_RATE,
        )
    t1 = perf_counter()
    
    print(f"Found in {(t1-t0)/N_TRIALS:.4f}")
    # print(f"Path distance: {nodes[-1].cost}")
    # print("path:", path)
    print(MAX_ITERATIONS - len(all_sampled), " unused nodes")
    
    if path is not None:
        env.pos, env.goal = path[0], path[-1]
        env.render_maze(path=path, nodes=all_sampled, save_path="rrt_star_result.png")
    else:
        # still useful to visualize the explored tree
        env.pos, env.goal = start, goal
        env.render_maze(nodes=all_sampled, save_path="rrt_star_result.png")
            
    

if __name__ == "__main__":
    main()