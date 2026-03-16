import json
import numpy as np 
import queue
import threading
from typing import List
from operator import itemgetter
import math
#GUI visualizer
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from collections import deque
import time
import cv2
from planner import rrt_star
from scipy.spatial import cKDTree
from visualizer import BevEnv, visualize_plan, init_visualizer

#we want matrix to be an occupancy matrix



#CONSTANTS
MATRIX_HEIGHT = 800
MATRIX_WIDTH = 800
WINDOW_NAME = "bev"

#queue = [mock_data]


#assume data is list of bbox params (listed above) #BAD GPT USED
def fill_bev_matrix(data: List) -> np.ndarray:

    matrix = np.full([MATRIX_HEIGHT, MATRIX_WIDTH], 255)    
    
    
    for obj in data: 
        object_tag, center, dimensions, yaw = itemgetter('object_tag', 'center', 'dimensions', 'yaw')(obj)
        #print("object_tag:", object_tag)
        #print("center", center)
        cx, cy  = center # cy = row, cx = col
        width, height = dimensions
        
        
        indices = np.arange(MATRIX_HEIGHT * MATRIX_WIDTH)
        x = np.mod(indices, MATRIX_WIDTH) - cx
        y = np.floor(indices / MATRIX_WIDTH) - cy

        x, y = np.meshgrid(
            np.arange(MATRIX_HEIGHT) - cx,
            np.arange(MATRIX_WIDTH)  - cy,
        )

        c = math.cos(yaw)
        s = math.sin(yaw)

        x_transformed = x * c + y * s
        y_transformed = -x * s + y * c

        mask = (
            (-width <= x_transformed) & (x_transformed <= width) &
            (-height <= y_transformed) & (y_transformed <= height)
        )
        matrix[mask] = 0
        
        #matrix[Y,X] = 0
    
    return matrix.astype(np.uint8)

#main loop 
def plan(input_queue):
    #queue 
    start = [500,10]
    goal = [500,700]
    while input_queue:
        current_data = input_queue.pop()
        bev = fill_bev_matrix(current_data)  # uint8, 0/255
      
        env = BevEnv(bev)
        path, nodes = rrt_star(
            env,
            start=start,
            goal=goal,
            step_size=10.0,
            radius=30.0,
            max_iter=1250,
            goal_thresh=15.0,
            rebuild_every=10,
            coord_order="rc",
        )

        overlay = visualize_plan(
            bev,
            start_rc=start,
            goal_rc=goal,
            nodes_rc=nodes, 
            path_rc=path,
            node_radius=3,
            max_edge_len=20.0,
            branch_alpha=0.8,
        )
        #extract actions
        #controls = get_controls(path) #acceleration, steering angle
        #enact(controls)
        cv2.imshow(WINDOW_NAME, overlay)
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord('q')):
            break

        


def generate_data() -> deque:
    dataQueue = deque()


    iterations = 20
    dt = 1
    movement_vel = 10


    for i in range(iterations):
        t = i * dt
        objects = []
        objectCount = 10
        distance = 2 * MATRIX_WIDTH
        minSize = np.log(10)
        maxSize = np.log(MATRIX_HEIGHT / 10)
        maxAngVel = 0.02
        for objectNumber in range(objectCount):
            objects.append(
                {
                    "object_tag": "car",
                    "center": [
                        np.random.default_rng(seed=0 + objectNumber).random() * distance - movement_vel * t,
                        np.random.default_rng(seed=1 + objectNumber).random() * MATRIX_HEIGHT
                    ],
                    "dimensions": [
                        np.exp(minSize + (maxSize - minSize) * np.random.default_rng(seed=2 + objectNumber).random()),
                        np.exp(minSize + (maxSize - minSize) * np.random.default_rng(seed=3 + objectNumber).random())
                    ],
                    "yaw": np.random.default_rng(seed=4 + objectNumber).random() * 2 * np.pi + (2 * np.random.default_rng(seed=5 + objectNumber).random() - 1) * maxAngVel * t
                }
            )

        objects.append(
            {
                "object_tag": "car",
                "center": [MATRIX_WIDTH / 2 - movement_vel * t, MATRIX_HEIGHT / 2],
                "dimensions": [MATRIX_HEIGHT / 20, MATRIX_HEIGHT / 5],
                "yaw": 0,
            }
        )
 
        dataQueue.appendleft(objects)

    
    return dataQueue

def main():
    init_visualizer()
    data_queue = generate_data()
    plan(data_queue)


    cv2.destroyAllWindows()    
        
if __name__ == "__main__":
    main()
    
    