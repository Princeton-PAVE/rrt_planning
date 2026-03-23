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
import pdb


#(1) get_full_state -> GOOD?
#(2) get_controls -> GOOD?
#(3) get_control_path

#REMEMBER WE'RE SETTING SEED IN PLANNER.PY


#assumptions with state:
"""
(0,0) is top left [100, 50] would mean 100 down (y-dir) and 50 right (x-dir)

"""








#we want matrix to be an occupancy matrix



"""
when popped from queue [[{"object_tag": "car", "center": [400, 400], "dimensions": [100, 20], "yaw": 0},
             {"object_tag": "car", "center": [30, 100], "dimensions": [50, 10], "yaw": 0 },
             {"object_tag": "car", "center": [720, 700], "dimensions": [200.5, 20], "yaw": 3*np.pi/4}],
             
             
             
            [{"object_tag": "car", "center": [400, 400], "dimensions": [10, 20], "yaw": 0},
             {"object_tag": "car", "center": [30, 100], "dimensions": [50, 10], "yaw": 0 },
             {"object_tag": "car", "center": [730, 700], "dimensions": [200.5, 20], "yaw": 3*np.pi/4}],
            [{"object_tag": "car", "center": [40, 400], "dimensions": [10, 20], "yaw": 0},
             {"object_tag": "car", "center": [30, 100], "dimensions": [50, 10], "yaw": 0 },
             {"object_tag": "car", "center": [740, 70], "dimensions": [200.5, 20], "yaw": 10*np.pi/4}],
            [{"object_tag": "car", "center": [40, 40], "dimensions": [100, 20], "yaw": 0},
             {"object_tag": "car", "center": [30, 100], "dimensions": [50, 10], "yaw": 0 },
             {"object_tag": "car", "center": [70, 700], "dimensions": [200.5, 20], "yaw": 2.5*np.pi/4}],
            [{"object_tag": "car", "center": [400, 400], "dimensions": [100, 20], "yaw": 0},
             {"object_tag": "car", "center": [30, 100], "dimensions": [50, 10], "yaw": 0 },
             {"object_tag": "car", "center": [750, 700], "dimensions": [20.5, 20], "yaw": 3*np.pi/4}]
        ]
"""


#CONSTANTS
MATRIX_HEIGHT = 800
MATRIX_WIDTH = 800
WINDOW_NAME = "bev"

#queue = [mock_data]


m = 1 #mass
delta_t = 1


"""
(0,0) is top left [100, 50] would mean 100 down (y-dir) and 50 right (x-dir)
"""
def get_full_state(path):
    #input is List of Tuples (y,x)
    #output is List of List [Tuple(X,Y), vel, heading_angle]
    #output[0] = (X,Y), velocity, heading_angle
    
    output = []
    init_y, init_x = path[0]
    output.append([(init_y, init_x), 0, 0]) #x0, y0, 0, 0
    for y,x in path[1:]: #y,x
        prev_pos, _, _ = output[-1]
        prev_y, prev_x = prev_pos
        vel_x = (x - prev_x) / delta_t # divided by 1 since we assumme 1 time unit in between
        vel_y = (y - prev_y) / delta_t
        vel = math.sqrt(vel_x**2 + vel_y**2)
        
        #heading angle is (-pi/2, pi/2)
        heading_angle = math.atan2(y - prev_y, x - prev_x)
        output.append([(y,x), vel, heading_angle])
        #print("velocity:", vel, flush=True)
        #print("heading_angle:", heading_angle, flush=True)
    return output

#set of full states
#output is List of Tuples [(acceleration, steering angle]
#length is 1 less than states
#full states
def get_controls(states):
    #0 is to the right up is negative, down is positive for angles. ANGLE IS RELATIVE 
    #F=1, t=1
    #input is List of List [Tuple(X,Y), vel, heading_angle]
    #output is List of Tuples [(acceleration, steering angle] length is (states - 1)
    
    output = []
    (prev_y, prev_x), prev_vel, prev_heading = states[0]
    for (y,x), vel, heading in states[1:]:
        
        accel = (vel - prev_vel) / delta_t  
        steering_angle = heading - prev_heading #snaps into place for now
        
        output.append((accel, steering_angle))
        
        prev_y, prev_x, prev_vel, prev_heading = y, x, vel, heading
        #print("accel", accel, flush=True)
        #print("steering_angle", steering_angle, flush=True)
        #pdb.set_trace()
    return output


#HEADING ANGLE IS ABSOLUTE
#STEERING ANGLE IS RELATIVE
def get_control_path(controls, init_state):
    position = np.array([init_state[0], init_state[1]], dtype = np.float64)
    heading = 0
    velocity = 0

    density = 6
    positions = []#np.array([], dtype=np.float64)
    positions.append(position.copy())
    #acceleration, angle is contr
    print("controls", controls)
    #pdb.set_trace()
    for a, steering_angle in controls:
        distance_traveled = delta_t * velocity + 1/2 * a * delta_t ** 2
        #print("distance_traveled", distance_traveled)
        #pdb.set_trace()
        
        
        heading += steering_angle
        
        #print("new_heading:", heading)
    

        direction = np.array([math.sin(heading), math.cos(heading)])
        diff = direction * distance_traveled
        #print("diff:", diff)
        position += diff #(y,x)
        #print("position:", position)
        positions.append(position.copy())
        #print("position:", position)
        velocity += delta_t * a #acceleration is added after already moving
    
    return positions



    
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
        #path comes in as List of (y,x)'s
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

        print(f"path: {path}")
        controls, init_state = None, None
        if path is not None: #sometimes there is no vialable path so check this
            full_states = get_full_state(path)
            #print("full_state:", full_states)
            controls = get_controls(full_states) #0 is to the left
            init_state = (start[0], start[1]) # starting position + velocity is zero, heading angle is zero (right)
        
        vis_controls = None
        if controls: 
            controls_path = get_control_path(controls, init_state)
            print(f"path recreated from controls: {controls_path}")
            
            
            
        overlay = visualize_plan(
            bev,
            start_rc=start,
            goal_rc=goal,
            nodes_rc=nodes, 
            path_rc=path,
            node_radius=3,
            max_edge_len=20.0,
            branch_alpha=0.8,
            vis_controls=controls_path
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
    
    