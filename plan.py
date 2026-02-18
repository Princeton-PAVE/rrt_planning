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

#we want matrix to be an occupancy matrix

"""
assume following json

{
    [
        {
        "object_tag": <str>,
        "center": <List<float>>(x,y)
        "dimensions": <List<float>>(width, height)
        "yaw" <float> in [0, 360]
        }
    ]   
}

#(0,0) is top left

#(1,1), width = 1, height = 1, 3x3 around (1,1)
#yaw is clockwise from x-axis

"""


queue = [[{"object_tag": "car", "center": [400, 400], "dimensions": [100, 20], "yaw": 0},
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

#queue = [mock_data]


MATRIX_HEIGHT = 1000
MATRIX_WIDTH = 1000





"""
data = {"object_tag": "Mayank in box form", "center": (500, 500), 
        "height": 10, "width": 5, "yaw": 0}
        """

#test_matrix = np.full([MATRIX_HEIGHT, MATRIX_DEPTH], 1)

#test_matrix[5][5] = 0

#assume data is list of bbox params (listed above) #BAD GPT USED
def fill_bev_matrix(data: List) -> np.ndarray:

    matrix = np.full([MATRIX_HEIGHT, MATRIX_WIDTH], 255)    
    
    
    for obj in data: 
        object_tag, center, dimensions, yaw = itemgetter('object_tag', 'center', 'dimensions', 'yaw')(obj)
        #print("object_tag:", object_tag)
        #print("center", center)
        cx, cy  = center # cy = row, cx = col
        width, height = dimensions
        
        """
        xs = np.arange(cx - width, cx + width + 1) #account for center
        ys = np.arange(cy - height, cy + height + 1)
        
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        print("X", X)
        print("Y", Y)
        #print("mask: 1", mask)  
        matrix[Y, X] = 0
        print("matrix", matrix)
        """
        
        
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
    
    return matrix
       


       
def visualize(matrix):
    
    _win.setImage(matrix.T, autoLevels=False)
    pg.QtWidgets.QApplication.processEvents() 
    
cv2.destroyAllWindows()
#main loop 
def plan(input_queue):
    #queue 
    while input_queue:
        #Loop is pull data from Mayank's team
        #fill occupancy matrix
        #do something with it !
        #data = input_queue.get().jsonify() #maybe unjsonify #this channel will block until new data comes in
        current_data = input_queue.popleft()
        bev_matrix = fill_bev_matrix(current_data) #bev cell
        visualize(bev_matrix)
        #show GUI
        #do_something(bev_matrix)
        #path, nodes = rrt_star(
            #bev_matrix, start, goal, step_size=MAZE_SIZE*MAX_STEP_SIZE, 
            #radius=0.5, max_iter=1000, goal_thresh=0.5)
        #do_something(bev_matrix) #render + calculate action  #MAYBE ILQR IF ROCCO APPROVES (need his approval)
        # time.sleep(1000)
        


def generate_data() -> deque:
    dataQueue = deque()


    iterations = 20
    dt = 1
    angular_freq = 0.1
    movement_vel = 1
    for i in range(iterations):
        t = i * dt
        dataQueue.appendleft([
            {"object_tag": "car", "center": [750 - movement_vel * t, 700], "dimensions": [400, 100], "yaw": angular_freq * t}
        ])
    
    return dataQueue


_win = pg.ImageView()
_win.show()



"""
from pyqtgraph.Qt import QtWidgets, QtCore

class DataProducer(QtCore.QObject):
        data_ready = QtCore.pyqtSignal(list) 
        
        def generate(self):
            dataQueue = deque()
            iterations = 20
            dt = 1
            angular_freq = 0.1
            movement_vel = 1
            for i in range(iterations):
                t = i * dt
                dataQueue.appendleft([
                    {"object_tag": "car", "center": [750 - movement_vel * t, 700], "dimensions": [400, 100], "yaw": angular_freq * t}
                ])
            self.data_ready.emit(dataQueue)
    
elsewhere in the code:

emitter = DataEmitter()
emitter.data_ready.connect(lambda data: plan(data))
"""

def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    data_queue = generate_data()
    

    thread = threading.Thread(target = plan, args=(data_queue,), kwargs={})
    thread.start()

    app.exec()
    
    """
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: plan(data_queue))
    """
    # timer.start(1000)  # Check the queue every 10ms (100 FPS)
    # print("data_queue", data_queue)
    # plan(data_queue)
    
    
        
if __name__ == "__main__":
    main()
    