import json
import numpy as np 
import queue
import threading
from typing import List
from operator import itemgetter
import math
#GUI visualizer
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

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


"""


mock_data = [{"object_tag": "car", "center": [1, 1], "dimensions": [1, 1], "yaw": 0}]


app = QtWidgets.QApplication([])


MATRIX_HEIGHT = 100
MATRIX_WIDTH = 100

input_queue = queue.Queue()


"""
data = {"object_tag": "Mayank in box form", "center": (500, 500), 
        "height": 10, "width": 5, "yaw": 0}
        """

#test_matrix = np.full([MATRIX_HEIGHT, MATRIX_DEPTH], 1)

#test_matrix[5][5] = 0

#assume data is list of bbox params (listed above) #BAD GPT USED
def fill_bev_matrix(data: List) -> np.ndarray:

    matrix = np.full([MATRIX_HEIGHT, MATRIX_WIDTH], 255)
    
    # add depth

    # from operator import itemgetter
    # ...
    # params = {'a': 1, 'b': 2}
    # a, b = itemgetter('a', 'b')(params)
    #print("")
    
    
    
    
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
       
       
#main loop 
def plan(json_file):
    
    while True:
        #Loop is pull data from Mayank's team
        #fill occupancy matrix
        #do something with it !
        if input_queue:
            data = input_queue.get().jsonify() #maybe unjsonify #this channel will block until new data comes in
            bev_matrix = fill_bev_matrix(data) #bev cell
            #show GUI
            do_something(bev_matrix) #render + calculate action  #MAYBE ILQR IF ROCCO APPROVES (need his approval)
        


#next time
#set up Pygame renderer
#make sure fill_occupancy_matrix works
#figure out what do_something() (planner) will do!

win = pg.ImageView()
bev_matrix = fill_bev_matrix(mock_data)
print("bev_matrix", bev_matrix)
win.setImage(bev_matrix.T)
win.show()
    
app.exec()



    
    


