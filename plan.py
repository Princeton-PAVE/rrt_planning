import json
import numpy as np 
import queue
import threading
from typing import List

#we want matrix to be an occupancy matrix

"""
assume following json

{
    [
    "object_tag": <str>,
    "center": <List<float>>(x,y,z)
    "height": <float>, 
    "width": <float>.
    "depth: <float>
    "yaw" <float> in [0, 360]
    ]   
}
"""
MATRIX_HEIGHT = 10
MATRIX_DEPTH = 10

input_queue = queue.Queue()

data = {"object_tag": "Mayank in box form", "center": (500, 500, 500), 
        "height": 10, "width": 5, "depth": 1, "yaw": 0}

#assume data is list of bbox params (listed above)
def fill_occupancy_matrix(data: List) -> np.ndarray:

    # add depth
    for object_tag, center, height, width, heading in data: 
        cy, cx = center # cy = row, cx = col
    
        # 1. Translate: Shift coordinates so center is (0,0)
        shifted_rows = MATRIX_HEIGHT - cy
        shifted_cols = MATRIX_DEPTH - cx
        
        # 2. Rotate: Apply inverse rotation to the grid
        # Rotation Matrix: [cos -sin; sin cos]
        cos_h = np.cos(-heading)
        sin_h = np.sin(-heading)
        
        # Calculate coordinates in the box's local frame
        local_r = shifted_rows * cos_h - shifted_cols * sin_h
        local_c = shifted_rows * sin_h + shifted_cols * cos_h
        
        # 3. Mask: Check which pixels fall within the bounds
        # We use height/2 and width/2 because the center is at 0
        mask = (np.abs(local_r) <= height / 2) & (np.abs(local_c) <= width / 2)
        
        # 4. Fill: Set those positions to 1
        matrix[mask] = 1
       
       
#main loop 
def plan(json_file):
    
    while True:
        #Loop is pull data from Mayank's team
        #fill occupancy matrix
        #do something with it !
        if input_queue:
            data = input_queue.get() #this channel will block until new data comes in
            occupany_matrix = fill_occupancy_matrix(data)
            do_something(occupancy_matrix) #render + calculate action 
        


#next time
#set up Pygame renderer
#make sure fill_occupancy_matrix works
#figure out what do_something() (planner) will do!
        
        
    
    
    


