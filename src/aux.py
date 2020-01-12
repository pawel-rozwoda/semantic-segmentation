import math
import numpy as np
from numba import njit

def get_x(value, height ):
    return int(value)//height

def get_y(value, height):
    return int(value)%height

def str_to_mask(*, df_rows, height, width): 

    # ret = np.zeros((4, height, width), np.uint8) 
    ret = np.zeros((4, height, width), np.bool) 
    df_rows = df_rows.reset_index()
    for index, row in df_rows.iterrows():
        if not isinstance(row['EncodedPixels'], float):#checks if attr is not empty
            vals = row['EncodedPixels'].split(' ')
            
            pairs = zip(vals[::2], vals[1::2])
            # arr = np.zeros((height, width), np.uint8)
            arr = np.zeros((height, width), np.bool)

            for pair in pairs:
                v = int(pair[0])
                move = int(pair[1])
                arr[get_y(v, height): get_y(v, height) + move , get_x(v, height)] = 1  
                
            ret[index] = arr
            
    return ret
    


@njit()
def get_indexes(*, arr, height, width): #* stands for requiring keyword args
    indexes = []
    # indexes = List()

    for i in range(width):
        flag = False
        counter = 0
        for j in range(height):
            if arr[j][i]:
                if not flag:
                    flag = True
                counter += 1

            else:
                if flag: 
                    indexes.append(((i*width) + j - counter , counter))
                    counter = 0
                    flag = False


        if arr[height - 1][i]: 
                indexes.append(((i*width) + j - counter + 1 , counter))
                counter = 0

    # print(indexes)
    return indexes 
