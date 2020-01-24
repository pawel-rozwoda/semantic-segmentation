import math
import numpy as np
from numba import njit
import torch

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

    return indexes 




def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    '''
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''

    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape)-1))
    numerator = 2. * torch.sum(y_pred * y_true, axes)
    # denominator = torch.sum(torch.square(y_pred) + torch.square(y_true), axes)
    denominator = torch.sum(y_pred**2 + y_true**2, axes)

    return 1. - torch.mean(numerator / (denominator + epsilon)) # average over classes and batch 
