import os
from skimage import io
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import torch
from aux import str_to_mask, get_indexes
import numpy as np
import scipy.misc 
import matplotlib
import itertools
from numba import njit, jit, prange
from PIL import Image
import matplotlib.pyplot as plt
import time
import shutil

import imageio.core.util


SLICE_HEIGHT = 140
SLICE_WIDTH = 140 

cat=[ 
        'FISH' ,
        'FLOWER' ,
        'GRAVEL' ,
        'SUGAR' 
        ]


def gen_data():
    high_res_directory = Path("../data/train_images/")
    destination = "../data/sliced_train_images/"
    high_res_images = sorted(os.listdir(high_res_directory))
    df = pd.read_csv("../data/train.csv")

    if os.path.exists(destination):
        shutil.rmtree(destination)
        
    os.makedirs(destination)



    d = {}
    for idx in tqdm(range(len(high_res_images))):
    # for idx in tqdm(range(1)):
        high_res_img_path = os.path.join(high_res_directory, high_res_images[idx])
        high_res_img = io.imread(high_res_img_path).astype(np.uint8)
        high_res_height, high_res_width, _  = high_res_img.shape

        x = df[df['Image_Label'].str.match( os.path.splitext(high_res_images[idx])[0])]
        x = x.reset_index()
        label=np.array(str_to_mask(df_rows=x, height=high_res_height, width=high_res_width)) 

        for i in range(high_res_height//SLICE_HEIGHT):
            for j in range(high_res_width//SLICE_WIDTH):
                row_pix = i*SLICE_WIDTH
                col_pix = j*SLICE_HEIGHT

                postfix=str((i*(high_res_width//SLICE_WIDTH)) + j)
                filename = os.path.basename(high_res_images[idx])
                filename = os.path.splitext(filename)[0] + '_' + postfix + '.jpg'
                path = destination + filename
                img = high_res_img[row_pix:row_pix+SLICE_HEIGHT, col_pix:col_pix+SLICE_WIDTH, :] 
                lbl = label[:, row_pix:row_pix+SLICE_HEIGHT, col_pix:col_pix+SLICE_WIDTH] 


                im = Image.fromarray(img)
                im.save(path)

                
                for k in range(lbl.shape[0]):
                    indexes = get_indexes(arr=lbl[k], height = SLICE_HEIGHT, width=SLICE_WIDTH) 
                    indexes = list(sum(indexes, ())) 
                    indexes =  ' '.join(list(map(str, indexes)))
                    d[filename+'_'+cat[k]] = indexes 


        
    df2 = pd.DataFrame(list(d.items()))
    df2.columns = ["Image_Label", "EncodedPixels"]
    df2.to_csv('../data/sliced.csv', index=False) 

gen_data()
