from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from pathlib import Path
import matplotlib
from aux import str_to_mask 

#tkagg for shell
matplotlib.use( 'tkagg' ) 


class MyDataset(Dataset):
    def __init__(self, *, train_dir, labels , transform=None, train, train_partition):
        self.train_dir = train_dir
        self.transform = transform 
        self.labels = pd.read_csv(labels)
        
        self.train_dir = Path(train_dir) 
        self.images = sorted(os.listdir(self.train_dir))
        if train:
            self.images = self.images[:int(train_partition * len(self.images))]
        else:
            self.images = self.images[int(train_partition * len(self.images)):]


        first_img_path = os.path.join(self.train_dir, self.images[0])
        first_img = io.imread(first_img_path)#just for shape 
        self.image_height = first_img.shape[0] 
        self.image_width = first_img.shape[1]


    def __len__(self):
        return len(self.images) 


    def __getitem__(self, idx):

        img_path = os.path.join(self.train_dir, self.images[idx])
        image = io.imread(img_path).astype('float32')/255.
        # image = image.transpose(2,0,1)

        df = self.labels
        search_key = os.path.splitext(self.images[idx])[0] + '.jpg'
        # print(search_key)
        
        rows = df[df['Image_Label'].str.match( search_key)]
        # print(rows)
        label=str_to_mask(df_rows=rows, height =  self.image_height, width = self.image_width) 
        
        # print('s ', label.sum())
        return image, label
