from __future__ import print_function, division
import os
import torch
import pandas as pd 
from skimage import io
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import matplotlib
from aux import str_to_mask 

class MyDataset(Dataset):
    def __init__(self, *, train_dir, labels , transform=None, train, train_partition, do_not_scale=False):
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
        
        self.labels = self.labels.sort_values(by='Image_Label')
        self.do_not_scale = do_not_scale


    def __len__(self):
        return len(self.images) 


    def __getitem__(self, idx):

        img_path = os.path.join(self.train_dir, self.images[idx])
        image = io.imread(img_path).astype('float32')/255.
        image = torch.Tensor(image)

        if not self.do_not_scale:
            image = image.unsqueeze(0) 
            image = image.permute(0,3,1,2)
            image = torch.nn.functional.interpolate(image,size=(128,128), mode='nearest')
            image = image.squeeze(0)

        else:
            image = image.permute(2,0,1)
        # print('image', image.shape)

        df = self.labels
        search_key = os.path.splitext(self.images[idx])[0] + '.jpg'
        
        row_fish = df['Image_Label'].searchsorted(search_key+ "_Fish",'left')
        row_flower = df['Image_Label'].searchsorted(search_key+ "_Flower",'left')
        row_gravel = df['Image_Label'].searchsorted(search_key+ "_Gravel",'left')
        row_sugar = df['Image_Label'].searchsorted(search_key+ "_Sugar",'left')

        rows = pd.concat([df[row_fish:row_fish+1],
                        df[row_flower:row_flower+1],
                        df[row_gravel:row_gravel+1],
                        df[row_sugar:row_sugar+1]])
 
        label=str_to_mask(df_rows=rows, height =  self.image_height, width = self.image_width) 
        label=torch.Tensor(label)
    
        if not self.do_not_scale:
            label = label.unsqueeze(0)
            label = torch.nn.functional.interpolate(label,size=(128,128), mode='nearest')
            label = label.squeeze(0)

        # print('label', label.shape)
        # print('-----')
        # print(torch.sum(label==0.) + torch.sum(label==1))
        # print(np.prod(label.shape))
        # print('-----')
        # print()

        
        return image, label
