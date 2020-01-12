import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from load import MyDataset
from PIL import Image
import numpy as np
import pandas as pd
import os
from skimage import io
from aux import str_to_mask
TRAIN_DIR = '../data/sliced_train_images'
LABELS = '../data/sliced.csv'
x = MyDataset(train_dir=TRAIN_DIR,labels=LABELS, train=True, train_partition=0.8)


HIGH_RES_HEIGHT = 1400
HIGH_RES_WIDTH = 2100

SLICE_HEIGHT  = 140
SLICE_WIDTH = 140
# SLICE_HEIGHT  = 350
# SLICE_WIDTH = 350


files =  sorted(os.listdir(TRAIN_DIR))

df = pd.read_csv(LABELS)

blank_image = Image.new('RGB', (HIGH_RES_WIDTH, HIGH_RES_HEIGHT))
blank_mask = Image.new('RGB', (HIGH_RES_WIDTH, HIGH_RES_HEIGHT))

image_count = (HIGH_RES_HEIGHT//SLICE_HEIGHT) * (HIGH_RES_WIDTH//SLICE_WIDTH)
for i in range(image_count):
    # label_name = '0011165_' + str(i)
    label = df[df['Image_Label'].str.match("0011165_" + str(i )+ '.jpg')]
    
    
    
    # print(str_to_mask(df_rows=label, height=140, width=140).sum())
    label = np.array(str_to_mask(df_rows=label, height=SLICE_HEIGHT, width=SLICE_WIDTH)[0]).astype(np.uint8)
    # print(label.shape)
    img_path = os.path.join(TRAIN_DIR, '0011165_' + str(i)+'.jpg')
    image = io.imread(img_path).astype('float32')
    image = (image * 255 / np.max(image)).astype('uint8')
    # label = (label * 255 / np.max(label)).astype('uint8')
    label = label * 255
    

    i_pos = (i*SLICE_HEIGHT)%HIGH_RES_WIDTH

    j_pos = (i//(HIGH_RES_WIDTH//SLICE_HEIGHT))*SLICE_HEIGHT

    image = Image.fromarray(image) 
    label = Image.fromarray(label)
    blank_image.paste(image, (i_pos, j_pos))
    blank_mask.paste(label, (i_pos, j_pos))


plt.imshow(blank_image)
plt.imshow(blank_mask, alpha = 0.8)

# # plt.imshow(img.permute(1,2,0))
# # plt.imshow(label[1], alpha=0.2)
plt.show()


