import torch
from aux import str_to_mask 
import matplotlib.pyplot as plt
from PIL import Image
from load import MyDataset

TRAIN_PATH = "../data/train_images/"
LABELS = "../data/train.csv"
TRAIN_PARTITION = 0.8


dataset = MyDataset(train_dir=TRAIN_PATH, labels=LABELS, train=False, train_partition=TRAIN_PARTITION)


for sample, label in dataset:
    plt.imshow(sample)
    plt.imshow(label[0], alpha = 0.8)
    # plt.imshow(label[1], alpha = 0.8)
    # plt.imshow(label[2], alpha = 0.8)
    # plt.imshow(label[3], alpha = 0.8)

    plt.show()

