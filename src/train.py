from load import MyDataset
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import torch
from torch import nn
import numpy as np
from model import SomeModel

# batch_size=512
batch_size=32
TRAIN_PARTITION = 0.8

TRAIN_PATH = '../data/sliced_train_images'
LABEL_PATH = '../data/sliced.csv'

train_dataset = MyDataset(train_dir=TRAIN_PATH, labels=LABEL_PATH, train=True, train_partition=TRAIN_PARTITION)

validation_dataset = MyDataset(train_dir=TRAIN_PATH, labels=LABEL_PATH, train=False, train_partition=TRAIN_PARTITION)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
print(len(train_dataset))
print(len(validation_dataset))



# activation_f = nn.ReLU()


model = SomeModel()

epochs = 1
lr = 0.0001

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)



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


for epoch in tqdm(range(epochs)):
    for step, (X, y) in enumerate(train_loader):
        y=y.type(torch.float)
        X = X.permute(0,3,1,2)

        pred = model.forward(X) 
        loss = soft_dice_loss(y, pred) 
        optimizer.zero_grad()   
        loss.backward()         
        optimizer.step()        



    with torch.no_grad():
        loss_v = []

        for X, y in validation_loader:
            y=y.type(torch.float)
            X = X.permute(0,3,1,2)
            pred = model.forward(X)
            loss_v.append(soft_dice_loss(y, pred).item())

        print(loss_v)
        print(np.mean(loss_v))
        loss_v = np.mean(loss_v)
        # loss_values.append((epoch, loss_v))

        print('epoka:, ', epoch)
        print('loss: ', loss_v)


# torch.save(model, "./model.pt")
torch.save(model.state_dict(), 'model_dict.pt')
