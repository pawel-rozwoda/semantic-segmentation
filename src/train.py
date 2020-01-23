from load import MyDataset
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from aux import soft_dice_loss
from datetime import datetime
import os
import shutil

batch_size=32
TRAIN_PARTITION = 0.8

now = datetime.now()
model_path = now.strftime("model_%m|%d|%Y--%H:%M:%S/")
if os.path.exists(model_path):
    shutil.rmtree(model_path)
    
os.makedirs(model_path)

TRAIN_PATH = '../data/sliced_train_images'
LABEL_PATH = '../data/sliced.csv'
# TRAIN_PATH = '../data/small_sliced_train_images'
# LABEL_PATH = '../data/small_sliced.csv'

train_dataset = MyDataset(train_dir=TRAIN_PATH, labels=LABEL_PATH, train=True, train_partition=TRAIN_PARTITION) 
validation_dataset = MyDataset(train_dir=TRAIN_PATH, labels=LABEL_PATH, train=False, train_partition=TRAIN_PARTITION)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=4, init_features=32)

model = model.to(device)

epochs = 10
initial_lr = 0.008

optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

bce = nn.BCELoss() 

for epoch in tqdm(range(epochs)):
    print('lr:', scheduler.get_lr())
    for step, (X, y) in tqdm(enumerate(train_loader)):
        y=y.type(torch.float) 
        X = X.to(device)
        y = y.to(device) 

        pred = model.forward(X) 
        loss = soft_dice_loss(pred, y) + bce(pred, y) 
        optimizer.zero_grad()   
        loss.backward()         
        optimizer.step()        


    test_losses = []
    train_losses = []
    with torch.no_grad():
        dice_loss = [] 
        bce_loss = []
        loss_v = []
        for X, y in validation_loader:
            # y=y.type(torch.float)
            X = X.to(device)
            y = y.to(device)
            pred = model.forward(X)
            loss_v.append(soft_dice_loss(pred, y).item() + bce(pred, y).item())
            dice_loss.append(soft_dice_loss(pred,y).item())
            bce_loss.append(bce(pred,y).item())


        dice_loss = np.mean(dice_loss)
        bce_loss = np.mean(bce_loss)
        loss_v = np.mean(loss_v)
        test_losses.append((epoch, loss_v, dice_loss, bce_loss))

        
    with torch.no_grad(): 
        dice_loss = [] 
        bce_loss = []
        loss_v = []
        for X, y in train_loader:
            # y=y.type(torch.float)
            X = X.to(device)
            y = y.to(device)
            pred = model.forward(X)
            loss_v.append(soft_dice_loss(pred, y).item() + bce(pred, y).item())
            dice_loss.append(soft_dice_loss(pred,y).item())
            bce_loss.append(bce(pred,y).item())


        dice_loss = np.mean(dice_loss)
        bce_loss = np.mean(bce_loss)
        loss_v = np.mean(loss_v)
        train_losses.append((epoch, loss_v, dice_loss, bce_loss))


    with open(model_path+"test_losses.csv", "a") as myfile:
        myfile.write(','.join(map(str, test_losses)) + '\n')

    with open(model_path+"train_losses.csv", "a") as myfile:
        myfile.write(','.join(map(str, train_losses)) + '\n')

    scheduler.step() 


    torch.save(model.state_dict(), model_path+"model_epoch_" + str(epoch) +"_dict.pt") 

# test_df = pd.DataFrame(test_losses, columns=['epoch', 'total_loss', 'dice_loss', 'bce_loss'])  
# train_df = pd.DataFrame(train_losses, columns=['epoch', 'total_loss', 'dice_loss', 'bce_loss'])  

# test_df.to_csv(model_path+'test_loss.csv', index=False)
# train_df.to_csv(model_path+'train_loss.csv', index=False)
