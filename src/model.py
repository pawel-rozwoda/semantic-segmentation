from torch import nn
import torch
from torchsummary import summary

activation_f = nn.ReLU()

class SomeModel(nn.Module):

    def __init__(self):
        super(SomeModel,self).__init__()

        self.initial_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128 , kernel_size=3,padding = 1),
            activation_f,
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=256 , kernel_size=3,padding = 1),
            activation_f,
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=512 , kernel_size=3, padding=1),
            activation_f,
            nn.MaxPool2d(kernel_size=2)
        ) 

        self.final_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256 , kernel_size=3, stride=2),
            activation_f,
            nn.ConvTranspose2d(in_channels=256, out_channels=128 , kernel_size=2, stride=2),
            activation_f,
            nn.ConvTranspose2d(in_channels=128, out_channels=4 , kernel_size=2, stride=2),
            activation_f,
        ) 

    def forward(self,x_):
        x = self.initial_block(x_)
        x = self.final_block(x)

        return x 

# m=SomeModel()
# summary(m, (3,140,140))
