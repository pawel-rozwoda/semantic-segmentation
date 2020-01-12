from torch import nn

activation_f = nn.Tanh()

class SomeModel(nn.Module):

    def __init__(self):
        super(SomeModel,self).__init__()

        self.initial_block = nn.Sequential(
            nn.Conv2d(3, 16 , kernel_size=3),
            activation_f,
            nn.Conv2d(16 , 48, kernel_size=3),
            activation_f, 
            nn.Conv2d(48 , 168, kernel_size=3),
            activation_f, 
        ) 

        self.final_block = nn.Sequential(
            nn.ConvTranspose2d(168, 48,kernel_size=3),
            activation_f,
            nn.ConvTranspose2d(48, 16,kernel_size=3),
            activation_f,
            nn.ConvTranspose2d(16, 4 ,kernel_size=3),
            nn.Sigmoid()
        ) 

    def forward(self,x_):
        x = self.initial_block(x_)
        x = self.final_block(x)

        return x 
