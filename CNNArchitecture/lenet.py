import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F


class Lenet(nn.Module):
    def __init__(self,in_c,layers):
        super().__init__()
        
        self.relu = nn.ReLU()

        self.conv_layer1 = nn.Conv2d(in_c,layers[0],kernel_size =5,stride = 1, padding = 0)
        self.avg_pool = nn.AvgPool2d(kernel_size=2,stride=2)
        self.conv_layer2 = nn.Conv2d(layers[1],layers[2],kernel_size = 5,stride = 1, padding = 0)
        self.conv_layer3 = nn.Conv2d(layers[2],layers[3],kernel_size = 5,stride = 1, padding = 0)
        self.Fc1 = nn.Linear(120,84) 
        self.Fc2 = nn.Linear(84,10) 
     
    def forward(self, x):
         x = self.conv_layer1(x)
         x = self.relu(x)
         x = self.avg_pool(x)
         x = self.conv_layer2(x)
         x = self.relu(x)
         x = self.avg_pool(x)
         x = self.conv_layer3(x)  
         x = self.relu(x)
         x = x.reshape(x.shape[0],-1)  
         x = self.Fc1(x)
         x = self.relu(x)
         x = self.Fc2(x)
         return x

if __name__ == '__main__':
    layers = [6,6,16,120]
    model = Lenet(1,layers)
    print(model)
    x = torch.randn(1,1,32,32)
    print(model(x).shape)

    





