import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim



class VGG(nn.Module):
    def __init__(self, in_c, output, layers):
        super().__init__()

        self.conv_block = self.__conv_layers(in_c,layers)
        self.Fc = nn.Sequential(
                nn.Linear(7*7*512,4096),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(4096,4096),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(4096,output)
        )
        
    def forward(self,x):
        x = self.conv_block(x)
        x = x.reshape(x.shape[0],-1)
        x = self.Fc(x)
        return x

    def __conv_layers(self,in_channels,layers):
        conv_block = []
 
        in_c = in_channels 
        for l in layers:
            if(l != 'M'):
                conv_layer = [nn.Conv2d(in_channels = in_c,
                                       out_channels = l, 
                                       kernel_size = 3,
                                       stride = 1,
                                       padding = (1,1)), nn.ReLU()]

                conv_block += conv_layer
                in_c = l

            else:
                maxpool_layer = [nn.MaxPool2d(kernel_size=2,stride = 2),nn.ReLU()]
                                           
                conv_block += maxpool_layer

        return nn.Sequential(*conv_block)
             

if __name__ == '__main__':
    vgg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

    model = VGG(3,1000,vgg16)
    x = torch.rand(1,3,224,244)
    print(model(x).shape)






