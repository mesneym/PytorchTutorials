import torch
import torch.nn as nn


class Inception_Module(nn.Module):
    def __init__(self,in_c,c_1x1,c_1x1_3,c_3x3,c_1x1_5,c_5x5,c_m_1x1):
        super().__init__()

        self.branch1 = nn.Sequential(
                     nn.Conv2d(in_c,c_1x1,kernel_size=1 ,stride = 1, padding =0),
                     nn.ReLU()
                )

        self.branch2 = nn.Sequential(
                    nn.Conv2d(in_c,c_1x1_3,kernel_size=1, stride = 1 , padding =0),
                    nn.ReLU(),
                    nn.Conv2d(c_1x1_3,c_3x3,kernel_size=3, stride = 1 , padding = 1),
                    nn.ReLU()
                )
        self.branch3 = nn.Sequential(
                    nn.Conv2d(in_c,c_1x1_5,kernel_size=1, stride = 1, padding = 0),
                    nn.ReLU(),
                    nn.Conv2d(c_1x1_5,c_5x5,kernel_size = 5, stride=1 , padding = 2),
                    nn.ReLU()
                )
        
        self.branch4 = nn.Sequential(
                    nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
                    nn.ReLU(),
                    nn.Conv2d(in_c,c_m_1x1,kernel_size = 1,stride =1 ,padding = 0),
                    nn.ReLU()
                )


    def forward(self,x):
        return torch.cat([self.branch1(x),self.branch2(x),self.branch3(x),self.branch4(x)],1)


class GoogLeNet(nn.Module):
    def __init__(self,in_c,output):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv_layer1 = nn.Conv2d(in_c,64,kernel_size = 7,stride = 2, padding = 3)
        self.pool_layer1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding =1)

        self.conv_layer2 = nn.Conv2d(64,192,kernel_size = 3,stride =1 , padding =1)
        self.pool_layer2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.inception3a = Inception_Module(192,64,96,128,16,32,32)
        self.inception3b = Inception_Module(256,128,128,192,32,96,64)
        self.pool_layer3 = nn.MaxPool2d(kernel_size=3,stride= 2,padding = 1)

        self.inception4a = Inception_Module(480 ,192,96,208,16,48,64)
        self.inception4b = Inception_Module(512,160,112,224,24,64,64)
        self.inception4c = Inception_Module(512,128,128,256,24,64,64)
        self.inception4d = Inception_Module(512,112,144,288,32,64,64)
        self.inception4e = Inception_Module(528 ,256,160,320,32,128,128)
        self.pool_layer4 = nn.MaxPool2d(kernel_size=3, stride = 2,padding = 1)

        self.inception5a = Inception_Module(832, 256,160,320, 32, 128, 128)
        self.inception5b = Inception_Module(832, 384, 192, 384, 48, 128, 128)
        self.avg_layer5 = nn.AvgPool2d(kernel_size=7,stride=1)
        
        self.dropout = nn.Dropout(p=0.4)
        self.Fc = nn.Linear(1024,output)

    def forward(self,x):
        x = self.relu(self.conv_layer1(x))

        x = self.pool_layer1(x)
        x = self.relu(self.conv_layer2(x))
        x = self.pool_layer1(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool_layer3(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.pool_layer4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avg_layer5(x)
        
        x = self.dropout(x)
        x = x.reshape(x.shape[0],-1)
        x = self.Fc(x)
        
        return x

if __name__=='__main__':
    x = torch.rand(1,192,28,28)
    model = Inception_Module(192,64,96,128,16,32,32)
    print(model(x).shape)

    x = torch.rand(1,3,224,224)
    model = GoogLeNet(3,1000)
    print(model)

    print(model(x).shape)


