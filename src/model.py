import torch
import torch.nn as nn
import torch.optim as optim
import torchvision


class NW_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
 
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
 
        self.flat = nn.Flatten()
 
        self.fc3 = nn.Linear(33408, 1080)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)
 
        self.fc4 = nn.Linear(1080,  6)
        self.logSoftmax = nn.LogSoftmax(dim=1)
 
    def forward(self, x):
        # input 1x77x62, output 32x75x60
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        # # input 32x75x60, output 32x73x58
        x = self.act1(self.conv2(x))
        # # input 32x73x58, output 32x36x29
        x = self.pool2(x)
        # input 32x36x29, output 8192
        x = self.flat(x)
        # # input 8192, output 512
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        # # input 512, output 10
        x = self.fc4(x)
        x = self.logSoftmax(x)
        return x