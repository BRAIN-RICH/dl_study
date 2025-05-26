import torch.nn as nn
import torch.nn.functional as F

class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN,self).__init__()
        self.conv1 = nn.Conv2d(3,16,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.fc1 = nn.Linear(32*8*8,128)
        self.fc2 = nn.Linear(128,10)
    
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,32*8*8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,32,3,padding=1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(32,64,3,padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(2,2),
                                   nn.Dropout(0.25))
        self.conv3 = nn.Sequential(nn.Conv2d(64,128,3,padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(2,2),
                                   nn.Dropout(0.25))#为什么dropout在maxpool后面
        self.fc1 = nn.Sequential(nn.Linear(128*8*8,256),
                                 nn.BatchNorm1d(256),#BN的1d和2D有什么区别
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(0.5))
        self.fc2 = nn.Linear(256,10)
    
    def forward(self,x):
        x = self.conv1(x) # [B, 32, 32, 32]
        x = self.conv2(x) # [B, 64,16,16]
        x = self.conv3(x)# [B, 128, 8, 8]
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x