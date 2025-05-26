import torchvision.models as models
import torch.nn as nn

def get_res18(num_classes =10):
  model = models.resnet18(pretrained = False)
  model.conv1 = nn.Conv2d(3,64,kernel_size=3,stride= 1,padding=1,bias=False)
  model.maxpool = nn.Identity() #删除maxpool,将maxpool替换为恒等层
  model.fc = nn.Linear(model.fc.in_features,num_classes)
  return model

def get_res34(num_classes =10):
  model = models.resnet34(pretrained = False)
  model.conv1 = nn.Conv2d(3,64,kernel_size=3,stride= 1,padding=1,bias=False)
  model.maxpool = nn.Identity() #删除maxpool,将maxpool替换为恒等层
  model.fc = nn.Linear(model.fc.in_features,num_classes)
  return model