import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.optim as optim
import torch.nn as nn

from models.basic_cnn import BasicCNN,ImprovedCNN
from models.resnet import get_res18,get_res34
from utils.data_loader import get_dataloader
from test import evaluate
from train import train
import time

from torch.optim.lr_scheduler import CosineAnnealingLR


@torch.no_grad()
def evaluate_loss(model,data_loader,criterion,device):
    model.eval()
    total_loss = 0
    total_samples = 0
    for x,y in data_loader:
        x,y = x.to(device),y.to(device)
        logits = model(x)
        loss = criterion(logits,y)
        total_loss += loss.item()* x.size(0)
        total_samples += x.size(0)
    return total_loss / total_samples


@torch.no_grad()
def compute_acc(model,data_loader,device):
    model.eval()
    correct = 0
    total = 0
    for x,y in data_loader:
        x,y = x.to(device),y.to(device)
        logits = model(x)
        preds = torch.argmax(logits,dim=1)
        correct += (preds==y).sum().item()
        total += y.size(0)
    return correct / total

if __name__  == '__main__':

    '''
    1、基础两层CNN，无数据增强，固定学习率无label smoothing,精度70.7%
    2、5层CNN（加入BN），有简单数据增强，固定学习率，有label smoothing,有dropout,batch_size 64
        在80轮后将lr从1e-3调整到1e-4，精度从85.52到86.16,model_4
    3、换网络结构为resnet18,有复杂数据增强，新增weight_decay,batch_size为128,新增cosine学习率调度
        训练一轮7min,训练28轮，有轻微过拟合，loss反而上升，精度最好91.96%
        换到RTX4090训练，精度变成93.44%,一轮13s
    4、换网络结构为res34,label_smooth从0.1-0.05，batchsize从128-256，t_max:20->100

    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model = BasicCNN().to(device)
    # model = ImprovedCNN().to(device)
    model = get_res34().to(device)
    if os.path.exists("best_res_model_1.pth"):
        model.load_state_dict(torch.load("best_res_model_1.pth"))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.Adam(model.parameters(),lr = 1e-3,weight_decay=1e-4)
    train_loader,test_loader = get_dataloader(batch_size=128)

    best_acc = 0

    patience =5
    no_improve_count = 0

    scheduler = CosineAnnealingLR(optimizer,T_max=20)

    for epoch in range(100):
        start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        train_loss = train(model,train_loader,criterion,optimizer,device)
        train_acc = compute_acc(model,train_loader,device)

        test_acc = evaluate(model,test_loader,device)
        test_loss = evaluate_loss(model,test_loader,criterion,device)
        scheduler.step()
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),'best_res_model_2.pth')
            no_improve_count = 0
        else:
            no_improve_count += 1
        print(f"Epoch {epoch +1},LR={current_lr:.6f},\
              train loss={train_loss:.4f},train acc={train_acc:.2%},\
              Test acc={test_acc:2%},test loss = {test_loss:.4f}  time:{time.time() - start_time}")

        if no_improve_count >= patience:
            print("early stoppping triggered at epoch {epoch}")
            break


    print('train complete')

    