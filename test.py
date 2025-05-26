import torch

def evaluate(model,test_loader,device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs,labels in test_loader:
            inputs,labels = inputs.to(device),labels.to(device)
            outputs = model(inputs)
            _,predicted = torch.max(outputs,1)#在第二个维度中寻找最大值（Bacth,classes）,返回最大值、最大值所在的索引
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
    return correct / total
