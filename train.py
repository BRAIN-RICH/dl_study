import torch 

def train(model,train_loder,criterion,optimizer,device):
    model.train()
    running_loss =0 
    for inputs,labels in train_loder:
        inputs,labels = inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()#提取loss值
    return running_loss / len(train_loder)