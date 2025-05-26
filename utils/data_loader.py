import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dataloader(batch_size = 32):
    # transform = transforms.Compose([
    #     transforms.RandomCrop(32,padding=4),#先对图像上下左右填充4个像素，再随机裁剪
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5),(0.5))
    # ])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data',train=True,
                                            download = True,transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data',train=False,
                                           download=True,transform=transform_test)
    train_loader = DataLoader(trainset,batch_size= batch_size,shuffle=True,num_workers=0)
    test_loader = DataLoader(testset,batch_size = batch_size,shuffle=False,num_workers=0)
    return train_loader,test_loader