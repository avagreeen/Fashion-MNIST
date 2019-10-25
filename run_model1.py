import torch
from torchvision import datasets, transforms
#import helper
import argparse
# Define a transform to normalize the data
transform = transforms.Compose([transforms.RandomRotation(degrees=10),
                                transforms.Resize(28),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# Download and load the training data
trainset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

import numpy as np
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import network
from label_smooth import *
from sch import *
parser = argparse.ArgumentParser(description='stylenet')
parser.add_argument('--lr', type=float, default='3e-5')
parser.add_argument('--model',type=str,default='mynet4')
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--resume',type=bool, default=False)
parser.add_argument('--loss',type=str, default='cxe')
parser.add_argument('--optim',type=str, default='sgd')
parser.add_argument('--sch',type=bool, default=False)
parser.add_argument('--alpha', type=float, default=0.1)
opt = parser.parse_args()
print(opt)


use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
if opt.model=='di':
    model = network.DiNet().to(device)
elif opt.model =='znet':
    model = network.ZNet().to(device)
elif opt.model =='znet1':
    model = network.ZNet1().to(device)
elif opt.model =='mynet3':
    model = network.MyNet3().to(device)
elif opt.model =='mynet4':
    model = network.MyNet4().to(device)

if opt.optim=='sgd':
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, 
                model.parameters()), lr=opt.lr,
                momentum=0.9,
                weight_decay=0.001)
elif opt.optim=='adam':
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
                model.parameters()), lr=1e-4)
if opt.sch:
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epoch)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=10,after_scheduler=scheduler_cosine)
else:
    scheduler_warmup = None

if opt.loss=='cxe':
    criterion = nn.CrossEntropyLoss()
elif opt.loss == 'smooth':
    criterion = LabelSmoothingLoss(smoothing=opt.alpha)

def train(model,train_loader, optimizer, epoch,device,criterion,scheduler=None):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,target)
        loss.backward()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output,target).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num = count_parameters(model)

print('\n number of params {} \n'.format(num))

for epoch in range(opt.epoch):
    model.train()
    train(model,train_loader,optimizer,epoch,device,criterion,scheduler_warmup)
    model.eval()
    test(model, device, test_loader)


