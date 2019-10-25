import torch
import argparse
from torchvision import datasets, transforms
import numpy as np
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from network import *
from utils import AllTripletSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector 
from losses import OnlineTripletLoss,OnlineTripletLoss1
from label_smooth import *
from sch import *
from netvlad import NetVLAD
parser = argparse.ArgumentParser(description='stylenet')
parser.add_argument('--lr', type=float, default='3e-5')
parser.add_argument('--num_in_features', type=int, default=128)
parser.add_argument('--margin', type=float, default='0.05')
parser.add_argument('--fe_model',type=str, choices=['fe','fe1','fe2','fe3','fe11','feature'])
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--resume',type=bool, default=False)
parser.add_argument('--beta',type=bool, default=False)
parser.add_argument('--loss',type=str, default='cxe')
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--sch',type=bool, default=False)
parser.add_argument('--gamma', type=int, default=0)
parser.add_argument('--trip_loss',type=str, default='ranking')
parser.add_argument('--vlad',type=bool, default=False)
opt = parser.parse_args()
print(opt)

# Define a transform to normalize the data
transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# Download and load the training data
train_dataset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Download and load the test data
test_dataset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

from datasets import BalancedBatchSampler

# We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
train_batch_sampler = BalancedBatchSampler(train_dataset, n_classes=10, n_samples=25)
test_batch_sampler = BalancedBatchSampler(test_dataset, n_classes=10, n_samples=25)
cuda = True
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

#-------------------
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
num_in_features = opt.num_in_features
if opt.fe_model=='fe1':
    embedding_net = FENet1(num_in_features)
elif opt.fe_model=='fe2':
    num_in_features = 64*6*6*2
    embedding_net = FENet2(num_in_features)
elif opt.fe_model=='fe3':
    embedding_net = FENet3(num_in_features)
elif opt.fe_model=='fe':
    embedding_net = FENet(num_in_features)
elif opt.fe_model=='fe11':
    embedding_net = FENet11(num_in_features)
elif opt.fe_model=='feature':
    embedding_net = Feature()
    num_in_features = 2048

model = embedding_net.to(device)
model_cl = ClassificationNet(num_in_features=num_in_features,num_out_classes=10).to(device)


optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
                model.parameters()), lr=opt.lr)
epoch = opt.epoch

if opt.sch:
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epoch)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=10,after_scheduler=scheduler_cosine)
else:
    scheduler_warmup = None

margin=opt.margin
if opt.loss=='cxe':
    criterion_cl = nn.CrossEntropyLoss()
elif opt.loss == 'smooth':
    criterion_cl = LabelSmoothingLoss(smoothing=opt.alpha)

if opt.trip_loss == 'ranking':
    criterion = OnlineTripletLoss(margin, HardestNegativeTripletSelector(margin)).to(device)
elif opt.trip_loss == 'impr':
    criterion = OnlineTripletLoss1(margin, HardestNegativeTripletSelector(margin)).to(device)
if opt.vlad:
    vlad = NetVLAD().to(device)
def train(model,model_cl,triplet_train_loader,optimizer,criterion,criterion_cl,epoch,beta,scheduler=None,vlad=None):
    model.train()
    model_cl.train()
    for batch_idx, (data, target) in enumerate(triplet_train_loader):
        #anchor, pos, neg = data
        #anchor = anchor.to(device)
        #pos = pos.to(device)
        #neg = neg.to(device)
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        
        #out1,out2,out3 = model(anchor,pos,neg)
        out1 = model(data)
        if vlad is not None:
            out1=vlad(out1)
            #print(out1.size())   # torch.Size([250, 2048])
        pred = model_cl(out1)
        #print(pred.size(),target.size())
        loss_cl = criterion_cl(pred,target)
        #loss = criterion(out1,out2,out3)
        loss = criterion(out1,target)
        if beta:
            #beta = np.min([0.33, (np.max([epoch-5,0])/100)**2])
             beta = np.min([0.33, (np.max([epoch-opt.gamma,0])/100)**2])
        else:
            beta = 0.33
        sum_loss = loss_cl+loss*beta
        sum_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}  c_loss: {:.4f} r_loss:{:.4f}'.format(
            epoch, batch_idx * len(data), len(triplet_train_loader.dataset),
                   100. * batch_idx / len(triplet_train_loader), sum_loss.item(),loss_cl.item(),loss.item()))

def val(model,model_cl,test_loader,optimizer,criterion,criterion_cl,vlad=None):
    model_cl.eval()
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            out1=model(data)
            if vlad is not None:
                out1=vlad(out1)
            output = model_cl(out1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#num = count_parameters(model)
#num_cl = count_parameters(model_cl)
#print('\n number of params {} \n'.format(num+num_cl))

for epoch in range(opt.epoch):
    train(model,model_cl,online_train_loader,optimizer,criterion,criterion_cl,epoch,opt.beta,scheduler_warmup,vlad)
    val(model,model_cl,test_loader,optimizer,criterion,criterion_cl,vlad)
    name = 'model'+opt.fe_model+str(opt.lr)+str(opt.margin)+str(opt.epoch)
    torch.save(model.state_dict(), './checkpoint/'+name+'.pth')
    torch.save(model_cl.state_dict(), './checkpoint/'+'cl_'+name+'.pth')



