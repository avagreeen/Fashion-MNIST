import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import pdb
import torch.nn.functional as F
import h5py
import deform_conv, deform_conv_v2
from netvlad import NetVLAD

class MyNet3(nn.Module):
    def __init__(self):
        super(MyNet3, self).__init__()
 
        self.base_model = nn.Sequential(nn.Conv2d(1, 64, kernel_size=4,padding=2), 
                                        nn.BatchNorm2d(64),
                                        nn.PReLU(),
                                        nn.MaxPool2d(kernel_size=2),
                                        nn.Dropout2d(0.1),
                                        nn.Conv2d(64,64, kernel_size=4,padding=1,dilation=1),
                                        nn.BatchNorm2d(64),
                                        nn.PReLU(),
                                        nn.MaxPool2d(kernel_size=2),
                                        nn.Dropout2d(0.3),
                                        #deform_conv.DeformConv2D(128, 128, kernel_size=3, padding=1),
                                        )
        self.fc = nn.Sequential(nn.Linear(6*6*64, 512),
                                nn.PReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(512,64),
                                nn.PReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(64,10),
                                )
        #self.fc2 = nn.Linear(7*7*128, 512)
        #self.fc1 = nn.Linear(512,10)
    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = F.relu(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

class MyNet4(nn.Module):
    def __init__(self):
        super(MyNet4, self).__init__()
 
        self.base_model = nn.Sequential(nn.Conv2d(1, 64, kernel_size=4,padding=2), 
                                        nn.BatchNorm2d(64),
                                        nn.PReLU(),
                                        nn.MaxPool2d(kernel_size=2),
                                        nn.Dropout2d(0.1),
                                        nn.Conv2d(64,128, kernel_size=4,padding=1,dilation=1),
                                        nn.BatchNorm2d(128),
                                        nn.PReLU(),
                                        nn.MaxPool2d(kernel_size=2),
                                        nn.Dropout2d(0.3),
                                        #deform_conv.DeformConv2D(128, 128, kernel_size=3, padding=1),
                                        )
        self.fc = nn.Sequential(nn.Linear(6*6*128, 512),
                                nn.PReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(512,64),
                                nn.PReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(64,10),
                                )
        #self.fc2 = nn.Linear(7*7*128, 512)
        #self.fc1 = nn.Linear(512,10)
    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = F.relu(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
 
        self.base_model = nn.Sequential(nn.Conv2d(1, 64, kernel_size=4,padding=2), 
                                        nn.BatchNorm2d(64),
                                        nn.PReLU(),
                                        nn.MaxPool2d(kernel_size=2),
                                        nn.Dropout2d(0.1),
                                        nn.Conv2d(64,64, kernel_size=4,padding=1,dilation=1),
                                        nn.BatchNorm2d(64),
                                        nn.PReLU(),
                                        #nn.MaxPool2d(kernel_size=2),
                                        nn.Dropout2d(0.3),
                                        #deform_conv.DeformConv2D(128, 128, kernel_size=3, padding=1),
                                        )

    def forward(self, x):
        x = self.base_model(x)
        return x

class MyNet5(nn.Module):
    def __init__(self,num_in_features):
        super(MyNet5, self).__init__()
 
        self.base_model = nn.Sequential(nn.Conv2d(1, 64, kernel_size=4,padding=2), 
                                        nn.BatchNorm2d(64),
                                        nn.PReLU(),
                                        nn.MaxPool2d(kernel_size=2),
                                        nn.Dropout2d(0.1),
                                        nn.Conv2d(64,64, kernel_size=4,padding=1,dilation=1),
                                        nn.BatchNorm2d(64),
                                        nn.PReLU(),
                                        nn.MaxPool2d(kernel_size=2),
                                        nn.Dropout2d(0.3),
                                        #deform_conv.DeformConv2D(128, 128, kernel_size=3, padding=1),
                                        )
        self.fc1 = nn.Linear(6*6*64, num_in_features)

        self.fc = nn.Sequential(
                                nn.PReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(num_in_features,64),
                                nn.PReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(64,10),
                                )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        vec = self.fc1(x)
        #print(x.size())
        x = self.fc(vec)
        x = F.log_softmax(x, dim=1)
        return vec,x
class ZNet(nn.Module):
    def __init__(self):
        super(ZNet, self).__init__()
 
        self.base_model = nn.Sequential(nn.Conv2d(1, 64, kernel_size=4,padding=2), 
                                        #nn.BatchNorm2d(64),
                                        nn.PReLU(),
                                        nn.BatchNorm2d(64),
                                        nn.MaxPool2d(kernel_size=2),
                                        nn.Dropout2d(0.1),
                                        nn.Conv2d(64, 64, kernel_size=4,padding=1),
                                        #nn.BatchNorm2d(64),
                                        nn.PReLU(),
                                        nn.BatchNorm2d(64),
                                        nn.MaxPool2d(kernel_size=2),
                                        nn.Dropout2d(0.3),
                                        )
        self.fc = nn.Sequential(nn.Linear(6*6*64, 256),
                                nn.PReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(256,64),
                                nn.PReLU(),
                                nn.Linear(64,10),
                                )
    def forward(self, x):
        x = self.base_model(x)
        #print(x.size())
        x = x.view(-1, 6*6 *64)
        x = F.relu(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
class ZNet1(nn.Module):
    def __init__(self):
        super(ZNet1, self).__init__()
 
        self.base_model = nn.Sequential(nn.Conv2d(1, 64, kernel_size=4,padding=2), 
                                        nn.BatchNorm2d(64),
                                        nn.PReLU(),
                                        #nn.BatchNorm2d(64),
                                        nn.MaxPool2d(kernel_size=2),
                                        nn.Dropout2d(0.1),
                                        nn.Conv2d(64, 64, kernel_size=4,padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.PReLU(),
                                        #nn.BatchNorm2d(64),
                                        nn.MaxPool2d(kernel_size=2),
                                        nn.Dropout2d(0.3),
                                        )
        self.fc = nn.Sequential(nn.Linear(6*6*64, 256),
                                nn.PReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(256,64),
                                nn.PReLU(),
                                nn.Linear(64,10),
                                )
    def forward(self, x):
        x = self.base_model(x)
        #print(x.size())
        x = x.view(-1, 6*6 *64)
        x = F.relu(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


class MyAttenNet(nn.Module):
    def __init__(self):
        super(MyAttenNet, self).__init__()
 
        self.base_model = nn.Sequential(nn.Conv2d(1, 64, kernel_size=4,padding=2), 
                                        nn.BatchNorm2d(64),
                                        nn.PReLU(),
                                        nn.MaxPool2d(kernel_size=2),
                                        nn.Dropout2d(0.1),
                                        deform_conv.DeformConv2D(64, 128, kernel_size=4,padding=1),
                                        nn.BatchNorm2d(128),
                                        nn.PReLU(),
                                        nn.MaxPool2d(kernel_size=2),
                                        nn.Dropout2d(0.3),
                                        #deform_conv.DeformConv2D(128, 128, kernel_size=3, padding=1),
                                        )
        self.atten = nn.Sequential(nn.Conv2d(128,1,1,1),
                                   nn.Softplus(beta=1, threshold=20)

                                   )
        self.fc = nn.Sequential(nn.Linear(7*7*128, 512),
                                nn.PReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(512,64),
                                nn.PReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(64,10),
                                )
        #self.fc2 = nn.Linear(7*7*128, 512)
        #self.fc1 = nn.Linear(512,10)
    def forward(self, x):
        x = self.base_model(x)
        #print(x.size())
        atten = self.atten(x)
        x = x*atten
        #print(x.size())
        x = x.view(-1, 7* 7 * 128)
        x = F.relu(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

class FENet(nn.Module):
    def __init__(self, num_features=128):
        super(FENet, self).__init__()
        self.conv_a = nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1))
        self.conv_b = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        #self.conv_c = nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1))
        #self.conv_d = nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1))
        #self.conv_e = nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1))
        #self.conv_f = nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1))
        #self.conv_g = nn.Conv2d(256, 128, (3, 3), (1, 1), (1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        self.maxpool = nn.MaxPool2d(2)
        self.batchnorm_a = nn.BatchNorm2d(64)
        self.batchnorm_b = nn.BatchNorm2d(128)
        self.batchnorm_c = nn.BatchNorm2d(256)
        self.atten = nn.Sequential(nn.Conv2d(128,1,1,1),nn.Softplus(beta=1, threshold=20))
        self.fc = nn.Linear(64*7*7, num_features)

    def forward(self, input):
        x = self.relu(self.conv_a(input))
        x = self.maxpool(x)
        x = self.batchnorm_b(x)
        x = self.relu(self.conv_c(x))
        x = self.maxpool(self.dropout(x))
        #x = self.relu(self.conv_c(x))
        #x = self.relu(self.conv_d(x))
        #x = self.maxpool(self.dropout(x))
        #x = self.batchnorm_b(x)
        #x = self.relu(self.conv_e(x))
        #x = self.relu(self.conv_f(x))
        #x = self.maxpool(self.dropout(x))
        #x = self.batchnorm_c(x)
        #x = self.relu(self.conv_g(x))
        #print(x.size())
        atten = self.atten(x)
        x = x*atten
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        #x = F.relu(x)
        x = F.normalize(x,p=2,dim=1)
        return x
class FENet1(nn.Module):
    def __init__(self, num_features=128):
        super(FENet1, self).__init__()
        self.base_model = nn.Sequential(nn.Conv2d(1, 64, kernel_size=4,padding=2), 
                                       #nn.BatchNorm2d(64),
                                        nn.PReLU(),
                                        nn.BatchNorm2d(64),
                                        nn.MaxPool2d(kernel_size=2),
                                        nn.Dropout2d(0.1),
                                        nn.Conv2d(64, 64, kernel_size=4,padding=1),
                                        #nn.BatchNorm2d(64),
                                        nn.PReLU(),
                                        nn.BatchNorm2d(64),
                                        nn.MaxPool2d(kernel_size=2),
                                        nn.Dropout2d(0.3),
                                        #deform_conv.DeformConv2D(128, 128, kernel_size=3, padding=1),
                                        )
        #self.conv_a = nn.Conv2d(1, 64, kernel_size=4,padding=2)
        #self.conv_c = nn.Conv2d(64, 64, kernel_size=4,padding=1)
        #self.relu = nn.ReLU(inplace=True)
        #self.dropout = nn.Dropout(0.1)
        #self.maxpool = nn.MaxPool2d(kernel_size=2)
        #self.batchnorm_a = nn.BatchNorm2d(64)
        #self.batchnorm_b = nn.BatchNorm2d(128)

        #self.atten = nn.Sequential(nn.Conv2d(64,1,1,1),nn.Softplus(beta=1, threshold=20))
        self.fc = nn.Linear(64*6*6, num_features)

    def forward(self, input):
        x = self.base_model(input)
        #atten = self.atten(x)
        #x = x*atten
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x,p=2,dim=1)
        return x
class FENet11(nn.Module):
    def __init__(self, num_features=128):
        super(FENet11, self).__init__()
        self.base_model = nn.Sequential(nn.Conv2d(1, 64, kernel_size=4,padding=2), 
                                        nn.BatchNorm2d(64),
                                        nn.PReLU(),
                                        #nn.BatchNorm2d(64),
                                        nn.MaxPool2d(kernel_size=2),
                                        nn.Dropout2d(0.1),
                                        nn.Conv2d(64, 64, kernel_size=4,padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.PReLU(),
                                        #nn.BatchNorm2d(64),
                                        nn.MaxPool2d(kernel_size=2),
                                        nn.Dropout2d(0.3),
                                        #deform_conv.DeformConv2D(128, 128, kernel_size=3, padding=1),
                                        )
        self.conv_a = nn.Conv2d(1, 64, kernel_size=4,padding=2)
        self.conv_c = nn.Conv2d(64, 64, kernel_size=4,padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.batchnorm_a = nn.BatchNorm2d(64)
        self.batchnorm_b = nn.BatchNorm2d(128)

        #self.atten = nn.Sequential(nn.Conv2d(64,1,1,1),nn.Softplus(beta=1, threshold=20))
        self.fc = nn.Linear(64*6*6, num_features)

    def forward(self, input):
        x = self.base_model(input)
        #atten = self.atten(x)
        #x = x*atten
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x,p=2,dim=1)
        return x
class FENet2(nn.Module):
    def __init__(self, num_features=128):
        super(FENet2, self).__init__()
        self.base_model = nn.Sequential(nn.Conv2d(1, 64, kernel_size=4,padding=2), 
                                        nn.PReLU(),
                                        nn.BatchNorm2d(64),
                                        nn.MaxPool2d(kernel_size=2),
                                        nn.Dropout2d(0.1),
                                        nn.Conv2d(64, 64, kernel_size=4,padding=1),
                                        nn.PReLU(),
                                        nn.BatchNorm2d(64),
                                        nn.Dropout2d(0.3),
                                        )
       
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        self.batchnorm_a = nn.BatchNorm2d(64)
        self.batchnorm_b = nn.BatchNorm2d(128)

        self.atten = nn.Sequential(nn.Conv2d(64,1,1,1),nn.Softplus(beta=1, threshold=20))
        #self.fc = nn.Linear(64*6*6, num_features)

    def forward(self, input):
        x = self.base_model(input)
        atten = self.atten(x)
        x = x*atten
        x = x.view(x.size(0), -1)
        #x = self.fc(x)
        x = F.normalize(x,p=2,dim=1)
        return x
class FENet3(nn.Module):
    def __init__(self, num_features=128):
        super(FENet3, self).__init__()
        self.base_model = nn.Sequential(nn.Conv2d(1, 64, kernel_size=4,padding=2), 
                                        nn.PReLU(),
                                        nn.BatchNorm2d(64),
                                        nn.MaxPool2d(kernel_size=2),
                                        nn.Dropout2d(0.1),
                                        nn.Conv2d(64, 64, kernel_size=4,padding=1),
                                        nn.PReLU(),
                                        nn.BatchNorm2d(64),
                                        nn.MaxPool2d(kernel_size=2),
                                        nn.Dropout2d(0.3),
                                        )


        self.atten = nn.Sequential(nn.Conv2d(64,1,1,1),nn.Softplus(beta=1, threshold=20))
        self.fc = nn.Linear(64*6*6, num_features)

    def forward(self, input):
        x = self.base_model(input)
        atten = self.atten(x)
        x = x*atten
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x,p=2,dim=1)
        return x

class ClassificationNet(nn.Module):
    def __init__(self, num_in_features=128, num_out_classes=10):
        super(ClassificationNet, self).__init__()
        #self.batchnorm = nn.BatchNorm1d(num_in_features)
        self.relu = nn.ReLU(inplace=True)
        self.hidden = nn.Linear(num_in_features, num_in_features)
        self.classifier = nn.Sequential(nn.Linear(num_in_features, 512),
                                nn.PReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(512,10),
                                )

    def forward(self, input):
        x = self.hidden(self.relu(input))
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x
class ClassificationNet1(nn.Module):
    def __init__(self, num_in_features=128, num_out_classes=10):
        super(ClassificationNet1, self).__init__()
        #self.batchnorm = nn.BatchNorm1d(num_in_features)
        self.relu = nn.PReLU()
        self.hidden = nn.Linear(num_in_features, num_in_features)
        self.classifier = nn.Sequential(nn.Linear(num_in_features, int(num_in_features/2)),
                                nn.PReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(int(num_in_features/2),10),
                                )

    def forward(self, input):
        #x = self.batchnorm(input)
        x = self.hidden(self.relu(x))
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x
def MaxAvgPool(input):
    maxp = nn.MaxPool2d(kernel_size=2)
    avgp = nn.AvgPool2d(kernel_size=2)
    maxp = maxp(input)
    avgp = avgp(input)
    out = (maxp+avgp)/2.0
    return out

class DiNet(nn.Module):
    def __init__(self, num_features=128):
        super(DiNet, self).__init__()
        self.conv_a = nn.Conv2d(1, 64, kernel_size=4,padding=2)
        self.conv_b = nn.Conv2d(64, 64,  kernel_size=4,padding=1)
        #self.conv_c = nn.Conv2d(64, 128,  kernel_size=3,stride=1,padding=1,dilation=1)
        #self.conv_d = nn.Conv2d(128, 128,  kernel_size=3,stride=1,padding=1,dilation=1)
        self.relu = nn.PReLU()
        self.dropout = nn.Dropout(0.1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.batchnorm_a = nn.BatchNorm2d(64)
        self.batchnorm_b = nn.BatchNorm2d(128)

        #self.atten = nn.Sequential(nn.Conv2d(128,1,1,1),nn.Softplus(beta=1, threshold=20))
        self.fc = nn.Sequential(nn.Linear(64*6*6, 128),
                                nn.PReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(128,64),
                                nn.PReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(64,10),
                                )
        self.base_model = nn.Sequential(nn.Conv2d(1, 64, kernel_size=4,padding=2), 
                                        #nn.BatchNorm2d(64),
                                        nn.PReLU(),
                                        nn.BatchNorm2d(64),
                                        nn.MaxPool2d(kernel_size=2),
                                        nn.Dropout2d(0.1),
                                        nn.Conv2d(64, 64, kernel_size=4,padding=1),
                                        #nn.BatchNorm2d(64),
                                        nn.PReLU(),
                                        nn.BatchNorm2d(64),
                                        nn.MaxPool2d(kernel_size=2),
                                        nn.Dropout2d(0.3),
                                        #deform_conv.DeformConv2D(128, 128, kernel_size=3, padding=1),
                                        )

    def forward(self, input):
        x = self.relu(self.conv_a(input))
        x = self.batchnorm_a(x)
        #print(x.size())
        x = MaxAvgPool(x)
        #print(x.size())
        x = self.dropout(x)
        x = self.relu(self.conv_b(x))
        x = self.batchnorm_a(x)
        x = MaxAvgPool(x)
        x = self.dropout(x)
        #x = self.relu(x)
        #x = self.maxpool(self.dropout(x))
        #x = self.batchnorm_a(self.conv_b(x))
        #x = self.relu(x)
        #x = self.maxpool(self.dropout(x))
        #x = self.base_model(input)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        
        #print(x.size())
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
