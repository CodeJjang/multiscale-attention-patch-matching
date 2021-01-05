import torch
from torch.utils import data
import numpy as np
import torch.nn.functional as F
from torch.nn import Conv2d, Linear,MaxPool2d, BatchNorm2d, Dropout
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from mpl_toolkits.axes_grid1 import ImageGrid
from torchsummary import summary
from skimage.transform import resize
import copy
from network.spp_layer import spatial_pyramid_pool


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x

class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sum(torch.aBatchSize(x), dim = 1) + self.eps
        x= x / norm.expand_as(x)
        return x







class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.MaxPool2 = nn.MaxPool2d(kernel_size = 2, stride=2)
        self.MaxPool4 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=2, bias=False),  # stride = 2
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=2, bias=False),  # stride = 2  wrong:10368
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),  # BatchSize, 128,8,8
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )

        self.output_num = [8, 4, 2, 1]
        self.output_num = [8]

        self.fc1 = nn.Sequential(
            #nn.Linear(10880, 128),
            nn.Linear(8192, 128),
        )

        return

    def input_norm(self, x):
        flat = x.reshape(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(
            -1).unsqueeze(-1).unsqueeze(1).expand_as(x)


    def FreezeCnn(self, OnOff):
        for param in self.parameters():
            param.requires_grad = not OnOff


    def FreezeBlock(self, OnOff):
        for param in self.block.parameters():
            param.requires_grad = not OnOff


    def forward(self, input1,Mode = 'Normalized',ActivMode =False,DropoutP=0):
        BatchSize = input1.size(0)
        feat = self.block(self.input_norm(input1))
        spp_a = spatial_pyramid_pool(feat, BatchSize, [int(feat.size(2)), int(feat.size(3))], self.output_num)

        if Mode == 'NoFC':
            return spp_a

        spp_a = Dropout(DropoutP)(spp_a)  # 20% probability

        feature_a = self.fc1(spp_a).reshape(BatchSize, -1)

        if Mode ==  'Normalized':
            #return L2Norm()(feature_a)
            if ActivMode:
                return F.normalize(feature_a, dim=1, p=2),feat
            else:
                return F.normalize(feature_a, dim=1, p=2)
        else:
            if ActivMode:
                return feature_a,feat
            else:
                return feature_a


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, gain=0.3)
        try:
            nn.init.constant(m.bias.data, 0.01)
        except:
            pass
    return
