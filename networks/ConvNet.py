import torch
import torch.nn as nn
from spp_layer import spatial_pyramid_pool
from layers import L2Norm


class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()

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

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),  # bs, 128,8,8
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )

        self.output_num = [8, 4, 2, 1]

        self.fc1 = nn.Sequential(
            nn.Linear(10880, 128),
        )
        return

    def input_norm(self, x):
        # flat = x.view(x.size(0), -1)
        flat = x.reshape(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(
            -1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def FreezeCnn(self, OnOff):
        for param in self.parameters():
            param.requires_grad = not OnOff

    def forward(self, input1, Mode='Normalized'):
        bs = input1.size(0)
        feat = self.block(self.input_norm(input1))
        spp_a = spatial_pyramid_pool(feat, bs, [int(feat.size(2)), int(feat.size(3))], self.output_num)

        if Mode == 'NoFC':
            return spp_a

        feature_a = self.fc1(spp_a).view(bs, -1)

        if Mode == 'Normalized':
            return L2Norm()(feature_a)
        else:
            return feature_a
