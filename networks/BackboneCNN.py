import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout

from layers.spp_layer import spatial_pyramid_pool


class BackboneCNN(nn.Module):

    def __init__(self):
        super(BackboneCNN, self).__init__()

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

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),  # batch_size, 128,8,8
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )

        # self.output_num = [8, 4, 2, 1]
        self.output_num = [8]

        self.fc1 = nn.Sequential(
            # nn.Linear(10880, 128)
            nn.Linear(8192, 128),
        )

        return

    def input_norm(self, x):
        flat = x.reshape(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(
            -1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, x, mode='Normalized', return_activations=False, dropout=0):
        batch_size = x.size(0)
        conv_feats = self.block(self.input_norm(x))
        spp = spatial_pyramid_pool(conv_feats, batch_size, [int(conv_feats.size(2)), int(conv_feats.size(3))],
                                   self.output_num)

        if mode == 'NoFC':
            return spp

        spp = Dropout(dropout)(spp)  # 20% probability

        feature_a = self.fc1(spp).reshape(batch_size, -1)

        if mode == 'Normalized':
            if return_activations:
                return F.normalize(feature_a, dim=1, p=2), conv_feats
            else:
                return F.normalize(feature_a, dim=1, p=2)
        else:
            if return_activations:
                return feature_a, conv_feats
            else:
                return feature_a
