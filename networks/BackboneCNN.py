import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import Dropout

from layers.spp_layer import spatial_pyramid_pool


class BackboneCNN(nn.Module):

    def __init__(self, dropout, output_feat_map=False):
        super(BackboneCNN, self).__init__()

        # self.pre_block = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(32, affine=False),
        #     nn.ReLU()
        # )
        #
        # self.block = nn.Sequential(
        #
        #     nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(32, affine=False, momentum=0.1 ** 0.5),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=2, bias=False),
        #     nn.BatchNorm2d(64, affine=False, momentum=0.1 ** 0.5),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(64, affine=False, momentum=0.1 ** 0.5),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=2, bias=False),
        #     nn.BatchNorm2d(128, affine=False, momentum=0.1 ** 0.5),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(128, affine=False, momentum=0.1 ** 0.5),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128, affine=False, momentum=0.1 ** 0.5),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128, affine=False, momentum=0.1 ** 0.5),
        # )

        self.layer_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False, momentum=0.1 ** 0.5),
            nn.ReLU()
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False, momentum=0.1 ** 0.5),
            nn.ReLU()
        )

        self.layer_3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=2, bias=False),  # stride = 2
            nn.BatchNorm2d(64, affine=False, momentum=0.1 ** 0.5),
            nn.ReLU()
        )

        self.layer_4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False, momentum=0.1 ** 0.5),
            nn.ReLU()
        )

        self.layer_5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=2, bias=False),  # stride = 2  wrong:10368
            nn.BatchNorm2d(128, affine=False, momentum=0.1 ** 0.5),
            nn.ReLU()
        )

        self.layer_6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False, momentum=0.1 ** 0.5),
            nn.ReLU()
        )

        self.layer_7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),  # batch_size, 128,8,8
            nn.BatchNorm2d(128, affine=False, momentum=0.1 ** 0.5),
            nn.ReLU()
        )

        self.layer_8 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False, momentum=0.1 ** 0.5)
        )

        self.output_feat_map = output_feat_map
        if not output_feat_map:
            self.output_num = [8, 4, 2, 1]
            self.fc1 = nn.Sequential(
                nn.Linear(10880, 128)
            )

        self.dropout = dropout

        return

    def input_norm(self, x):
        flat = x.reshape(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(
            -1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def block_with_feats(self, x):
        feats = [x, self.layer_1(x)]
        for i in range(2, 9): # 9
            # feats.append(getattr(self, f'layer_{i}')(feats[-1]))
            feats.append(torch.utils.checkpoint.checkpoint(getattr(self, f'layer_{i}'), feats[-1]))
        if self.training:
            return feats[1:5] + [feats[-1]]
        return [feats[-1]]

    def forward(self, x, mode='Normalized'):
        batch_size = x.size(0)
        # conv_feats = self.pre_block(self.input_norm(x))
        # conv_feats = torch.utils.checkpoint.checkpoint(self.block, conv_feats)
        conv_feats = self.block_with_feats(self.input_norm(x))
        if self.output_feat_map:
            return conv_feats

        spp = spatial_pyramid_pool(conv_feats, batch_size, [int(conv_feats.size(2)), int(conv_feats.size(3))],
                                   self.output_num)

        spp = Dropout(self.dropout)(spp)

        feature_a = self.fc1(spp).reshape(batch_size, -1)

        if mode == 'Normalized':
            return F.normalize(feature_a, dim=1, p=2)
        else:
            return feature_a
