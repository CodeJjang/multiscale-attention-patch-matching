#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 19:55:41 2018

@author: lps
"""

import math
import torch
import torch.nn as nn
#out_pool_size: the output support we aim for
def spatial_pyramid_pool( previous_conv, num_sample, previous_conv_size, out_pool_size):
    
    for i in range(len(out_pool_size)):

        #Pooling support
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))

        #Padding to retain orgonal dimensions
        h_pad = int((h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2)
        w_pad = int((w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2)

        #apply pooling
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid,w_wid), padding=(h_pad,w_pad))
        
        x = maxpool(previous_conv)
        if(i==0):
            spp = x.view(num_sample, -1)
        else:
            spp = torch.cat((spp, x.view(num_sample, -1)), 1)
    return spp


