import torch
from torch.utils import data
import numpy as np
import torch.nn.functional as F
from torch.nn import Conv2d, Linear,MaxPool2d, BatchNorm2d, Dropout, init
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage.transform import resize,rotate
import copy
from network.nets import Model
import cv2
import math
import albumentations as A
from torchvision.transforms import transforms
#from network import transforms
from network.positional_encodings  import PositionalEncoding2D
from network.spp_layer import spatial_pyramid_pool
from network.transformer import Transformer, TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
from PIL import Image


def NormalizeImages(x):
    #Result = (x/255.0-0.5)/0.5
    Result = x / (255.0/2)
    return Result


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:

            try:
                sigma = (self.max - self.min) * np.random.random_sample() + self.min
                sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
            except:
                aa=9

        return sample



class Compose1(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)

            if img.ndim == 3:
                aa=2
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string






class DatasetPairwiseTriplets(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, Data, Labels,batch_size, Augmentation, Mode,NegativeMode='Random'):
        'Initialization'
        self.PositiveIdx = np.squeeze(np.asarray(np.where(Labels == 1)))
        self.NegativeIdx = np.squeeze(np.asarray(np.where(Labels == 0)))

        self.PositiveIdxNo = len(self.PositiveIdx)
        self.NegativeIdxNo = len(self.NegativeIdx)

        self.Data   = Data
        self.Labels = Labels

        self.batch_size = batch_size
        self.Augmentation = Augmentation

        self.Mode = Mode
        self.NegativeMode = NegativeMode

        self.ChannelMean1 = Data[:, :, :, 0].mean()
        self.ChannelMean2 = Data[:, :, :, 1].mean()

        self.RowsNo = Data.shape[1]
        self.ColsNo = Data.shape[2]


        self.transform = A.ReplayCompose([
            A.Rotate(limit=5, interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_REFLECT_101, always_apply=False,p=0.5),
            A.HorizontalFlip(always_apply=False, p=0.5),
            A.VerticalFlip(always_apply=False, p=0.5),
        ])

    def __len__(self):
        'Denotes the total number of samples'
        return self.Data.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'

        # Select pos2 pairs
        PosIdx = np.random.randint(self.PositiveIdxNo, size=self.batch_size)


        PosIdx    = self.PositiveIdx[PosIdx]
        PosImages = self.Data[PosIdx, :, :, :].astype(np.float32)

        # imshow(torchvision.utils.make_grid(PosImages[0,:,:,0]))
        # plt.imshow(np.squeeze(PosImages[2040, :, :, :]));  # plt.show()

        pos1 = PosImages[:, :, :, 0]
        pos2 = PosImages[:, :, :, 1]


        for i in range(0, PosImages.shape[0]):

            # Flip LR
            if (np.random.uniform(0, 1) > 0.5) and self.Augmentation["HorizontalFlip"]:
                pos1[i,] = np.fliplr(pos1[i,])
                pos2[i,] = np.fliplr(pos2[i,])

            #flip UD
            if (np.random.uniform(0, 1) > 0.5) and self.Augmentation["VerticalFlip"]:
                pos1[i,] = np.flipud(pos1[i,])
                pos2[i,] = np.flipud(pos2[i,])

            #test
            if self.Augmentation["Test"]:

                #plt.imshow(pos1[i,:,:],cmap='gray');plt.show();
                data= self.transform(image=pos1[i, :, :])
                pos1[i,] = data['image']
                pos2[i,] = A.ReplayCompose.replay(data['replay'], image=pos2[i, :, :])['image']


            # rotate:0, 90, 180,270,
            if self.Augmentation["Rotate90"]:
                idx = np.random.randint(low=0, high=4, size=1)[0]  # choose rotation
                pos1[i,] = np.rot90(pos1[i, ], idx)
                pos2[i,] = np.rot90(pos2[i, ], idx)


            #random crop
            if  (np.random.uniform(0, 1) > 0.5) & self.Augmentation["RandomCrop"]['Do']:
                dx = np.random.uniform(self.Augmentation["RandomCrop"]['MinDx'], self.Augmentation["RandomCrop"]['MaxDx'])
                dy = np.random.uniform(self.Augmentation["RandomCrop"]['MinDy'], self.Augmentation["RandomCrop"]['MaxDy'])

                dx=dy

                x0 = int(dx*self.ColsNo)
                y0 = int(dy*self.RowsNo)

                #ShowRowImages(pos1[0:1,:,:])
                #plt.imshow(pos1[i,:,:],cmap='gray');plt.show();
                #aa = pos1[i,y0:,x0:]

                pos1[i, ] = resize(pos1[i,y0:,x0:], (self.RowsNo, self.ColsNo))

                #ShowRowImages(pos1[0:1, :, :])

                pos2[i,] = resize(pos2[i,y0:,x0:], (self.RowsNo, self.ColsNo))


        Result = dict()

        pos1 -= self.ChannelMean1
        pos2 -= self.ChannelMean2

        Result['pos1']   = NormalizeImages(pos1)
        Result['pos2']   = NormalizeImages(pos2)

        return Result







