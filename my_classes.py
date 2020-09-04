import torch
from torch.utils import data
import numpy as np
import torch.nn.functional as F
from torch.nn import Conv2d, Linear, MaxPool2d, BatchNorm2d, Dropout
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage.transform import resize, rotate
import copy
from nets import Model
import cv2
# import torchfunc
import albumentations as A


def separate_cnn_paras(modules):
    all_parameters = modules.parameters()
    paras_only_bn = []

    for pname, p in modules.named_parameters():
        if pname.find('bn') >= 0:
            paras_only_bn.append(p)

    paras_only_bn_id = list(map(id, paras_only_bn))
    paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id, all_parameters))

    # return paras_only_bn, paras_wo_bn
    return paras_wo_bn


def ComputeAllErros(TestData, net, device, StepSize):
    Errors = dict()
    Loss = 0
    for DataName in TestData:
        EmbTest1 = EvaluateNet(net.module.GetChannelCnn(0), TestData[DataName]['Data'][:, :, :, :, 0], device, StepSize)
        EmbTest2 = EvaluateNet(net.module.GetChannelCnn(1), TestData[DataName]['Data'][:, :, :, :, 1], device, StepSize)
        Dist = np.power(EmbTest1 - EmbTest2, 2).sum(1)
        Errors['TestError'] = FPR95Accuracy1(Dist, TestData[DataName]['Labels'])
        Loss += Errors['TestError']

    Errors['Mean'] /= len(TestData)



def FPR95Accuracy(Dist, Labels):
    PosIdx = np.squeeze(np.asarray(np.where(Labels == 1)))
    NegIdx = np.squeeze(np.asarray(np.where(Labels == 0)))

    NegDist = Dist[NegIdx]
    PosDist = np.sort(Dist[PosIdx])

    Val = PosDist[int(0.95 * PosDist.shape[0])]

    FalsePos = sum(NegDist < Val);

    FPR95Accuracy = FalsePos / float(NegDist.shape[0])

    return FPR95Accuracy


def FPR95Threshold(PosDist):
    PosDist = PosDist.sort(dim=-1, descending=False)[0]
    Val = PosDist[int(0.95 * PosDist.shape[0])]

    return Val


def EvaluateNet(net, data, device, StepSize):
    with torch.no_grad():

        for k in range(0, data.shape[0], StepSize):

            # ShowTwoRowImages(x[0:3, :, :, 0], x[0:3, :, :, 1])
            a = data[k:(k + StepSize), :, :, :]  # - my_training_Dataset.VisAvg

            # ShowTwoRowImages(a[0:10,0,:,:], b[0:10,0,:,:])
            a = a.to(device)
            a = net(a)

            if k == 0:
                EmbA = np.zeros((data.shape[0], a.shape[1]), dtype=np.float32)

            EmbA[k:(k + StepSize)] = a.cpu()

    return EmbA


def EvaluateDualNets(net, Data1, Data2, device, StepSize, p=0):
    with torch.no_grad():

        for k in range(0, Data1.shape[0], StepSize):

            # ShowTwoRowImages(x[0:3, :, :, 0], x[0:3, :, :, 1])
            a = Data1[k:(k + StepSize), :, :, :]  # - my_training_Dataset.VisAvg
            b = Data2[k:(k + StepSize), :, :, :]  # - my_training_Dataset.VisAvg

            # ShowTwoRowImages(a[0:10,0,:,:], b[0:10,0,:,:])
            a, b = a.to(device), b.to(device)
            x = net(a, b, p=p)

            if k == 0:
                keys = list(x.keys())
                Emb = dict()
                for key in keys:
                    Emb[key] = np.zeros((Data1.shape[0], x[key].shape[1]), dtype=np.float32)

            for key in keys:
                Emb[key][k:(k + StepSize)] = x[key].cpu()

    return Emb


def EvaluateSofmaxNet(net, Labels, device, Data, StepSize):
    with torch.no_grad():
        ValAccuracy = 0
        PosValAccuracy = 0
        NegValAccuracy = 0
        m = 0
        for k in range(0, Data.shape[0], StepSize):
            x = Data[k:min(k + StepSize, Data.shape[0]), :, :, :, :]
            # ShowTwoRowImages(x[0:3, :, :, 0], x[0:3, :, :, 1])
            a = x[:, :, :, :, 0]  # - my_training_Dataset.VisAvg
            b = x[:, :, :, :, 1]  # - my_training_Dataset.IrAvg
            # ShowTwoRowImages(a[0:10,0,:,:], b[0:10,0,:,:])
            a, b = a.to(device), b.to(device)

            Emb = net(a, b)
            Vals, EstimatedValLabels = torch.max(Emb, 1)

            CurrentLabels = Labels[k:(k + StepSize)]
            ValAccuracy += torch.mean((EstimatedValLabels.cpu() != CurrentLabels.cpu()).type(torch.DoubleTensor))
            m = m + 1

        ValAccuracy /= m

        return ValAccuracy.numpy(), Emb


def ShowRowImages(image_data):
    fig = plt.figure(figsize=(1, image_data.shape[0]))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(1, image_data.shape[0]),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    # for ax, im in zip(grid, image_data):
    for ax, im in zip(grid, image_data):
        # Iterating over the grid returns the Axes.
        ax.imshow(im, cmap='gray')
    plt.show()


def ShowTwoRowImages(image_data1, image_data2):
    fig = plt.figure(figsize=(2, image_data1.shape[0]))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(2, image_data1.shape[0]),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    # for ax, im in zip(grid, image_data):
    for ax, im in zip(grid, image_data1):
        # Iterating over the grid returns the Axes.
        if im.shape[0] == 1:
            ax.imshow(im, cmap='gray')
        if im.shape[0] == 3:
            ax.imshow(im)

    for i in range(image_data2.shape[0]):
        # Iterating over the grid returns the Axes.
        if im.shape[0] == 1:
            grid[i + image_data1.shape[0]].imshow(image_data2[i], cmap='gray')
        if im.shape[0] == 3:
            grid[i + image_data1.shape[0]].imshow(image_data2[i])
    plt.show()


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)
        x = x / norm.unsqueeze(-1).expand_as(x)
        return x


# Define a Convolutional Neural Network
class BasicSingleNet(nn.Module):
    def __init__(self):
        super(BasicSingleNet, self).__init__()

        # MaxPool2d(kernel_size, stride, padding)
        self.pool2A = MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2B = MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2C = MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2D = MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2E = MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv0 = Conv2d(1, 32, 5, stride=1, padding=2)
        self.conv1 = Conv2d(32, 64, 5, stride=1, padding=2)
        self.conv2 = Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv3 = Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv4 = Conv2d(256, 256, 3, stride=1, padding=1)
        self.fc1 = Linear(1024, 128)
        self.fc2 = Linear(256, 512)  # for soft max

        self.BatchNorm1 = BatchNorm2d(64)
        self.BatchNorm2 = BatchNorm2d(256)

        self.Dropout = nn.Dropout(p=0.1)

    def forward(self, x):  # 1,64,64
        x = self.pool2A(F.relu(self.conv0(x)))  # 32,32,32
        x = self.pool2B(F.relu(self.conv1(x)))  # 32,16,16

        x = self.BatchNorm1(x)

        x = self.pool2C(F.relu(self.conv2(x)))  # 128, 8, 8
        x = self.pool2D(F.relu(self.conv3(x)))  # 256, 4, 4

        x = self.BatchNorm2(x)

        x = self.pool2E(F.relu(self.conv4(x)))
        x = x.reshape(x.size(0), -1)
        # output = output.view(-1, 16 * 16 * 24)
        x = self.fc1(x)
        x = self.Dropout(x)

        return x


class SingleNet(nn.Module):
    def __init__(self):
        super(SingleNet, self).__init__()

        self.net = BasicSingleNet()

    def forward(self, x):
        x = self.net(x)
        x = F.normalize(x, dim=1, p=2)

        return x


class MetricLearningCnn(nn.Module):
    # def __init__(self):
    #   super(SiameseTripletCnn, self).__init__()

    def __init__(self, Mode):
        super(MetricLearningCnn, self).__init__()

        self.Mode = Mode

        # self.netS   = SingleNet()
        # self.netAS1 = SingleNet()
        # self.netAS2 = SingleNet()

        self.netS = Model()
        self.netAS1 = Model()
        self.netAS2 = Model()

        self.fc1 = Linear(128 * 2, 128)
        self.fc2 = Linear(128 * 2, 128)
        self.fc3 = Linear(128 * 2, 4)

        self.fc2 = copy.deepcopy(self.fc1)

        K = 16
        self.fc1A = Linear(128 * 2, K)
        self.fc1B = Linear(K, 128)

        self.fc2A = Linear(128 * 2, K)
        self.fc2B = Linear(K, 128)

    def SymmCnnParams(self):
        netSParams = separate_cnn_paras(self.netS)
        return netSParams

    def AsymmCnnParams(self):
        netAS1Params = separate_cnn_paras(self.netAS1)
        netAS2Params = separate_cnn_paras(self.netAS2)

        Params = netAS1Params + netAS2Params

        return Params

    def BaseCnnParams(self):
        netSParams = separate_cnn_paras(self.netS)
        netAS1Params = separate_cnn_paras(self.netAS1)
        netAS2Params = separate_cnn_paras(self.netAS2)

        Params = netSParams + netAS1Params + netAS2Params

        return Params

    def HeadCnnParams(self):
        fc1Params = separate_cnn_paras(self.fc1)
        fc2Params = separate_cnn_paras(self.fc2)

        Params = fc1Params + fc2Params

        return Params

    def GetChannelCnn(self, ChannelId, Mode):

        if (Mode == 'TripletSymmetric') | (Mode == 'PairwiseSymmetric'):
            return self.netS

        if (Mode == 'TripletAsymmetric') | (Mode == 'PairwiseAsymmetric'):
            if ChannelId == 0:
                return self.netAS1

            if ChannelId == 1:
                return self.netAS2

        if (Mode == 'Hybrid') | (Mode == 'Hybrid1') | (Mode == 'Hybrid2'):
            if ChannelId == 0:
                self.Mode = 'Hybrid1'
                return self

            if ChannelId == 1:
                self.Mode = 'Hybrid2'
                return self

    def FreezeSymmetricCnn(self, OnOff):
        self.netS.FreezeCnn(OnOff)

    def FreezeAsymmetricCnn(self, OnOff):
        self.netAS1.FreezeCnn(OnOff)
        self.netAS2.FreezeCnn(OnOff)

    # output CNN
    def forward(self, S1A, S1B=0, Mode=-1, Rot1=0, Rot2=0, p=0.0):

        if (S1A.nelement() == 0):
            return 0

        if Mode == -1:
            Mode = self.Mode

        if Mode == 'PairwiseSymmetric':
            # source#1: vis
            output1 = self.netS(S1A)
            output2 = self.netS(S1B)
            # summary(self.netS, (1, 64, 64))

            Result = dict()
            Result['Emb1'] = output1
            Result['Emb2'] = output2
            return Result

        if Mode == 'PairwiseSymmetricRot':
            # source#1: vis

            # unnormalized
            output1 = self.netS.net(S1A)
            output2 = self.netS.net(S1B)

            # normalized
            Norm1 = F.normalize(output1, dim=1, p=2)
            Norm2 = F.normalize(output2, dim=1, p=2)

            # rotated unnormalized
            Rot1 = self.netS.net(Rot1)
            Rot2 = self.netS.net(Rot2)

            Result = dict()
            Result['Unnormalized1'] = output1
            Result['Unnormalized2'] = output2

            Result['Emb1'] = Norm1
            Result['Emb2'] = Norm2

            Result['RotUnnormalized1'] = Rot1
            Result['RotUnnormalized2'] = Rot2

            return Result

        if Mode == 'PairwiseAsymmetric':
            # source#1: vis
            output1 = self.netAS1(S1A)
            output2 = self.netAS2(S1B)

            Result = dict()
            Result['Emb1'] = output1
            Result['Emb2'] = output2
            return Result

        if (Mode == 'Hybrid') | (Mode == 'HybridCat'):
            # p: probability of an element to be zeroed.Default: 0.5
            p = 0.0

            Result = dict()

            # source#1: vis
            # channel1
            EmbSym1 = self.netS(S1A, 'Normalized')
            EmbAsym1 = self.netAS1(S1A, 'Normalized')

            # concat embeddings and apply relu: 128+128=256
            Hybrid1 = torch.cat((EmbSym1, EmbAsym1), 1)

            # if Mode == 'HybridCat':
            #    Result['Hybrid1'] = Hybrid1

            Hybrid1 = F.relu(Hybrid1)
            Hybrid1 = Dropout(p)(Hybrid1)  # 20% probabilit
            # Hybrid1 = torch.cat((EmbSym1, EmbAsym1), 1)

            # prepare output
            Hybrid1 = self.fc1(Hybrid1)
            # Hybrid1 = self.fc1B(self.fc1A(Hybrid1))

            #

            # Hybrid1 = F.relu((self.fc1A(Hybrid1))
            # Hybrid1 = self.fc1B(Hybrid1)

            Hybrid1 = F.normalize(Hybrid1, dim=1, p=2)
            # Hybrid1 = L2Norm()(Hybrid1)

            # channel2
            EmbSym2 = self.netS(S1B, 'Normalized')
            EmbAsym2 = self.netAS2(S1B, 'Normalized')

            # concat embeddings and apply relu: 128+128=256
            Hybrid2 = torch.cat((EmbSym2, EmbAsym2), 1)

            # if Mode == 'HybridCat':
            #   Result['Hybrid2'] = Hybrid2
            #  return Result

            Hybrid2 = F.relu(Hybrid2)
            Hybrid2 = Dropout(p)(Hybrid2)

            # prepare output
            Hybrid2 = self.fc2(Hybrid2)

            # Hybrid2 = F.relu((self.fc2A(Hybrid2))
            # Hybrid2 = self.fc2B(Hybrid2)

            Hybrid2 = F.normalize(Hybrid2, dim=1, p=2)
            # Hybrid2 = L2Norm()(Hybrid2)

            Result['Hybrid1'] = Hybrid1
            Result['Hybrid2'] = Hybrid2
            Result['EmbSym1'] = EmbSym1
            Result['EmbSym2'] = EmbSym2
            Result['EmbAsym1'] = EmbAsym1
            Result['EmbAsym2'] = EmbAsym2
            return Result

        if (Mode == 'Hybrid1') | (Mode == 'HybridCat1'):
            # source#1: vis
            # channel1
            EmbSym1 = self.netS(S1A, 'Normalized')
            EmbAsym1 = self.netAS1(S1A, 'Normalized')

            # concat embeddings and apply relu: 128+128=256
            Hybrid1 = torch.cat((EmbSym1, EmbAsym1), 1)

            if Mode == 'HybridCat':
                return Hybrid1

            Hybrid1 = F.relu(Hybrid1)
            # Hybrid1 = torch.cat((EmbSym1, EmbAsym1), 1)

            # prepare output
            Hybrid1 = self.fc1(Hybrid1)
            # Hybrid1 = self.fc1B(self.fc1A(Hybrid1))
            Hybrid1 = F.normalize(Hybrid1, dim=1, p=2)
            # Hybrid1 = L2Norm()(Hybrid1)

            return Hybrid1

        if (Mode == 'Hybrid2') | (Mode == 'HybridCat2'):
            # source#2: IR
            EmbSym2 = self.netS(S1A, 'Normalized')
            EmbAsym2 = self.netAS2(S1A, 'Normalized')

            # concat embeddings and apply relu: 128+128=256
            Hybrid2 = torch.cat((EmbSym2, EmbAsym2), 1)

            if Mode == 'HybridCat2':
                return Hybrid2

            Hybrid2 = F.relu(Hybrid2)
            # Hybrid2 = torch.cat((EmbSym2, EmbAsym2), 1)

            # prepare output
            Hybrid2 = self.fc2(Hybrid2)
            # Hybrid2 = self.fc2B(self.fc2A(Hybrid2))
            Hybrid2 = F.normalize(Hybrid2, dim=1, p=2)
            # Hybrid2 = L2Norm()(Hybrid2)

            return Hybrid2

        if Mode == 'SM':
            z = F.relu(torch.cat((S1A, S1B), 1))
            RotFlag = self.fc3(z)

            return RotFlag


class SiamesePairwiseSoftmax(nn.Module):
    def __init__(self):
        super(SiamesePairwiseSoftmax, self).__init__()

        self.net = SingleNet()
        self.fc1 = Linear(128 * 2, 256)
        self.fc2 = Linear(256, 2)

    def GetChannelCnn(self, ChannelId=0):
        return self.net

    # output CNN
    def forward(self, x1, x2):
        emb_x1 = self.net(x1)
        emb_x2 = self.net(x2)

        # prepare pos result
        y = torch.cat((emb_x1, emb_x2), 1)
        y = F.relu(self.fc1(y))
        y = self.fc2(y)

        return y
