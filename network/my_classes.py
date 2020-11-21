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
from network.positional_encodings  import PositionalEncoding2D
from network.spp_layer import spatial_pyramid_pool
from network.transformer import Transformer, TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer



def separate_cnn_paras(modules):
    all_parameters = modules.parameters()
    paras_only_bn = []

    for pname, p in modules.named_parameters():
        if pname.find('bn') >= 0:
            paras_only_bn.append(p)

    paras_only_bn_id = list(map(id, paras_only_bn))
    paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id, all_parameters))

    #return paras_only_bn, paras_wo_bn
    return paras_wo_bn


def ComputeAllErros(TestData,net,device,StepSize):

    Errors = dict()
    Loss = 0
    for DataName in TestData:
        EmbTest1 = EvaluateNet(net.module.GetChannelCnn(0), TestData[DataName]['Data'][:, :, :, :, 0], device, StepSize)
        EmbTest2 = EvaluateNet(net.module.GetChannelCnn(1), TestData[DataName]['Data'][:, :, :, :, 1], device, StepSize)
        Dist = np.power(EmbTest1 - EmbTest2, 2).sum(1)
        Errors['TestError'] = FPR95Accuracy1(Dist, TestData[DataName]['Labels'])
        Loss += Errors['TestError']

    Errors['Mean'] /= len(TestData)





def NormalizeImages(x):
    #Result = (x/255.0-0.5)/0.5
    Result = x / (255.0/2)
    return Result




def FPR95Accuracy(Dist,Labels):
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






def EvaluateNet(net,data,device,StepSize):

    if (torch.cuda.device_count() > 1):
        net = nn.DataParallel(net)

    with torch.no_grad():

        for k in range(0, data.shape[0], StepSize):

            # ShowTwoRowImages(x[0:3, :, :, 0], x[0:3, :, :, 1])
            a = data[k:(k + StepSize), :, :, :]  # - my_training_Dataset.VisAvg

            # ShowTwoRowImages(a[0:10,0,:,:], b[0:10,0,:,:])
            a = a.to(device)
            a = net(a)

            if k==0:
                EmbA = np.zeros((data.shape[0], a.shape[1]),dtype=np.float32)

            EmbA[k:(k + StepSize)] = a.cpu()

    return EmbA










def EvaluateDualNets(net,Data1, Data2,device,StepSize,p=0):
    with torch.no_grad():

        for k in range(0, Data1.shape[0], StepSize):

            # ShowTwoRowImages(x[0:3, :, :, 0], x[0:3, :, :, 1])
            a = Data1[k:(k + StepSize), :, :, :]  # - my_training_Dataset.VisAvg
            b = Data2[k:(k + StepSize), :, :, :]  # - my_training_Dataset.VisAvg

            # ShowTwoRowImages(a[0:10,0,:,:], b[0:10,0,:,:])
            a,b = a.to(device),b.to(device)
            x = net(a,b ,p=p)

            if k == 0:
                keys = list(x.keys())
                Emb = dict()
                for key in keys:
                    Emb[key] = np.zeros((Data1.shape[0], x[key].shape[1]), dtype=np.float32)

            for key in keys:
                Emb[key][k:(k + StepSize)] = x[key].cpu()

    return Emb





def EvaluateSofmaxNet(net,Labels,device,Data,StepSize):

    with torch.no_grad():
        ValAccuracy = 0
        PosValAccuracy = 0
        NegValAccuracy = 0
        m = 0
        for k in range(0, Data.shape[0], StepSize):
            x = Data[k:min(k + StepSize,Data.shape[0]), :, :, :, :]
            # ShowTwoRowImages(x[0:3, :, :, 0], x[0:3, :, :, 1])
            a = x[:, :, :, :, 0]  # - my_training_Dataset.VisAvg
            b = x[:, :, :, :, 1]  # - my_training_Dataset.IrAvg
            # ShowTwoRowImages(a[0:10,0,:,:], b[0:10,0,:,:])
            a, b = a.to(device), b.to(device)

            Emb = net(a, b)
            Vals,EstimatedValLabels = torch.max(Emb, 1)

            CurrentLabels = Labels[k:(k + StepSize)]
            ValAccuracy += torch.mean((EstimatedValLabels.cpu() != CurrentLabels.cpu()).type(torch.DoubleTensor))
            m = m + 1

        ValAccuracy /= m

        return ValAccuracy.numpy(),Emb






def ShowRowImages(image_data):
    fig = plt.figure(figsize=(1,image_data.shape[0]))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(1,image_data.shape[0]),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    #for ax, im in zip(grid, image_data):
    for ax, im in zip(grid, image_data):
        # Iterating over the grid returns the Axes.
        ax.imshow(im,cmap='gray')
    plt.show()





def ShowTwoRowImages(image_data1,image_data2):
    fig = plt.figure(figsize=(2, image_data1.shape[0]))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(2,image_data1.shape[0]),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    #for ax, im in zip(grid, image_data):
    for ax, im in zip(grid, image_data1):
        # Iterating over the grid returns the Axes.
        if im.shape[0]==1:
            ax.imshow(im,cmap='gray')
        if im.shape[0]==3:
            ax.imshow(im)

    for i in range(image_data2.shape[0]):
        # Iterating over the grid returns the Axes.
        if im.shape[0] == 1:
            grid[i+image_data1.shape[0]].imshow(image_data2[i],cmap='gray')
        if im.shape[0] == 3:
            grid[i + image_data1.shape[0]].imshow(image_data2[i])
    plt.show()





def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()









class DatasetPairwiseTriplets(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, Data, Labels,batch_size, Augmentation, Mode,NegativeMode='Random'):
        'Initialization'
        self.PositiveIdx = np.squeeze(np.asarray(np.where(Labels == 1)));
        self.NegativeIdx = np.squeeze(np.asarray(np.where(Labels == 0)));

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
            # A.Transpose(always_apply=False, p=0.5),
            # A.Flip(always_apply=False, p=0.5),
            # A.RandomResizedCrop(self.RowsNo, self.ColsNo,scale=(0.9, 1.1) ,ratio=(0.9, 1.1), interpolation=cv2.INTER_CUBIC,always_apply=False,p=0.5),
            A.Rotate(limit=5, interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_REFLECT_101, always_apply=False,p=0.5),
            A.HorizontalFlip(always_apply=False, p=0.5),
            A.VerticalFlip(always_apply=False, p=0.5),
            #A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True,always_apply=False, p=0.5),
            #A.RandomGamma(gamma_limit=136, always_apply=False, p=0.5),
            #A.JpegCompression(quality_lower=40, quality_upper=100, p=0.5),
            #A.HueSaturationValue(hue_shift_limit=172, sat_shift_limit=20, val_shift_limit=27, always_apply=False, p=0.5)
        ])

    def __len__(self):
        'Denotes the total number of samples'
        return self.Data.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'

        # Select pos2 pairs
        if self.Mode == 'Pairwise':
            PosIdx = np.random.randint(self.PositiveIdxNo, size=self.batch_size)

        if self.Mode == 'Test':
            PosIdx = index


        PosIdx    = self.PositiveIdx[PosIdx]
        PosImages = self.Data[PosIdx, :, :, :].astype(np.float32)

        # imshow(torchvision.utils.make_grid(PosImages[0,:,:,0]))
        # plt.imshow(np.squeeze(PosImages[2040, :, :, :]));  # plt.show()

        pos1 = PosImages[:, :, :, 0]
        pos2 = PosImages[:, :, :, 1]

        if self.Mode == 'Pairwise':
            for i in range(0, PosImages.shape[0]):

                # Flip LR
                if (np.random.uniform(0, 1) > 0.5) & self.Augmentation["HorizontalFlip"]:
                    pos1[i,] = np.fliplr(pos1[i,])
                    pos2[i,] = np.fliplr(pos2[i,])

                #flip UD
                if (np.random.uniform(0, 1) > 0.5) & self.Augmentation["VerticalFlip"]:
                    pos1[i,] = np.flipud(pos1[i,])
                    pos2[i,] = np.flipud(pos2[i,])

                #test
                if self.Augmentation["Test"]['Do']:

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










class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x




#Define a Convolutional Neural Network
class BasicSingleNet(nn.Module):
    def __init__(self):
        super(BasicSingleNet, self).__init__()

        #MaxPool2d(kernel_size, stride, padding)
        self.pool2A = MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2B = MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2C = MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2D = MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2E = MaxPool2d(kernel_size=2, stride=2, padding=0)


        # Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv0 = Conv2d(1, 32, 5, stride=1, padding=2)
        self.conv1 = Conv2d(32,   64, 5, stride=1, padding=2)
        self.conv2 = Conv2d(64,  K, 3, stride=1, padding=1)
        self.conv3 = Conv2d(K, 256, 3, stride=1, padding=1)
        self.conv4 = Conv2d(256, 256, 3, stride=1, padding=1)
        self.fc1   = Linear(1024, K)
        self.fc2   = Linear(256, 512)#for soft max

        self.BatchNorm1 = BatchNorm2d(64)
        self.BatchNorm2 = BatchNorm2d(256)

        self.Dropout = nn.Dropout(p=0.1)


    def forward(self, x):#1,64,64
        x = self.pool2A(F.relu(self.conv0(x)))#32,32,32
        x = self.pool2B(F.relu(self.conv1(x)))#32,16,16

        x = self.BatchNorm1(x)

        x = self.pool2C(F.relu(self.conv2(x)))#K, 8, 8
        x = self.pool2D(F.relu(self.conv3(x)))#256, 4, 4

        x = self.BatchNorm2(x)

        x = self.pool2E(F.relu(self.conv4(x)))
        x = x.reshape(x.size(0), -1)
        #output = output.view(-1, 16 * 16 * 24)
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



def Prepare2DPosEncoding(PosEncodingX,PosEncodingY,RowNo,ColNo):

    PosEncodingX = PosEncodingX[0:ColNo].unsqueeze(0)#x=[1,..,20]
    PosEncodingY = PosEncodingY[0:RowNo]

    for i in range(RowNo):

        CurrentY = PosEncodingY[i, :].unsqueeze(0).unsqueeze(0).repeat(1,ColNo,1)

        if i==0:
            PosEncoding2D = torch.cat((PosEncodingX, CurrentY),2)
        else:
            CurrentPosEncoding2D = torch.cat((PosEncodingX, CurrentY), 2)

            PosEncoding2D        = torch.cat((PosEncoding2D, CurrentPosEncoding2D), 0)

    return PosEncoding2D




class MetricLearningCnn(nn.Module):
    #def __init__(self):
     #   super(SiameseTripletCnn, self).__init__()


    def __init__(self,Mode,DropoutP=0):
        super(MetricLearningCnn, self).__init__()

        self.Mode   = Mode

        #self.netS   = SingleNet()
        #self.netAS1 = SingleNet()
        #self.netAS2 = SingleNet()

        self.netS   = Model()
        self.netAS1 = Model()
        self.netAS2 = Model()

        K =128
        self.AttenS   =  AttentionEmbeddingCNN(K,EmbeddingMaxDim=20)
        self.AttenAS1 = AttentionEmbeddingCNN(K, EmbeddingMaxDim=20)
        self.AttenAS2 = AttentionEmbeddingCNN(K, EmbeddingMaxDim=20)

        self.fc1 = Linear(2*K, K)
        self.fc2 = Linear(2*K, K)
        self.fc2 = copy.deepcopy(self.fc1)

        self.fc1A = Linear(K, K)
        self.fc2A = copy.deepcopy(self.fc1A)


        self.Gain = torch.nn.Parameter(torch.ones(1))

        self.Gain1 = torch.nn.Parameter(torch.ones(1))
        self.Gain2 = torch.nn.Parameter(torch.ones(1))




    def GetChannelCnn(self,ChannelId,Mode):


        if (Mode == 'TripletSymmetric') | (Mode == 'PairwiseSymmetric'):
            return self.netS

        if (Mode == 'TripletAsymmetric') | (Mode == 'PairwiseAsymmetric'):
            if ChannelId == 0:
                return self.netAS1

            if ChannelId == 1:
                return self.netAS2

        if (Mode == 'Hybrid') | (Mode == 'Hybrid1') | (Mode == 'Hybrid2') :
            if ChannelId == 0:
                self.Mode = 'Hybrid1'
                return self

            if ChannelId == 1:
                self.Mode = 'Hybrid2'
                return self


        if (Mode == 'PairwiseSymmetricAttention') :
            if ChannelId == 0:
                self.Mode = 'PairwiseSymmetricAttention1'
                return self

            if ChannelId == 1:
                self.Mode = 'PairwiseSymmetricAttention2'
                return self
            return self



        if (Mode == 'PairwiseAsymmetricAttention'):
            if ChannelId == 0:
                self.Mode = 'PairwiseAsymmetricAttention1'
                return self

            if ChannelId == 1:
                self.Mode = 'PairwiseAsymmetricAttention2'
                return self
            return self


    def FreezeSymmetricCnn(self,OnOff):
        self.netS.FreezeCnn(OnOff)

    def FreezeSymmetricBlock(self,OnOff):
        self.netS.FreezeBlock(OnOff)

    def FreezeAsymmetricCnn(self,OnOff):
        self.netAS1.FreezeCnn(OnOff)
        self.netAS2.FreezeCnn(OnOff)


    def FreezeAsymmetricBlock(self,OnOff):
        self.netAS1.FreezeBlock(OnOff)
        self.netAS2.FreezeBlock(OnOff)









    #output CNN
    def forward(self,S1A, S1B=0,Mode = -1,DropoutP = 0.0):

        if (S1A.nelement() == 0):
            return 0

        if Mode == -1:
            Mode = self.Mode


        if Mode == 'PairwiseSymmetricAttention':

                output1 = self.AttenS(S1A, ActivMode=True, DropoutP=DropoutP)
                output2 = self.AttenS(S1B, ActivMode=True, DropoutP=DropoutP)

                Result = dict()
                Result['Emb1'] = output1
                Result['Emb2'] = output2

                return Result

        if Mode == 'PairwiseAsymmetricAttention':

            output1 = self.AttenAS1(S1A, ActivMode=True, DropoutP=DropoutP)
            output2 = self.AttenAS2(S1B, ActivMode=True, DropoutP=DropoutP)

            Result = dict()
            Result['Emb1'] = output1
            Result['Emb2'] = output2

            return Result


        if Mode == 'PairwiseSymmetric':


            # source#1: vis
            output1 = self.netS(S1A,DropoutP=DropoutP)
            output2 = self.netS(S1B,DropoutP=DropoutP)
            #summary(self.netS, (1, 64, 64))

            Result = dict()
            Result['Emb1'] = output1
            Result['Emb2'] = output2

            return Result




        if Mode == 'PairwiseAsymmetric':
            # source#1: vis
            output1 = self.netAS1(S1A,DropoutP=DropoutP)
            output2 = self.netAS2(S1B,DropoutP=DropoutP)


            Result = dict()
            Result['Emb1'] = output1
            Result['Emb2'] = output2
            return Result




        if (Mode == 'Hybrid') :
            #p: probability of an element to be zeroed.Default: 0.5
            DropoutP1 = 0

            Result = dict()

            # source#1: vis
            # channel1
            EmbSym1  = self.netS(S1A,'Normalized',DropoutP=DropoutP1)
            EmbAsym1 = self.netAS1(S1A,'Normalized',DropoutP=DropoutP1)

            # concat embeddings and apply relu: K+K=256
            #Hybrid1 = torch.cat((EmbSym1, EmbAsym1), 1)

            Hybrid1 = EmbSym1+self.Gain1*EmbAsym1
            #Hybrid1 = F.normalize(Hybrid1, dim=1, p=2)


            Hybrid1 = F.relu(Hybrid1)
            #Hybrid1 = Dropout(DropoutP)(Hybrid1)  # 20% probabilit

            # prepare output
            #Hybrid1 = self.fc1(Hybrid1)
            Hybrid1 = self.fc1A(Hybrid1)
            #Hybrid1 = self.fc1B(self.fc1A(Hybrid1))

            Hybrid1 = F.normalize(Hybrid1, dim=1, p=2)



            # channel2
            EmbSym2  = self.netS(S1B,'Normalized',DropoutP=DropoutP1)
            EmbAsym2 = self.netAS2(S1B,'Normalized',DropoutP=DropoutP1)

            # concat embeddings and apply relu: K+K=256
            #Hybrid2 = torch.cat((EmbSym2, EmbAsym2), 1)

            Hybrid2 = EmbSym2+self.Gain2*EmbAsym2
            #Hybrid2 = F.normalize(Hybrid2, dim=1, p=2)


            Hybrid2 = F.relu(Hybrid2)


            # prepare output
            #Hybrid2 = self.fc2(Hybrid2)
            Hybrid2 = self.fc2A(Hybrid2)

            Hybrid2 = F.normalize(Hybrid2, dim=1, p=2)

            if torch.any(torch.isnan(Hybrid1)) or torch.any(torch.isnan(Hybrid2)):
                print('Nan found')

            Result['Hybrid1']  = Hybrid1
            Result['Hybrid2']  = Hybrid2
            Result['EmbSym1']  = EmbSym1
            Result['EmbSym2']  = EmbSym2
            Result['EmbAsym1'] = EmbAsym1
            Result['EmbAsym2'] = EmbAsym2
            return Result





        if (Mode == 'Hybrid1'):
            # source#1: vis
            EmbSym1 = self.netS(S1A, 'Normalized')
            EmbAsym1 = self.netAS1(S1A, 'Normalized')

            # concat embeddings and apply relu: K+K=256
            #Hybrid1 = torch.cat((EmbSym1, EmbAsym1), 1)
            Hybrid1 = EmbSym1+self.Gain1*EmbAsym1
            Hybrid1 = F.relu(Hybrid1)

            # prepare output
            #Hybrid1 = self.fc1(Hybrid1)
            Hybrid1 = self.fc1A(Hybrid1)

            Hybrid1 = F.normalize(Hybrid1, dim=1, p=2)

            return Hybrid1




        if (Mode == 'Hybrid2'):
            # source#2: IR
            EmbSym2 = self.netS(S1A, 'Normalized')
            EmbAsym2 = self.netAS2(S1A, 'Normalized')

            # concat embeddings and apply relu: K+K=256
            #Hybrid2 = torch.cat((EmbSym2, EmbAsym2), 1)
            Hybrid2 = EmbSym2+self.Gain2*EmbAsym2
            Hybrid2 = F.relu(Hybrid2)

            # prepare output
            #Hybrid2 = self.fc2(Hybrid2)
            Hybrid2 = self.fc2A(Hybrid2)

            Hybrid2 = F.normalize(Hybrid2, dim=1, p=2)

            return Hybrid2





        if (Mode == 'PairwiseSymmetricAttention1') or (Mode == 'PairwiseSymmetricAttention2'):

            output1 = self.AttenS(S1A, ActivMode=True, DropoutP=DropoutP)

            return output1

        if (Mode == 'PairwiseAsymmetricAttention1'):

            output1 = self.AttenAS1(S1A, ActivMode=True, DropoutP=DropoutP)

            return output1

        if (Mode == 'PairwiseAsymmetricAttention2'):

            output1 = self.AttenAS2(S1A, ActivMode=True, DropoutP=DropoutP)

            return output1













class AttentionEmbeddingCNN(nn.Module):


    def __init__(self,K,EmbeddingMaxDim=20,):
        super(AttentionEmbeddingCNN, self).__init__()

        self.net = Model()

        self.AgeQuery = nn.Parameter(torch.randn(1, K))
        self.QueryPosEncode = nn.Parameter(torch.randn(1, K))

        EmbeddingMaxDim = 20
        self.PosEncodingX = nn.Parameter(torch.randn(EmbeddingMaxDim, int(K / 2)))
        self.PosEncodingY = nn.Parameter(torch.randn(EmbeddingMaxDim, int(K / 2)))

        self.output_num = [8, 4, 2, 1]
        # self.output_num = [8, 4]
        # self.output_num = [4,8]
        # self.output_num = [8]

        EncoderLayersNo = 2
        EncoderHeadsNo = 2


        self.DetrEncoderLayer = TransformerEncoderLayer(d_model=K, nhead=EncoderHeadsNo, dim_feedforward=int(K),
                                                        dropout=0.1, activation="relu", normalize_before=False)
        self.DetrEncoder = TransformerEncoder(encoder_layer=self.DetrEncoderLayer, num_layers=EncoderLayersNo)

        self.SPFC = nn.Linear(10880, K)
        self.SPFC = nn.Linear(8576, K)
        # self.SPFC = nn.Linear(10240, K)
        # self.SPFC = nn.Linear(8192, K)


    def spatial_pyramid_pool_2D(self, previous_conv, num_sample, previous_conv_size):
        for i in range(len(self.output_num)):

            # Pooling support
            h_wid = int(math.ceil(previous_conv_size[0] / self.output_num[i]))
            w_wid = int(math.ceil(previous_conv_size[1] / self.output_num[i]))

            # Padding to retain orgonal dimensions
            h_pad = int((h_wid * self.output_num[i] - previous_conv_size[0] + 1) / 2)
            w_pad = int((w_wid * self.output_num[i] - previous_conv_size[1] + 1) / 2)

            # apply pooling
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))

            y = maxpool(previous_conv)

            if (i == 0):
                spp = y.reshape(num_sample, -1)
            else:
                PosEncoding2D = Prepare2DPosEncoding(self.PosEncodingX,
                                                     self.PosEncodingY,
                                                     y.shape[2], y.shape[3])

                PosEncoding = PosEncoding2D.permute(2, 0, 1)
                PosEncoding = PosEncoding[:, 0:y.shape[2], 0:y.shape[3]]
                PosEncoding = PosEncoding.reshape((PosEncoding.shape[0], PosEncoding.shape[1] * PosEncoding.shape[2]))
                PosEncoding = PosEncoding.permute(1, 0).unsqueeze(1)
                PosEncoding = torch.cat((self.QueryPosEncode.unsqueeze(0), PosEncoding), 0)

                x = y.reshape((y.shape[0], y.shape[1], y.shape[2] * y.shape[3]))
                x = x.permute(2, 0, 1)

                AgeQuery = self.AgeQuery.repeat(1, x.shape[1], 1)
                x = torch.cat((AgeQuery, x), 0)
                # x = self.Encoder(src=x)
                x = self.DetrEncoder(src=x, pos=PosEncoding)

                # x = x.permute(1, 0, 2)
                x = x[0,]

                spp = torch.cat((spp, x.reshape(num_sample, -1)), 1)

        return spp

    def forward(self, x,DropoutP,ActivMode=True):

        output1, ActivMap1 = self.net(x, ActivMode=True, DropoutP=DropoutP)
        #return output1, ActivMap1

        spp_a = self.spatial_pyramid_pool_2D(ActivMap1, x.size(0),
                                             [int(ActivMap1.size(2)), int(ActivMap1.size(3))])
        Result = self.SPFC(spp_a)
        Result = F.normalize(Result, dim=1, p=2)

        return Result
