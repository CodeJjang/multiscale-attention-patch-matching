import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import copy
from tensorboardX import SummaryWriter
from torch import device
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

# my classes
from my_classes import imshow, ShowRowImages, ShowTwoRowImages
from my_classes import DatasetPairwiseTriplets, FPR95Accuracy
from my_classes import SingleNet, MetricLearningCnn, EvaluateNet, SiamesePairwiseSoftmax, NormalizeImages
from losses import ContrastiveLoss, TripletLoss, OnlineTripletLoss, OnlineHardNegativeMiningTripletLoss, InnerProduct
from read_matlab_imdb import read_matlab_imdb

import warnings
warnings.filterwarnings("ignore", message="UserWarning: albumentations.augmentations.transforms.RandomResizedCrop")


from multiprocessing import Process, freeze_support

if __name__ == '__main__':
    freeze_support()

    device: device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NumGpus = torch.cuda.device_count()

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    name = torch.cuda.get_device_name(0)

    ModelsDirName = 'best models/'
    ModelsDirName = 'data/Vis-Nir_grid/assymetric/'
    ModelsDirName = 'restore/models/'
    ModelsDirName = 'models1/'
    ModelsDirName = 'data/Vis-Nir/models/'
    ModelsDirName = 'models/'

    BestFileName = 'visnir_best_hybrid.pth'
    # TestDir = '/home/keller/Dropbox/multisensor/python/data/test/'
    TestDir = 'F:\\multisensor\\test\\'
    #TestDir = './data/test/'
    #TestDir = './data/Vis-Nir_grid/test/'
    #TestDir = 'F:\\multisensor\\test\\'  # error 1.1
    # TrainFile = '/home/keller/Dropbox/multisensor/python/data/Vis-Nir_Train.mat'
    TestDecimation = 1

    CnnMode = 'Hybrid'
    CnnMode = 'PairwiseAsymmetric'
    CnnMode = 'PairwiseSymmetric'

    # Load all datasets

    FileList = glob.glob(TestDir + "*.hdf5")
    TestData = dict()
    for File in FileList:
        path, DatasetName = os.path.split(File)
        DatasetName = os.path.splitext(DatasetName)[0]

        Data = read_matlab_imdb(File)

        #x = np.transpose(Data['testData'], (0, 3, 2, 1))
        x = Data['Data'].astype(np.float32)
        #x = np.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3]), order='F')


        x[:, :, :, :, 0] -= x[:, :, :, :, 0].mean()
        x[:, :, :, :, 1] -= x[:, :, :, :, 1].mean()
        x = torch.from_numpy(NormalizeImages(x));

        TestLabels = torch.from_numpy(np.squeeze(Data['Labels']))
        #TestLabels = torch.from_numpy(2 - Data['testLabels'])

        TestData[DatasetName] = dict()
        TestData[DatasetName]['Data'] = x
        TestData[DatasetName]['Labels'] = TestLabels

    # ------------------------------------------------------------------------------------------








    # -------------------------    loading previous results   ------------------------

    FileList = glob.glob(ModelsDirName + "visnir*")
    FileList = glob.glob(ModelsDirName + 'visnir_best120.pth')
    FileList = glob.glob(ModelsDirName + 'visnir_best_FPR_loss_1.2.pth')
    FileList = glob.glob(ModelsDirName + 'visnir_sym_triplet32_best_assymetric.pth')
    FileList = glob.glob(ModelsDirName + 'symmetric_visnir_best.pth')
    FileList = glob.glob(ModelsDirName + 'visnir_best.pth')
    FileList = glob.glob(ModelsDirName + 'visnir_sym_triplet40.pth')
    FileList = glob.glob(ModelsDirName + 'best hybrid after freeze - Copy.pth')
    FileList = glob.glob(ModelsDirName + 'symmetric_visnir_best.pth')
    for File in FileList:

        print(File+ ' loded')

        checkpoint = torch.load(File)
        net = MetricLearningCnn(checkpoint['Mode'])
        print('Mode:' + checkpoint['Mode'] + ' InnerBatchSize:' + repr(checkpoint['InnerBatchSize']))
        net.to(device)
        net.load_state_dict(checkpoint['state_dict'])

        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
        net.to(device)

        net = net.eval()



        StepSize = 2048

        # test accuracy
        NoSamples = 0
        TotalTestError = 0
        for DataName in tqdm(TestData):
            EmbTest1 = EvaluateNet(net.module.GetChannelCnn(0, CnnMode),
                                   TestData[DataName]['Data'][0::TestDecimation, :, :, :, 0], device,
                                   StepSize)
            EmbTest2 = EvaluateNet(net.module.GetChannelCnn(1, CnnMode),
                                   TestData[DataName]['Data'][0::TestDecimation, :, :, :, 1], device,
                                   StepSize)
            Dist = np.power(EmbTest1 - EmbTest2, 2).sum(1)
            TestData[DataName]['TestError'] = FPR95Accuracy(Dist, TestData[DataName]['Labels'][0::TestDecimation]) * 100
            TotalTestError += TestData[DataName]['TestError'] * TestData[DataName]['Data'].shape[0]
            NoSamples += TestData[DataName]['Data'].shape[0]
        TotalTestError /= NoSamples

        str = File+ ' Test Mode:' + CnnMode  + ' Cnn File Mode: ' + checkpoint['Mode']
        for DataName in TestData:
            str +=' ' + DataName + ': ' + repr(TestData[DataName]['TestError'])[0:6]
        str += ' Mean: ' + repr(TotalTestError)[0:6]
        print(str)