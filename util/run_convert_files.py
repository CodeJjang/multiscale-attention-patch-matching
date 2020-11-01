import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import copy
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

#my classes
from my_classes import imshow, ShowRowImages, ShowTwoRowImages, EvaluateSofmaxNet
from my_classes import DatasetPairwiseTriplets, FPR95Accuracy
from my_classes import SingleNet, MetricLearningCnn, EvaluateNet, SiamesePairwiseSoftmax,NormalizeImages
from losses import ContrastiveLoss, TripletLoss,OnlineTripletLoss,OnlineHardNegativeMiningTripletLoss
from read_matlab_imdb import read_matlab_imdb
import h5py




from multiprocessing import Process, freeze_support

if __name__ == '__main__':
    freeze_support()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NumGpus = torch.cuda.device_count()

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    name = torch.cuda.get_device_name(0)

    ModelsDirName   = './models/'
    BestFileName    = 'visnir_best'
    TestDir = './data/test/'
    TestDir = './data/Vis-Nir_grid/'
    LoadAllTestSets = True
    TrainFile = 'f:/multisensor/train/Vis-Nir_Train.mat'
    TrainFile = './data/brown/patchdata_64x64.h5'

    ConvertTrainFiles = True
    ConvertTestFiles  = True
    ConvertPatchFiles = True


    if ConvertPatchFiles:
        Data              = read_matlab_imdb(TrainFile)
        Data['liberty']   = TrainingSetData = np.reshape(Data['liberty'], (Data['liberty'].shape[0], 1, 64, 64), order='F')
        Data['notredame'] = TrainingSetData = np.reshape(Data['notredame'],(Data['notredame'].shape[0], 1, 64, 64), order='F')
        Data['yosemite']  = TrainingSetData = np.reshape(Data['yosemite'],(Data['yosemite'].shape[0], 1, 64, 64), order='F')

        with h5py.File('patchdata1' + '.h5', 'w') as f:
            f.create_dataset('liberty', data=Data['liberty'])
            f.create_dataset('notredame', data=Data['notredame'])
            f.create_dataset('yosemite', data=Data['yosemite'])



    if ConvertTrainFiles:
        path, DatasetName = os.path.split(TrainFile)
        DatasetName = os.path.splitext(TrainFile)[0]

        Data              = read_matlab_imdb(TrainFile)
        TrainingSetData   = np.transpose(Data['data'], (0, 3, 2, 1))
        TrainingSetLabels = np.squeeze(Data['labels'])
        TrainingSetSet    = np.squeeze(Data['set'])

        TrainingSetData   = np.reshape(TrainingSetData,(TrainingSetData.shape[0], 1, TrainingSetData.shape[1], TrainingSetData.shape[2], TrainingSetData.shape[3]),order='F')
        TrainingSetLabels = 2 - TrainingSetLabels

        with h5py.File(DatasetName + '.hdf5', 'w') as f:
            f.create_dataset('Data', data=TrainingSetData, compression='gzip', compression_opts=9)
            f.create_dataset('Labels', data=TrainingSetLabels, compression='gzip', compression_opts=9)
            f.create_dataset('Set', data=TrainingSetSet, compression='gzip', compression_opts=9)




    if ConvertTestFiles:

        #Load all datasets
        FileList = glob.glob(TestDir + "*.mat")

        if LoadAllTestSets == False:
            FileList = [FileList[0]]

        FileList = ['./data/Vis-Nir_grid/Vis-Nir_grid_Test.mat']

        TestData = dict()
        for File in FileList:

            path, DatasetName = os.path.split(File)
            DatasetName       = os.path.splitext(DatasetName)[0]

            print(File)
            Data     = read_matlab_imdb(File)

            x    = np.transpose(Data['testData'], (0, 3, 2, 1))
            TestLabels = torch.from_numpy(2 - Data['testLabels'])

            x    = np.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3]),order='F')
            with h5py.File(path+'/'+DatasetName[:-5] + '.hdf5', 'w') as f:
                f.create_dataset('Data', data=x,compression='gzip',compression_opts=9)
                f.create_dataset('Labels', data=TestLabels,compression='gzip',compression_opts=9)

