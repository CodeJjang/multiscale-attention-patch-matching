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
from my_classes import DatasetPairwiseTriplets, FPR95Accuracy1, FPR95Accuracy2, EvaluateTripletNet
from my_classes import SingleNet, MetricLearningCnn, EvaluateNet, SiamesePairwiseSoftmax,NormalizeImages
from losses import ContrastiveLoss, TripletLoss,OnlineTripletLoss,OnlineHardNegativeMiningTripletLoss,InnerProduct
from read_matlab_imdb import read_matlab_imdb




from multiprocessing import Process, freeze_support

if __name__ == '__main__':
    freeze_support()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NumGpus = torch.cuda.device_count()

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    name = torch.cuda.get_device_name(0)

    ModelsDirName   = './models/'
    LogsDirName     = './logs/'
    Description     = 'Symmetric CNN with Triplet loss, no HM'
    FileName        = 'visnir_sym_triplet'
    BestFileName    = 'visnir_best'
    TestDir         = 'F:\\multisensor\\test\\'
    LoadAllTestSets = False

    Varibales2Add = ['fc1A','fc1B','fc2A','fc2B']



    #loading previous results
    filepath = ModelsDirName + "visnir_sym_triplet32_best_assymetric.pth"
    FileList = glob.glob(filepath)

    if FileList:
        FileList.sort(key=os.path.getmtime)

        checkpoint = torch.load(FileList[-1])

        net = MetricLearningCnn(checkpoint['Mode'])
        net.to(device)

        for Var in Varibales2Add:

            stdv = 1. / np.sqrt(getattr(net, Var).weight.shape[0])
            checkpoint['state_dict'][Var+'.weight'] = torch.zeros(getattr(net, Var).weight.shape[0], getattr(net, Var).weight.shape[1])
            checkpoint['state_dict'][Var+'.weight'].uniform_(-stdv, stdv)
            checkpoint['state_dict'][Var+'.bias'] = torch.zeros(getattr(net, Var).bias.shape)
            checkpoint['state_dict'][Var+'.bias'].uniform_(-stdv, stdv)



        net.load_state_dict(checkpoint['state_dict'])
        torch.save(checkpoint, filepath)
