import torch
import torchvision
import matplotlib.pyplot as plt
from termcolor import colored
import numpy as np
import glob
import os
import copy
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR
from torch.utils import data
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import GPUtil
import json
import math
from pathlib import Path

# my classes
from hpatches.utils.hpatch import hpatch_descr, hpatch_sequence
from hpatches.utils.results import results_verification, results_matching, results_retrieval
from hpatches.utils.tasks import eval_verification, eval_matching, eval_retrieval
from network.my_classes import imshow, ShowRowImages, ShowTwoRowImages, EvaluateDualNets, EvaluateSingleNet
from network.my_classes import FPR95Accuracy,SingleNet, MetricLearningCnn, EvaluateNet,EvaluateDualNets
from network.losses import ContrastiveLoss, TripletLoss, OnlineTripletLoss, OnlineHardNegativeMiningTripletLoss
from util.utils import LoadModel
import h5py
import warnings
warnings.filterwarnings("ignore", message="UserWarning: albumentations.augmentations.transforms.RandomResizedCrop")

def assert_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def load_datasets_paths(ds_name):
    if ds_name == 'hpatches-liberty':
        test_dir = 'D:\\multisensor\\datasets\\hpatches-benchmark\\data\\hpatches-release-multisensor\\data.hdf5'
    elif ds_name == 'hpatches-all-brown':
        test_dir = 'D:\\multisensor\\datasets\\hpatches-benchmark\\data\\hpatches-release-multisensor\\data.hdf5'
    return test_dir


def evaluate_hpatch(TestData, TestDecimation1, CnnMode, device, StepSize, splits, taskdir):
    descriptors = {}
    for seq_name, seq_data in TestData.items():
        local_descr_map = {}
        for t in hpatch_descr.itr:
            EmbTest = EvaluateSingleNet(net, getattr(seq_data, t), CnnMode, device, StepSize)['Emb1']
            local_descr_map[t] = EmbTest
        descriptors[seq_name] = hpatch_descr(descr_map=local_descr_map)
    descriptors['dim'] = descriptors[list(descriptors.keys())[0]].dim
    descriptors['distance'] = 'L2'
    eval = eval_verification(descriptors, splits['full'], taskdir)
    TotalTestError = results_verification(CnnMode, splits['full'], eval)
    print('Verification mAP:', TotalTestError * 100)
    eval = eval_matching(descriptors, splits['full'])
    TotalTestError = results_matching(CnnMode, splits['full'], eval)
    print('Matching mAP:', TotalTestError * 100)
    eval = eval_retrieval(descriptors, splits['full'], taskdir)
    TotalTestError = results_retrieval(CnnMode, splits['full'], eval)
    print('Retrieval mAP:', TotalTestError * 100)

if __name__ == '__main__':
    np.random.seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#"cuda:0"
    NumGpus = torch.cuda.device_count()
    torch.cuda.empty_cache()
    GPUtil.showUtilization()

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    name = torch.cuda.get_device_name(0)

    ModelsDirName = './artifacts/symmetric_enc_transformer_hpatches_liberty_1/models/'
    Description = 'Symmetric CNN with Triplet loss, no HM'
    BestFileName = 'best_model'
    FileName = 'model_epoch_'
    # TestDir = '/home/keller/Dropbox/multisensor/python/data/test/'
    # TestDir = 'F:\\multisensor\\test\\'
    # TestDir = 'data\\Vis-Nir_grid\\test\\'
    # TrainFile = '/home/keller/Dropbox/multisensor/python/data/Vis-Nir_Train.mat'
    # TrainFile = 'f:\\multisensor\\train\\Vis-Nir_Train.hdf5'
    # TrainFile = './data/Vis-Nir_grid/Vis-Nir_grid_Train.hdf5'
    ds_name = 'hpatches-liberty'
    TestDir = load_datasets_paths(ds_name)
    TestDecimation = 1
    FPR95 = 0.8


    torch.manual_seed(0)
    np.random.seed(0)
    #torch.set_deterministic(True)

    if True:
        GeneratorMode = 'Pairwise'
        CnnMode = 'Symmetric'
        CnnMode = 'SymmetricAttention'
        # CnnMode = 'SymmetricDecoder'
        NegativeMiningMode = 'Random'
        criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode=NegativeMiningMode,device=device)
        Description = 'Symmetric Hardest'

        DropoutP = 0.5



        StartBestModel = False
        UseBestScore   = False
        load_epoch = 60


    if False:
        GeneratorMode = 'Pairwise'
        CnnMode = 'Asymmetric'
        CnnMode = 'AsymmetricAttention'
        # CnnMode = 'AsymmetricDecoder'

        NegativeMiningMode = 'Random'
        #NegativeMiningMode = 'Hardest'
        criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode=NegativeMiningMode,device=device)
        # criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='HardPos', MarginRatio=1.0/2, PosRatio=1. / 2)

        InitializeOptimizer = True
        UseWarmUp           = True

        StartBestModel      = False
        UseBestScore        = False

        FreezeSymmetricCnn   = True
        FreezeAsymmetricCnn  = False

        LearningRate = 1e-1
        OuterBatchSize = 2 * 12
        InnerBatchSize = 2 * 12



        weight_decay = 0
        DropoutP = 0.5

        Augmentation["Test"] = {'Do': False}
        Augmentation["HorizontalFlip"] = True
        Augmentation["VerticalFlip"] = True
        Augmentation["Rotate90"] = True
        Augmentation["RandomCrop"] = {'Do': True, 'MinDx': 0, 'MaxDx': 0.2, 'MinDy': 0, 'MaxDy': 0.2}

        #AssymetricInitializationPhase = True
        Description = 'Asymmetric'




    if False:
        GeneratorMode = 'Pairwise'
        # CnnMode            = 'HybridRot'
        CnnMode = 'Hybrid'
        CnnMode = 'AttenHybrid'

        NegativeMiningMode = 'Random'
        #NegativeMiningMode = 'Hardest'

        criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode=NegativeMiningMode,device=device)
        #criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode="Hardest",device=device)
        #criterion        = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='HardPos', MarginRatio=1.0/4, PosRatio=1./4)
        #HardestCriterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='Hardest')

        #criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='HardPos', MarginRatio=1.0 / 2, PosRatio=1. / 2,device=device)

        PairwiseLoss      = PairwiseLoss()

        InitializeOptimizer = True
        UseWarmUp           = True

        StartBestModel = False
        UseBestScore = False

        LearningRate = 1e-1
        OuterBatchSize = 2*12
        InnerBatchSize = 2*12


        DropoutP = 0.5
        weight_decay= 0#1e-5

        TestMode = False
        TestDecimation = 10



        FreezeSymmetricCnn  = False
        FreezeAsymmetricCnn = False

        AssymetricInitializationPhase = False

        Augmentation["Test"] = {'Do': False}
        Augmentation["HorizontalFlip"] = True
        Augmentation["VerticalFlip"] = True
        Augmentation["Rotate90"] = True
        Augmentation["RandomCrop"] = {'Do': True, 'MinDx': 0, 'MaxDx': 0.2, 'MinDy': 0, 'MaxDy': 0.2}



    # TestData = load_hpatches_dataset(TestDir)
    print('Loading hpatchs...')
    TestData = {}
    with h5py.File(TestDir, 'r') as handle:
        for seq_name in handle.keys():
            hpatch = hpatch_sequence()
            seq_data = handle.get(seq_name)
            for t in seq_data.keys():
                setattr(hpatch, t, torch.from_numpy(np.array(seq_data.get(t))))
            TestData[seq_name] = hpatch
    hpatch_taskdir = "D:\\multisensor\\datasets\\hpatches-benchmark\\tasks"
    with open(os.path.join(hpatch_taskdir, "splits", "splits.json")) as f:
        hpatch_splits = json.load(f)
    LowestError = 0

    # ------------------------------------------------------------------------------------------




    # -------------------------    loading previous results   ------------------------
    net = MetricLearningCnn(CnnMode,DropoutP)

    net,optimizer,LowestError,StartEpoch,scheduler,LodedNegativeMiningMode = LoadModel(net, StartBestModel, ModelsDirName, BestFileName, UseBestScore, device, load_epoch)
    print('LoadedNegativeMiningMode: ' + LodedNegativeMiningMode)



    if NumGpus > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device)
    net.eval()
    StepSize = 800
    TotalTestError = evaluate_hpatch(TestData, 1, CnnMode, device, StepSize, hpatch_splits,
                                     hpatch_taskdir)

    print('Finished Evaluation')