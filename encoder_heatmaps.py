import pickle
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
import ntpath
import pathlib
# my classes
from hpatches.utils.hpatch import hpatch_descr, hpatch_sequence
from hpatches.utils.results import results_verification, results_matching, results_retrieval
from hpatches.utils.tasks import eval_verification, eval_matching, eval_retrieval
from network.my_classes import imshow, ShowRowImages, ShowTwoRowImages, EvaluateDualNets, EvaluateSingleNet
from network.my_classes import FPR95Accuracy,SingleNet, MetricLearningCnn, EvaluateNet,EvaluateDualNets
from network.generator import DatasetPairwiseTriplets,NormalizeImages
from network.losses import ContrastiveLoss, TripletLoss, OnlineTripletLoss, OnlineHardNegativeMiningTripletLoss
from network.losses import InnerProduct, FindFprTrainingSet, FPRLoss, PairwiseLoss, HardTrainingLoss
from network.losses import Compute_FPR_HardNegatives, ComputeFPR
from util.warmup_scheduler import GradualWarmupSchedulerV2
from util.read_matlab_imdb import read_matlab_imdb
from util.utils import LoadModel,MultiEpochsDataLoader,MyGradScaler, save_best_model_stats
from network.nt_xent import NTXentLoss
from hpatches.utils.load_dataset import load_dataset as load_hpatches_dataset
import h5py
import warnings
from run_match_patches3 import load_datasets_paths
warnings.filterwarnings("ignore", message="UserWarning: albumentations.augmentations.transforms.RandomResizedCrop")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from skimage.filters import gaussian

def display(rgb, Emb1Attention, nir_orig, Emb2Attention):
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].imshow(rgb)
    ax[0, 0].axis('off')
    ax[0, 0].set_title('Input Image')

    ax[0, 1].imshow(rgb)
    ax[0, 1].imshow(Emb1Attention, alpha=0.25, cmap='jet')
    ax[0, 1].axis('off')
    ax[0, 1].set_title('Attention')

    ###############

    ax[1, 0].imshow(nir_orig, cmap="gray")
    ax[1, 0].axis('off')
    ax[1, 0].set_title('Input Image')

    ax[1, 1].imshow(nir_orig, cmap="gray")
    ax[1, 1].imshow(Emb2Attention, alpha=0.25, cmap='jet')
    ax[1, 1].axis('off')
    ax[1, 1].set_title('Attention')

    plt.show()

def get_file_name(path):
    head, tail = ntpath.split(path)
    fname = tail or ntpath.basename(head)
    return fname.split('.')[0]

def save_image(img, fname, out_folder):
    fname += '.png'
    mpimg.imsave(os.path.join(out_folder, fname), img)

if __name__ == '__main__':
    # np.random.seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#"cuda:0"
    NumGpus = torch.cuda.device_count()
    torch.cuda.empty_cache()
    GPUtil.showUtilization()

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    name = torch.cuda.get_device_name(0)

    ModelsDirName = './artifacts/symmetric_enc_transformer_visnir_10/models/'
    BestFileName = 'best_model'
    FileName = 'model_epoch_'
    ds_name = 'VisNir'
    TrainFile, TestDir = load_datasets_paths(ds_name)
    TestDecimation = 1
    FPR95 = 0.8

    # ----------------------------     configuration   ---------------------------
    Augmentation = {}
    Augmentation["HorizontalFlip"] = False
    Augmentation["VerticalFlip"] = False
    Augmentation["Rotate90"] = False
    Augmentation["Test"] = {'Do': False}
    Augmentation["RandomCrop"] = {'Do': False, 'MinDx': 0, 'MaxDx': 0.2, 'MinDy': 0, 'MaxDy': 0.2}


    TestMode = False
    use_validation = True
    FreezeSymmetricBlock = False

    # torch.manual_seed(0)
    # np.random.seed(0)

    GeneratorMode = 'Pairwise'

    CnnMode = 'SymmetricAttention'
    NegativeMiningMode = 'Random'

    # criterion = OnlineTripletLoss(margin=1)
    criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode=NegativeMiningMode,device=device)
    #criterion         = OnlineHaOnlineHardNegativeMiningTripletLossrdNegativeMiningTripletLoss(margin=1, Mode='HardPos', MarginRatio=0.5)
    #criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='HardPos', MarginRatio=1.0/2, PosRatio=1. / 2)
    Description = 'Symmetric Hardest'


    LearningRate = 1e-1#0.1

    weight_decay = 0#1e-5
    DropoutP = 0.5

    # torch.manual_seed(0)
    # np.random.seed(0)

    OuterBatchSize = 4*12
    InnerBatchSize = 2*12
    PrintStep = 100

    StartBestModel = True
    UseBestScore   = False






    # ----------------------------- read data----------------------------------------------
    Data = read_matlab_imdb(TrainFile)
    TrainingSetData = Data['Data']
    TrainingSetLabels = np.squeeze(Data['Labels'])
    if use_validation:
        TrainingSetSet = np.squeeze(Data['Set'])
    del Data

    TrainingIdx = np.squeeze(np.asarray(np.where(TrainingSetSet == 1)))

    TrainingSetLabels = torch.from_numpy(TrainingSetLabels[TrainingIdx])

    TrainingSetData = TrainingSetData[TrainingIdx, :, :, :].astype(np.float32)
    rgb_mean = TrainingSetData[:, :, :, :, 0].mean()
    nir_mean = TrainingSetData[:, :, :, :, 1].mean()
    TrainingSetData[:, :, :, :, 0] -= rgb_mean
    TrainingSetData[:, :, :, :, 1] -= nir_mean
    TrainingSetData = torch.from_numpy(NormalizeImages(TrainingSetData))

    # -------------------------    loading previous results   ------------------------
    output_attention_weights = True
    net = MetricLearningCnn(CnnMode,DropoutP, output_attention_weights)
    # optimizer = torch.optim.Adam(net.parameters(), lr=LearningRate)


    StartEpoch = 0

    net,optimizer,LowestError,StartEpoch,scheduler,LodedNegativeMiningMode = LoadModel(net, StartBestModel, ModelsDirName, BestFileName, UseBestScore, device)
    print('LodaedNegativeMiningMode: ' + LodedNegativeMiningMode)

    if NumGpus > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device)

    print(CnnMode + ' evaluating\n')

    # bar = tqdm(TrainingSetData, 0, leave=False)
    # for i, Data in enumerate(bar):
        #
        # # get the inputs
        # pos1 = Data['pos1']
        # pos2 = Data['pos2']
        #
        # pos1 = np.reshape(pos1, (pos1.shape[0] * pos1.shape[1], 1, pos1.shape[2], pos1.shape[3]), order='F')
        # pos2 = np.reshape(pos2, (pos2.shape[0] * pos2.shape[1], 1, pos2.shape[2], pos2.shape[3]), order='F')
        #

        # pos1, pos2 = pos1.to(device), pos2.to(device)

    weights_path = 'artifacts/attention_weights/weights.pickle'
    if os.path.isfile(weights_path):
        with open(weights_path, 'rb') as handle:
            weights = pickle.load(handle)
    else:
        StepSize = 800
        net.eval()
        # data = TrainingSetData[13:14]
        # data = TrainingSetData[13:14]
        rgb_path = "D:\\multisensor\\datasets\\Vis-Nir\\data\\street\\0007_rgb.tiff"
        nir_path = "D:\\multisensor\\datasets\\Vis-Nir\\data\\street\\0007_nir.tiff"

        rgb = mpimg.imread(rgb_path)
        rgb_gray_orig = cv2.imread(rgb_path)
        rgb_gray_orig = cv2.cvtColor(rgb_gray_orig, cv2.COLOR_BGR2GRAY)
        nir_orig = mpimg.imread(nir_path)
        outpath = "D:\\multisensor\\attentions\\street"
        pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)

        # Emb = EvaluateDualNets(net, data[:,:,:,:,0], data[:,:,:,:,1], CnnMode,device, StepSize)
        rgb_gray = rgb_gray_orig.copy().reshape(1, 1, rgb_gray_orig.shape[0], rgb_gray_orig.shape[1])
        nir = nir_orig.copy().reshape(1, 1, nir_orig.shape[0], nir_orig.shape[1])
        rgb_gray = torch.from_numpy(NormalizeImages(rgb_gray.astype(np.float32)))
        nir = torch.from_numpy(NormalizeImages(nir.astype(np.float32)))
        Emb = EvaluateDualNets(net, rgb_gray, nir, CnnMode, device, StepSize)
        Emb1Attention = np.array(Emb['Emb1Attention'])
        Emb2Attention = np.array(Emb['Emb2Attention'])

        # Emb1Attention = np.sqrt((Emb1Attention.reshape(8,8,-1) ** 2).sum(axis=2))
        # Emb1Attention = Emb1Attention.repeat(8, axis=0).repeat(8, axis=1)
        # Emb2Attention = np.sqrt((Emb2Attention.reshape(8, 8, -1) ** 2).sum(axis=2))
        # Emb2Attention = Emb2Attention.repeat(8, axis=0).repeat(8, axis=1)

        # TrainingSetData *= 255.0 / 2
        # TrainingSetData[:, :, :, :, 0] += rgb_mean
        # TrainingSetData[:, :, :, :, 1] += nir_mean
        # img = TrainingSetData[14,:,:,:,0].reshape(64,64)
        # img = rgb.copy()
        # img = 255 * (img - img.min()) / (img.max() - img.min())
        # img = cv2.resize(np.uint8(img), (300,300))
        # img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
        # Emb1Attention = np.mean(Emb1Attention.reshape(128, 8, 8), axis=0)
        Emb1Attention = np.sqrt((Emb1Attention.reshape(128, 8, 8) ** 2).sum(axis=0))
        Emb1Attention = 255 * (Emb1Attention - Emb1Attention.min()) / (Emb1Attention.max() - Emb1Attention.min())
        Emb1Attention = np.uint8(Emb1Attention)
        Emb1Attention = cv2.resize(Emb1Attention, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)

        # Emb2Attention = np.mean(Emb2Attention.reshape(128, 8, 8), axis=0)
        Emb2Attention = np.sqrt((Emb2Attention.reshape(128, 8, 8) ** 2).sum(axis=0))
        Emb2Attention = 255 * (Emb2Attention - Emb2Attention.min()) / (Emb2Attention.max() - Emb2Attention.min())
        Emb2Attention = np.uint8(Emb2Attention)
        Emb2Attention = cv2.resize(Emb2Attention, (nir_orig.shape[1], nir_orig.shape[0]), interpolation=cv2.INTER_CUBIC)

        display(rgb, Emb1Attention, nir_orig, Emb2Attention)
        # cv2.waitKey(0)
        plt.close()

        dpi = 80
        figsize = rgb.shape[1] / dpi, rgb.shape[0] / dpi
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(rgb)
        plt.axis('off')
        plt.savefig(os.path.join(outpath, get_file_name(rgb_path) + '.png'))
        plt.close()
        figsize = rgb.shape[1] / dpi, rgb.shape[0] / dpi
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(rgb)
        ax.imshow(Emb1Attention, alpha=0.25, cmap='jet')
        plt.axis('off')
        plt.savefig(os.path.join(outpath, get_file_name(rgb_path) + '_attention' + '.png'))
        plt.close()
        figsize = nir_orig.shape[1] / dpi, nir_orig.shape[0] / dpi
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(nir_orig, cmap="gray")
        plt.savefig(os.path.join(outpath, get_file_name(nir_path) + '.png'))
        plt.axis('off')
        plt.close()
        figsize = nir_orig.shape[1] / dpi, nir_orig.shape[0] / dpi
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(nir_orig, cmap="gray")
        ax.imshow(Emb2Attention, alpha=0.25, cmap='jet')
        plt.axis('off')
        plt.savefig(os.path.join(outpath, get_file_name(nir_path) + '_attention' + '.png'))
        plt.close()

        # with open('artifacts/attention_weights/weights.pickle', 'wb') as handle:
        #     weights = {}
        #     weights['Emb1Attention'] = Emb['Emb1Attention']
        #     weights['Emb2Attention'] = Emb['Emb2Attention']
        #     pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Dist = np.power(Emb['Emb1'] - Emb['Emb2'], 2).sum(1)
    # PosIdx = np.squeeze(np.asarray(np.where(TrainingSetLabels == 1)))
    # PosDist = np.sort(Dist[PosIdx])
    print('Finished evaluation')
