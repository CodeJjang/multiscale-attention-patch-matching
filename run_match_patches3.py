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
import math
from util.warmup_scheduler import GradualWarmupSchedulerV2

# my classes
from network.my_classes import imshow, ShowRowImages, ShowTwoRowImages, EvaluateSofmaxNet, EvaluateDualNets
from network.my_classes import DatasetPairwiseTriplets, FPR95Accuracy
from network.my_classes import SingleNet, MetricLearningCnn, EvaluateNet, SiamesePairwiseSoftmax, NormalizeImages
from network.losses import ContrastiveLoss, TripletLoss, OnlineTripletLoss, OnlineHardNegativeMiningTripletLoss
from network.losses import InnerProduct, FindFprTrainingSet, FPRLoss, PairwiseLoss, HardTrainingLoss
from network.losses import Compute_FPR_HardNegatives, ComputeFPR

from util.read_matlab_imdb import read_matlab_imdb
from util.utils import LoadModel,MultiEpochsDataLoader

import warnings
warnings.filterwarnings("ignore", message="UserWarning: albumentations.augmentations.transforms.RandomResizedCrop")



if __name__ == '__main__':
    np.random.seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NumGpus = torch.cuda.device_count()
    torch.cuda.empty_cache()
    GPUtil.showUtilization()

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    name = torch.cuda.get_device_name(0)

    ModelsDirName = './models3/'
    LogsDirName = './logs3/'
    Description = 'Symmetric CNN with Triplet loss, no HM'
    BestFileName = 'visnir_best'
    FileName = 'visnir_sym_triplet'
    # TestDir = '/home/keller/Dropbox/multisensor/python/data/test/'
    TestDir = 'F:\\multisensor\\test\\'
    # TestDir = 'data\\Vis-Nir_grid\\test\\'
    # TrainFile = '/home/keller/Dropbox/multisensor/python/data/Vis-Nir_Train.mat'
    TrainFile = 'f:\\multisensor\\train\\Vis-Nir_Train.hdf5'
    # TrainFile = './data/Vis-Nir_grid/Vis-Nir_grid_Train.hdf5'
    TestDecimation = 10
    FPR95 = 0.8
    MaxNoImages = 400
    FprHardNegatives = False

    writer = SummaryWriter(LogsDirName)
    LowestError = 1e10

    # ----------------------------     configuration   ---------------------------
    Augmentation = {}
    Augmentation["HorizontalFlip"] = False
    Augmentation["VerticalFlip"] = False
    Augmentation["Rotate90"] = True
    Augmentation["Test"] = {'Do': True}
    Augmentation["RandomCrop"] = {'Do': False, 'MinDx': 0, 'MaxDx': 0.2, 'MinDy': 0, 'MaxDy': 0.2}

    # default values
    FreezeSymmetricCnn = True
    FreezeAsymmetricCnn = True

    AssymetricInitializationPhase = False

    TestMode = False
    TestDecimation = 10

    FreezeSymmetricBlock = False

    if True:
        GeneratorMode = 'Pairwise'
        CnnMode = 'PairwiseSymmetric'
        CnnMode = 'PairwiseSymmetricAttention'
        NegativeMiningMode = 'Random'
        #NegativeMiningMode = 'Hardest'
        #NegativeMiningMode = 'HardPos'
        criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode=NegativeMiningMode,device=device)
        #criterion         = OnlineHaOnlineHardNegativeMiningTripletLossrdNegativeMiningTripletLoss(margin=1, Mode='HardPos', MarginRatio=0.5)
        #criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='HardPos', MarginRatio=1.0/2, PosRatio=1. / 2)
        Description = 'PairwiseSymmetric Hardest'

        InitializeOptimizer = True
        UseWarmUp           = True

        StartBestModel      = False
        UseBestScore        = False

        LearningRate =  1e-1

        weight_decay = 0#1e-5
        DropoutP = 0.5

        OuterBatchSize = 4*12
        InnerBatchSize = 2*12
        Augmentation["Test"] = {'Do': False}
        Augmentation["HorizontalFlip"] = True
        Augmentation["VerticalFlip"] = True
        Augmentation["Rotate90"] = True
        Augmentation["RandomCrop"] = {'Do': True, 'MinDx': 0, 'MaxDx': 0.2, 'MinDy': 0, 'MaxDy': 0.2}


        FreezeSymmetricCnn  = False
        FreezeSymmetricBlock = False

        FreezeAsymmetricCnn = True


        FprHardNegatives = False

        StartBestModel = False
        UseBestScore   = False

    if False:
        GeneratorMode = 'Pairwise'
        CnnMode = 'PairwiseAsymmetric'
        NegativeMiningMode = 'Random'
        #NegativeMiningMode = 'Hardest'
        criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode=NegativeMiningMode)
        # criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='HardPos', MarginRatio=1.0/2, PosRatio=1. / 2)

        InitializeOptimizer = False
        UseWarmUp           = True
        StartBestModel      = False
        UseBestScore        = True

        FreezeSymmetricCnn   = True
        FreezeAsymmetricCnn  = False

        LearningRate = 1e-1
        OuterBatchSize = 24;
        InnerBatchSize = 6

        FprHardNegatives = False

        weight_decay = 0
        DropoutP = 0.5

        Augmentation["Test"] = {'Do': False}
        Augmentation["HorizontalFlip"] = True
        Augmentation["VerticalFlip"] = True
        Augmentation["Rotate90"] = True
        Augmentation["RandomCrop"] = {'Do': True, 'MinDx': 0, 'MaxDx': 0.2, 'MinDy': 0, 'MaxDy': 0.2}

        #AssymetricInitializationPhase = True
        Description = 'PairwiseAsymmetric'


    if False:
        GeneratorMode = 'Pairwise'
        # CnnMode            = 'HybridRot'
        CnnMode = 'Hybrid'

        criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode="Random",device=device)
        #criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode="Hardest",device=device)
        #criterion        = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='HardPos', MarginRatio=1.0/4, PosRatio=1./4)
        #HardestCriterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='Hardest')

        #criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='HardPos', MarginRatio=1.0 / 2, PosRatio=1. / 2,device=device)

        PairwiseLoss      = PairwiseLoss()

        InitializeOptimizer = True
        UseWarmUp = True
        StartBestModel = False
        UseBestScore = False

        OuterBatchSize = 12*2#24  # 24
        InnerBatchSize = 12*2#24  # 24
        LearningRate =  0#1e-1
        DropoutP = 0.0
        weight_decay= 0#1e-5

        TestMode = False
        TestDecimation = 10

        FprHardNegatives = False

        FreezeSymmetricCnn  = False
        FreezeAsymmetricCnn = False

        AssymetricInitializationPhase = False

        Augmentation["Test"] = {'Do': False}
        Augmentation["HorizontalFlip"] = True
        Augmentation["VerticalFlip"] = True
        Augmentation["Rotate90"] = True
        Augmentation["RandomCrop"] = {'Do': True, 'MinDx': 0, 'MaxDx': 0.2, 'MinDy': 0, 'MaxDy': 0.2}


        MaxNoImages = 400




    # ----------------------------- read data----------------------------------------------
    Data = read_matlab_imdb(TrainFile)
    TrainingSetData = Data['Data']
    TrainingSetLabels = np.squeeze(Data['Labels'])
    TrainingSetSet = np.squeeze(Data['Set'])
    del Data


    TrainIdx = np.squeeze(np.asarray(np.where(TrainingSetSet == 1)))
    ValIdx = np.squeeze(np.asarray(np.where(TrainingSetSet == 3)))

    # VALIDATION data
    ValSetLabels = torch.from_numpy(TrainingSetLabels[ValIdx])

    ValSetData = TrainingSetData[ValIdx, :, :, :].astype(np.float32)
    ValSetData[:, :, :, :, 0] -= ValSetData[:, :, :, :, 0].mean()
    ValSetData[:, :, :, :, 1] -= ValSetData[:, :, :, :, 1].mean()
    ValSetData = torch.from_numpy(NormalizeImages(ValSetData));





    # TRAINING data
    TrainingSetData = np.squeeze(TrainingSetData[TrainIdx,])
    TrainingSetLabels = TrainingSetLabels[TrainIdx]

    # define generators
    Training_Dataset = DatasetPairwiseTriplets(TrainingSetData, TrainingSetLabels, InnerBatchSize, Augmentation, GeneratorMode)
    # Training_DataLoader = data.DataLoader(Training_Dataset, batch_size=OuterBatchSize, shuffle=True,num_workers=8,pin_memory=True)
    Training_DataLoader = MultiEpochsDataLoader(Training_Dataset, batch_size=OuterBatchSize, shuffle=True,
                                                num_workers=8, pin_memory=True)




    # Load all TEST datasets
    FileList = glob.glob(TestDir + "*.hdf5")
    TestData = dict()
    for File in FileList:
        path, DatasetName = os.path.split(File)
        DatasetName = os.path.splitext(DatasetName)[0]

        Data = read_matlab_imdb(File)

        x = Data['Data'].astype(np.float32)
        TestLabels = torch.from_numpy(np.squeeze(Data['Labels']))
        del Data

        x[:, :, :, :, 0] -= x[:, :, :, :, 0].mean()
        x[:, :, :, :, 1] -= x[:, :, :, :, 1].mean()

        x = NormalizeImages(x)
        x = torch.from_numpy(x)

        # TestLabels = torch.from_numpy(2 - Data['testLabels'])

        TestData[DatasetName] = dict()
        TestData[DatasetName]['Data'] = x
        TestData[DatasetName]['Labels'] = TestLabels
        del x
    # ------------------------------------------------------------------------------------------




    # -------------------------    loading previous results   ------------------------
    net = MetricLearningCnn(CnnMode,DropoutP)
    optimizer = torch.optim.Adam(net.parameters(), lr=LearningRate)


    StartEpoch = 0

    net,optimizer,LowestError,StartEpoch,scheduler,LodedNegativeMiningMode =  LoadModel(net, StartBestModel, ModelsDirName, BestFileName, UseBestScore, device)
    print('LodedNegativeMiningMode: ' + LodedNegativeMiningMode)

    # -------------------------------------  freeze layers --------------------------------------
    net.FreezeSymmetricCnn(FreezeSymmetricCnn)
    net.FreezeSymmetricBlock(FreezeSymmetricBlock)

    net.FreezeAsymmetricCnn(FreezeAsymmetricCnn)
    # ------------------------------------------------------------------------------------------

    # -------------------- Initialization -----------------------
    if AssymetricInitializationPhase:
        net.netAS1 = copy.deepcopy(net.module.netS)
        net.netAS2 = copy.deepcopy(net.module.netS)


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device)

    if InitializeOptimizer:

        optimizer = torch.optim.Adam(
            [{'params': filter(lambda p: p.requires_grad == True, net.parameters()),'lr': LearningRate, 'weight_decay': weight_decay},
             {'params': filter(lambda p: p.requires_grad == False, net.parameters()),'lr': 0, 'weight_decay': 0}],
            lr=0, weight_decay=0.00)

        # scheduler = StepLR(optimizer, step_size=10, gamma=math.sqrt(0.1))

    # ------------------------------------------------------------------------------------------





    ########################################################################
    # Train the network
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    #LRscheduler =  StepLR(optimizer, step_size=10, gamma=0.1)


    if UseWarmUp:
        WarmUpEpochs = 4
        scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=1, total_epoch=WarmUpEpochs, after_scheduler= StepLR(optimizer, step_size=10, gamma=0.1))
    else:
        WarmUpEpochs = 0

    InnerProductLoss = InnerProduct()
    CeLoss = nn.CrossEntropyLoss()


    print(CnnMode + ' training\n')


    # writer.add_graph(net, images)
    for epoch in range(StartEpoch, 1000):  # loop over the dataset multiple times

        running_loss_pos = 0
        running_loss_neg = 0
        optimizer.zero_grad()

        print('\n' + colored('Gain1 = ' +repr(net.module.Gain1.item())[0:6], 'cyan', attrs=['reverse', 'blink']))
        print('\n' + colored('Gain2 = ' +repr(net.module.Gain2.item())[0:6], 'cyan', attrs=['reverse', 'blink']))


        #warmup
        if InitializeOptimizer and (epoch - StartEpoch < WarmUpEpochs) and UseWarmUp:
            print(colored('\n Warmup step #' + repr(epoch - StartEpoch), 'green', attrs=['reverse', 'blink']))
            #print('\n Warmup step #' + repr(epoch - StartEpoch))
            scheduler_warmup.step()
        else:
            if epoch > StartEpoch:
                print('CurrentError=' + repr(TotalTestError)[0:8])

                if type(scheduler).__name__ == 'StepLR':
                    scheduler.step()

                if type(scheduler).__name__ == 'ReduceLROnPlateau':
                    scheduler.step(TotalTestError)

        running_loss = 0
        #scheduler_warmup.step(epoch-StartEpoch,running_loss)

        str = '\n LR: '
        for param_group in optimizer.param_groups:
            str += repr(param_group['lr']) + ' '
        print(colored(str, 'blue', attrs=['reverse', 'blink']))

        print('FreezeSymmetricCnn  = ' + repr(FreezeSymmetricCnn) + '\nFreezeAsymmetricCnn = '+repr(FreezeAsymmetricCnn) + '\n')
        print('NegativeMiningMode='+criterion.Mode)

        Case1 = (criterion.Mode == 'Random') and (optimizer.param_groups[0]['lr'] <= (1e-4 + 1e-8)) and (epoch-StartEpoch>WarmUpEpochs)
        Case2 = (CnnMode == 'Hybrid') and (criterion.Mode == 'Hardest') and (optimizer.param_groups[0]['lr'] <= (1e-4 +1e-8)) and (FreezeSymmetricCnn==True)
        if Case1 or Case2:
            if Case1:
                #print('Switching Random->Hardest')
                print(colored('Switching Random->Hardest', 'green', attrs=['reverse', 'blink']))
                criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode = 'Hardest',device=device)

                #if CnnMode == 'Hybrid':
                LearningRate = 1e-1
                optimizer = torch.optim.Adam(
                    [{'params': filter(lambda p: p.requires_grad == True, net.parameters()), 'lr': LearningRate,
                      'weight_decay': weight_decay},
                     {'params': filter(lambda p: p.requires_grad == False, net.parameters()), 'lr': 0,
                      'weight_decay': 0}],
                    lr=0, weight_decay=0.00)

                if type(scheduler).__name__ == 'StepLR':
                    scheduler =  StepLR(optimizer, step_size=10, gamma=0.1)

                if type(scheduler).__name__ == 'ReduceLROnPlateau':
                    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)



            if Case2:
                print(colored('Hybrid: unfreezing symmetric and assymetric\n', 'green', attrs=['reverse', 'blink']))
                #print('Hybrid: unfreezing symmetric and assymetric\n')
                if NumGpus == 1:
                    net.FreezeSymmetricCnn(False)
                    net.FreezeAsymmetricCnn(False)
                else:
                    net.module.FreezeSymmetricCnn(False)
                    net.module.FreezeAsymmetricCnn(False)

                FreezeSymmetricCnn  = False
                FreezeAsymmetricCnn = False


        bar = tqdm(Training_DataLoader, 0, leave=False)
        for i, Data in enumerate(bar):
        #for i, Data in enumerate(tqdm(Training_DataLoader, 0)):

            net = net.train()

            # get the inputs
            pos1 = Data['pos1']
            pos2 = Data['pos2']

            pos1 = np.reshape(pos1, (pos1.shape[0] * pos1.shape[1], 1, pos1.shape[2], pos1.shape[3]), order='F')
            pos2 = np.reshape(pos2, (pos2.shape[0] * pos2.shape[1], 1, pos2.shape[2], pos2.shape[3]), order='F')



            if (CnnMode == 'PairwiseAsymmetric') | (CnnMode == 'PairwiseSymmetric') | (CnnMode == 'PairwiseSymmetricAttention'):

                if FprHardNegatives:
                    Embed = Compute_FPR_HardNegatives(net, pos1, pos2, device, FprValPos=0.7*FPR95,
                                                           FprValNeg=1.5 * FPR95, MaxNoImages=MaxNoImages)
                    pos1 = Embed['PosIdx1']
                    pos2 = Embed['PosIdx2']

                pos1, pos2     = pos1.to(device), pos2.to(device)
                Embed = net(pos1, pos2,DropoutP=DropoutP)
                loss           = criterion(Embed['Emb1'], Embed['Emb2']) + criterion(Embed['Emb2'], Embed['Emb1'])


            if CnnMode == 'Hybrid':

                # GPUtil.showUtilization()
                if FprHardNegatives:

                    Embed = Compute_FPR_HardNegatives(net, pos1, pos2, device, FprValPos=0.7 * FPR95,
                                                      FprValNeg=1.3*FPR95, MaxNoImages=MaxNoImages)


                    Embed['PosIdx1'], Embed['PosIdx2'] = Embed['PosIdx1'].to(device), Embed['PosIdx2'].to(device)
                    EmbedPos = net(Embed['PosIdx1'], Embed['PosIdx2'])
                    loss = PairwiseLoss(EmbedPos['Hybrid1'], EmbedPos['Hybrid2'])
                    pos_loss = loss.item()

                    # loss = HardestCriterion(EmbedPos['Hybrid1'], EmbedPos['Hybrid2'])

                    if (Embed['NegIdxA1'].nelement() > 1) & (Embed['NegIdxA1'].shape[0] > 1):
                        Embed['NegIdxA1'], Embed['NegIdxA2'] = Embed['NegIdxA1'].to(device), Embed['NegIdxA2'].to(
                            device)
                        EmbedNegA = net(Embed['NegIdxA1'], Embed['NegIdxA2'])

                        neg_loss1 = PairwiseLoss(EmbedNegA['Hybrid1'], EmbedNegA['Hybrid2'])
                        loss     -= neg_loss1
                        neg_loss  = neg_loss1.item()

                    #loss -= PairwiseLoss(EmbedNegA['EmbAsym1'], EmbedNegA['EmbAsym2'])

                        #del EmbedNegA

                    if (Embed['NegIdxB1'].nelement() > 1) & (Embed['NegIdxB1'].shape[0] > 1):
                        Embed['NegIdxB1'], Embed['NegIdxB2'] = Embed['NegIdxB1'].to(device), Embed['NegIdxB2'].to(
                            device)
                        EmbedNegB = net(Embed['NegIdxB1'], Embed['NegIdxB2'])

                        neg_loss2 = PairwiseLoss(EmbedNegB['Hybrid1'], EmbedNegB['Hybrid2'])
                        loss     -= neg_loss2
                        neg_loss += neg_loss2.item()


                    del Embed, EmbedPos

                    running_loss_neg += neg_loss
                    running_loss_pos += pos_loss
                else:
                        pos1, pos2 = pos1.to(device), pos2.to(device)


                        #loss  = HardTrainingLoss(net, pos1, pos2,PosRatio=0.25,MarginRatio=0.25,T=1,device=device)
                        #loss += HardTrainingLoss(net, pos2, pos1, PosRatio=0.25, MarginRatio=0.25, T=1, device=device)

                        Embed = net(pos1, pos2,DropoutP=DropoutP)
                        loss = criterion(Embed['Hybrid1'], Embed['Hybrid2']) + criterion(Embed['Hybrid2'],Embed['Hybrid1'])
                        #loss += Random(Embed['Hybrid1'], Embed['Hybrid2'])    + Random(Embed['Hybrid2'],Embed['Hybrid1'])
                        #loss += Hardest(Embed['Hybrid1'], Embed['Hybrid2']) + Hardest(Embed['Hybrid2'],Embed['Hybrid1'])
                        #loss  = HardTrainingLoss(net, pos1, pos2, PosRatio=1, MarginRatio=1.0/2, T=1, device=device)
                        #loss += HardTrainingLoss(net, pos2, pos1, PosRatio=1, MarginRatio=1.0/4, T=1, device=device)

                        #loss += InnerProductLoss(Embed['EmbAsym1'], Embed['EmbSym1']) + InnerProductLoss(Embed['EmbAsym2'],Embed['EmbSym2'])
                        loss += criterion(Embed['EmbSym1'], Embed['EmbSym2']) + criterion(Embed['EmbSym2'], Embed['EmbSym1'])
                        #loss +=criterion(Embed['EmbAsym1'], Embed['EmbAsym2'])+criterion(Embed['EmbAsym2'], Embed['EmbAsym1'])
                        #loss += loss1

                        #TrainFpr = ComputeFPR(Embed['Hybrid1'], Embed['Hybrid2'], FPR95 * 0.9, FPR95 * 1.1)
                        # print('TrainFpr = ' + repr(TrainFpr))



            # backward + optimize
            loss.backward()

            optimizer.step()  # Now we can do an optimizer step

            # zero the parameter gradients
            optimizer.zero_grad()

            running_loss     += loss.item()

            SchedularUpadteInterval = 200
            if (i % SchedularUpadteInterval == 0) &(i>0):
                print('running_loss: '+repr(running_loss/i)[0:8])




            PrintStep = 1000
            if (((i % PrintStep == 0) or (i * InnerBatchSize >= len(Training_DataLoader) - 1)) and (i > 0)) or TestMode:

                if i > 0:
                    running_loss     /=i
                    running_loss_neg /= i
                    running_loss_pos /= i

                    #print('running_loss_neg: ' + repr(100*running_loss_neg)[0:5] + ' running_loss_pos: ' + repr(100*running_loss_pos)[0:5])

                # val accuracy
                net.eval()
                StepSize = 800
                EmbVal1 = EvaluateNet(net.module.GetChannelCnn(0, CnnMode), ValSetData[:, :, :, :, 0], device,StepSize)
                EmbVal2 = EvaluateNet(net.module.GetChannelCnn(1, CnnMode), ValSetData[:, :, :, :, 1], device,StepSize)
                Dist = np.power(EmbVal1 - EmbVal2, 2).sum(1)
                ValError = FPR95Accuracy(Dist, ValSetLabels) * 100

                del EmbVal1, EmbVal2

                # estimate fpr95 threshold
                PosValIdx = np.squeeze(np.asarray(np.where(ValSetLabels == 1)))
                CurrentFPR95 = np.sort(Dist[PosValIdx])[int(0.95 * PosValIdx.shape[0])]
                if i > 0:
                    print('FPR95: ' + repr(CurrentFPR95)[0:4] + ' Loss= ' + repr(running_loss)[0:6])

                if (net.module.Mode == 'Hybrid1') | (net.module.Mode == 'Hybrid2'):
                    net.module.Mode = 'Hybrid'

                if (net.module.Mode == 'PairwiseSymmetricAttention1') | (net.module.Mode == 'PairwiseSymmetricAttention2'):
                    net.module.Mode = 'PairwiseSymmetricAttention'

                print('FPR95 changed: ' + repr(FPR95)[0:5])

                # compute stats


                if i >= len(Training_DataLoader):
                    TestDecimation1 = 1
                else:
                    TestDecimation1 = TestDecimation;

                # test accuracy
                NoSamples = 0
                TotalTestError = 0
                for DataName in TestData:
                    EmbTest1 = EvaluateNet(net.module.GetChannelCnn(0, CnnMode),
                                           TestData[DataName]['Data'][0::TestDecimation1, :, :, :, 0], device,
                                           StepSize)
                    EmbTest2 = EvaluateNet(net.module.GetChannelCnn(1, CnnMode),
                                           TestData[DataName]['Data'][0::TestDecimation1, :, :, :, 1], device,
                                           StepSize)

                    Dist = np.power(EmbTest1 - EmbTest2, 2).sum(1)
                    TestData[DataName]['TestError'] = FPR95Accuracy(Dist, TestData[DataName]['Labels'][
                                                                          0::TestDecimation1]) * 100
                    TotalTestError += TestData[DataName]['TestError'] * TestData[DataName]['Data'].shape[0]
                    NoSamples += TestData[DataName]['Data'].shape[0]
                TotalTestError /= NoSamples

                del EmbTest1, EmbTest2

                if (net.module.Mode == 'Hybrid1') | (net.module.Mode == 'Hybrid2'):
                    net.module.Mode = 'Hybrid'

                if (net.module.Mode == 'PairwiseSymmetricAttention1') | (net.module.Mode == 'PairwiseSymmetricAttention2'):
                    net.module.Mode = 'PairwiseSymmetricAttention'

                state = {'epoch': epoch,
                         'state_dict': net.module.state_dict(),
                         'optimizer_name': type(optimizer).__name__,
                         #'optimizer': optimizer.state_dict(),
                         'optimizer': optimizer,
                         'scheduler_name': type(scheduler).__name__,
                         #'scheduler': scheduler.state_dict(),
                         'scheduler': scheduler,
                         'Description': Description,
                         'LowestError': LowestError,
                         'OuterBatchSize': OuterBatchSize,
                         'InnerBatchSize': InnerBatchSize,
                         'Mode': net.module.Mode,
                         'CnnMode': CnnMode,
                         'NegativeMiningMode': criterion.Mode,
                         'GeneratorMode': GeneratorMode,
                         'Loss': criterion.Mode,
                         'FPR95': FPR95}

                if (TotalTestError < LowestError):
                    LowestError = TotalTestError

                    print(colored('Best error found and saved: ' + repr(LowestError)[0:5], 'red', attrs=['reverse', 'blink']))
                    #print('Best error found and saved: ' + repr(LowestError)[0:5])
                    filepath = ModelsDirName + BestFileName + '.pth'
                    torch.save(state, filepath)


                str = '[%d, %5d] loss: %.3f' % (epoch, i, 100 * running_loss) + ' Val Error: ' + repr(ValError)[0:6]


                # for DataName in TestData:
                #   str +=' ' + DataName + ': ' + repr(TestData[DataName]['TestError'])[0:6]
                str += ' FPR95 = ' + repr(FPR95)[0:6] + ' Mean: ' + repr(TotalTestError)[0:6]
                print(str)

                if False:
                    writer.add_scalar('Val Error', ValError, epoch * len(Training_DataLoader) + i)
                    writer.add_scalar('Test Error', TotalTestError, epoch * len(Training_DataLoader) + i)
                    writer.add_scalar('Loss', 100 * running_loss, epoch * len(Training_DataLoader) + i)
                    writer.add_scalar('FPR95', FPR95, epoch * len(Training_DataLoader) + i)
                    writer.add_text('Text', str)
                    writer.close()


                # save epoch
                filepath = ModelsDirName + FileName + repr(epoch) + '.pth'
                torch.save(state, filepath)

            if (i * InnerBatchSize) > (len(Training_DataLoader) - 1):
                bar.clear()
                bar.close()
                break

    print('Finished Training')