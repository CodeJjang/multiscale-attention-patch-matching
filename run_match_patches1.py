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
import GPUtil
from multiprocessing import Pool

# my classes
from my_classes import imshow, ShowRowImages, ShowTwoRowImages, EvaluateSofmaxNet, EvaluateDualNets
from my_classes import DatasetPairwiseTriplets, FPR95Accuracy
from my_classes import SingleNet, MetricLearningCnn, EvaluateNet, SiamesePairwiseSoftmax, NormalizeImages
from losses import ContrastiveLoss, TripletLoss, OnlineTripletLoss, OnlineHardNegativeMiningTripletLoss
from losses import InnerProduct, FindFprTrainingSet, FPRLoss, PairwiseLoss, HardTrainingLoss

from read_matlab_imdb import read_matlab_imdb
from losses import Compute_FPR_HardNegatives, ComputeFPR

import warnings

warnings.filterwarnings("ignore", message="UserWarning: albumentations.augmentations.transforms.RandomResizedCrop")

from multiprocessing import Process, freeze_support

if __name__ == '__main__':
    freeze_support()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NumGpus = torch.cuda.device_count()
    torch.cuda.empty_cache()
    GPUtil.showUtilization()

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    name = torch.cuda.get_device_name(0)
    pool = Pool(processes=1)

    ModelsDirName = './models1/'
    LogsDirName = './logs1/'
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
    grad_accumulation_steps = 1


    if False:
        # GeneratorMode = 'PairwiseRot'
        GeneratorMode = 'Pairwise'
        CnnMode = 'PairwiseSymmetric'
        #criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='Random')
        criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='Hardest')
        #criterion         = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='MostHardest', HardRatio=1.0/8)
        Description = 'PairwiseSymmetric Hardest'

        InitializeOptimizer = True
        LearningRate = 1e-4
        OuterBatchSize = 24
        InnerBatchSize = 6
        Augmentation["Test"] = {'Do': False}
        Augmentation["HorizontalFlip"] = True
        Augmentation["VerticalFlip"] = True
        Augmentation["Rotate90"] = True
        Augmentation["RandomCrop"] = {'Do': True, 'MinDx': 0, 'MaxDx': 0.2, 'MinDy': 0, 'MaxDy': 0.2}
        grad_accumulation_steps = 1

        FreezeSymmetricCnn = False
        FreezeAsymmetricCnn = True

        FprHardNegatives = False

        StartBestModel = True

    if True:
        GeneratorMode = 'Pairwise'
        CnnMode = 'PairwiseAsymmetric'
        # criterion     = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='Random')
        criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='Hardest')
        # criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='HardPos', HardRatio=1.0/2, PosRatio=1. / 2)

        InitializeOptimizer = True
        StartBestModel      = True

        FreezeSymmetricCnn  = True
        FreezeAsymmetricCnn = False

        LearningRate = 1e-4
        OuterBatchSize = 24;
        InnerBatchSize = 6

        FprHardNegatives = True

        weight_decay = 1e-5

        Augmentation["Test"] = {'Do': False}
        Augmentation["HorizontalFlip"] = True
        Augmentation["VerticalFlip"] = True
        Augmentation["Rotate90"] = True
        Augmentation["RandomCrop"] = {'Do': True, 'MinDx': 0, 'MaxDx': 0.2, 'MinDy': 0, 'MaxDy': 0.2}

        #AssymetricInitializationPhase = True
        Description = 'PairwiseAsymmetric'

        StartBestModel = True

    if False:
        # GeneratorMode      = 'PairwiseRot'
        GeneratorMode = 'Pairwise'
        # CnnMode            = 'HybridRot'
        CnnMode = 'Hybrid'
        Random           = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='Random')
        Hardest          = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='Hardest')
        criterion        = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='HardPos', HardRatio=1.0/4, PosRatio=1./4)
        #HardestCriterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='Hardest')
        PairwiseLoss      = PairwiseLoss()
        InitializeOptimizer = True
        OuterBatchSize = 16  # 24
        InnerBatchSize = 2 * 12  # 24
        LearningRate = 1e-4
        DropoutP = 0.0
        weight_decay=1e-4

        FprHardNegatives = True
        StartBestModel = True

        FreezeSymmetricCnn  = True
        FreezeAsymmetricCnn = False

        AssymetricInitializationPhase = False

        Augmentation["Test"] = {'Do': False}
        Augmentation["HorizontalFlip"] = True
        Augmentation["VerticalFlip"] = True
        Augmentation["Rotate90"] = True
        Augmentation["RandomCrop"] = {'Do': True, 'MinDx': 0, 'MaxDx': 0.2, 'MinDy': 0, 'MaxDy': 0.2}
        grad_accumulation_steps = 1

        MaxNoImages = 400
        TestDecimation = 10


    ContinueMode = True


    # ----------------------------- read data----------------------------------------------
    Data = read_matlab_imdb(TrainFile)
    TrainingSetData = Data['Data']
    TrainingSetLabels = np.squeeze(Data['Labels'])
    TrainingSetSet = np.squeeze(Data['Set'])

    del Data

    # ShowTwoImages(TrainingSetData[0:3, :, :, 0])
    # ShowTwoRowImages(TrainingSetData[0:3, :, :, 0], TrainingSetData[0:3, :, :, 1])

    TrainIdx = np.squeeze(np.asarray(np.where(TrainingSetSet == 1)))
    ValIdx = np.squeeze(np.asarray(np.where(TrainingSetSet == 3)))

    # get val data
    ValSetLabels = torch.from_numpy(TrainingSetLabels[ValIdx])
    ValSetData = TrainingSetData[ValIdx, :, :, :].astype(np.float32)

    # ShowTwoRowImages(np.squeeze(ValSetData[0:4, :, :, :, 0]), np.squeeze(ValSetData[0:4, :, :, :, 1]))
    # ValSetData = torch.from_numpy(ValSetData).float().cpu()
    ValSetData[:, :, :, :, 0] -= ValSetData[:, :, :, :, 0].mean()
    ValSetData[:, :, :, :, 1] -= ValSetData[:, :, :, :, 1].mean()
    ValSetData = torch.from_numpy(NormalizeImages(ValSetData));

    # train data
    TrainingSetData = np.squeeze(TrainingSetData[TrainIdx,])
    TrainingSetLabels = TrainingSetLabels[TrainIdx]

    # define generators
    my_training_Dataset = DatasetPairwiseTriplets(TrainingSetData, TrainingSetLabels, InnerBatchSize, Augmentation,
                                                  GeneratorMode)
    my_training_DataLoader = data.DataLoader(my_training_Dataset, batch_size=OuterBatchSize, shuffle=True,
                                             num_workers=8)

    # Load all datasets
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
    net = MetricLearningCnn(CnnMode)
    optimizer = torch.optim.Adam(net.parameters(), lr=LearningRate)

    StartEpoch = 0
    if ContinueMode:
        # & os.path.isfile(ModelName):

        if StartBestModel:
            FileList = glob.glob(ModelsDirName + "visnir_best.pth")
        else:
            FileList = glob.glob(ModelsDirName + "visnir*")

        if FileList:
            FileList.sort(key=os.path.getmtime)

            print(FileList[-1] + ' loded')

            checkpoint = torch.load(FileList[-1])
            net.load_state_dict(checkpoint['state_dict'],strict=True)
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                aa=5

            StartEpoch = checkpoint['epoch'] + 1

            if 'FPR95' in checkpoint:
                FPR95 = checkpoint['FPR95']
                print('Loaded FPR95 = ' + repr(FPR95)[0:4])

        FileList = glob.glob(ModelsDirName + "visnir_best.pth")
        if FileList:
            checkpoint = torch.load(FileList[-1])
            LowestError = checkpoint['LowestError']
            print('LowestError: ' + repr(LowestError))
            #LowestError = 1e10
        else:
            LowestError = 1e10

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device)

    # -------------------- Initialization -----------------------
    if AssymetricInitializationPhase:
        net.module.netAS1 = copy.deepcopy(net.module.netS)
        net.module.netAS2 = copy.deepcopy(net.module.netS)

    if InitializeOptimizer:

        #optimizer = torch.optim.Adam(net.parameters(), lr=LearningRate,weight_decay=weight_decay)
        #optimizer = torch.optim.SGD(net.parameters(), lr=LearningRate,weight_decay=weight_decay)


        if False:
            optimizer = torch.optim.Adam(
                [{'params': AsymmCnnParams, 'weight_decay': 1e-5}, {'params': HeadCnnParams, 'weight_decay': 0},
                 {'params': SymmCnnParams, 'weight_decay': 0}],
                lr=LearningRate)

        if False:
            optimizer = optim.SGD([{'params': net.module.netS.parameters(), 'lr': 1e-5},
                        {'params': net.module.netAS1.parameters(), 'lr': 1e-5},
                        {'params': net.module.netAS2.parameters(), 'lr': 1e-5},
                        {'params': net.module.fc1.parameters(), 'lr': 1e-4},
                        {'params': net.module.fc2.parameters(), 'lr': 1e-4},
                        {'params': net.module.fc3.parameters(), 'lr': 1e-4}],
                                      lr=LearningRate, momentum=0.0, weight_decay=0.00)#momentum=0.9

        optimizer = optim.SGD([{'params': filter(lambda p: p.requires_grad == True, net.parameters()), 'weight_decay': 1e-5},
                               {'params': filter(lambda p: p.requires_grad == False, net.parameters()), 'weight_decay': 0}],
                              lr=LearningRate, momentum=0.0, weight_decay=0.00)  # momentum=0.9

    # ------------------------------------------------------------------------------------------



    # -------------------------------------  freeze layers --------------------------------------
    net.module.FreezeSymmetricCnn(FreezeSymmetricCnn)
    net.module.FreezeAsymmetricCnn(FreezeAsymmetricCnn)
    # ------------------------------------------------------------------------------------------

    ########################################################################
    # Train the network
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    InnerProductLoss = InnerProduct()
    CeLoss = nn.CrossEntropyLoss()






    # writer.add_graph(net, images)
    for epoch in range(StartEpoch, 80):  # loop over the dataset multiple times

        running_loss = 0
        running_loss_ce = 0
        running_loss_pos = 0
        running_loss_neg = 0
        optimizer.zero_grad()
        for i, Data in enumerate(tqdm(my_training_DataLoader, 0)):

            net = net.train()

            # get the inputs
            pos1 = Data['pos1']
            pos2 = Data['pos2']

            pos1 = np.reshape(pos1, (pos1.shape[0] * pos1.shape[1], 1, pos1.shape[2], pos1.shape[3]), order='F')
            pos2 = np.reshape(pos2, (pos2.shape[0] * pos2.shape[1], 1, pos2.shape[2], pos2.shape[3]), order='F')


            if GeneratorMode == 'Pairwise':

                if (CnnMode == 'PairwiseAsymmetric') | (CnnMode == 'PairwiseSymmetric'):

                    if FprHardNegatives:
                        Embed = Compute_FPR_HardNegatives(net, pos1, pos2, device, FprValPos=0.7*FPR95,
                                                               FprValNeg=1.5 * FPR95, MaxNoImages=MaxNoImages)
                        pos1 = Embed['PosIdx1']
                        pos2 = Embed['PosIdx2']

                    pos1, pos2 = pos1.to(device), pos2.to(device)
                    Embed = net(pos1, pos2)
                    loss = criterion(Embed['Emb1'], Embed['Emb2']) + criterion(Embed['Emb2'], Embed['Emb1'])


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


                            #loss  = HardTrainingLoss(net, pos1, pos2,PosRatio=0.25,HardRatio=0.25,T=1,device=device)
                            #loss += HardTrainingLoss(net, pos2, pos1, PosRatio=0.25, HardRatio=0.25, T=1, device=device)

                            Embed = net(pos1, pos2,p=DropoutP)
                            loss  = criterion(Embed['Hybrid1'], Embed['Hybrid2']) + criterion(Embed['Hybrid2'],Embed['Hybrid1'])
                            #loss += Random(Embed['Hybrid1'], Embed['Hybrid2'])    + Random(Embed['Hybrid2'],Embed['Hybrid1'])
                            #loss += Hardest(Embed['Hybrid1'], Embed['Hybrid2']) + Hardest(Embed['Hybrid2'],Embed['Hybrid1'])
                            #loss  = HardTrainingLoss(net, pos1, pos2, PosRatio=1, HardRatio=1.0/2, T=1, device=device)
                            #loss += HardTrainingLoss(net, pos2, pos1, PosRatio=1, HardRatio=1.0/4, T=1, device=device)

                            #loss += InnerProductLoss(Embed['EmbAsym1'], Embed['EmbSym1']) + InnerProductLoss(Embed['EmbAsym2'],Embed['EmbSym2'])
                            #loss +=criterion(Embed['EmbSym1'], Embed['EmbSym2']) + criterion(Embed['EmbSym2'],Embed['EmbSym1'])
                            #loss +=criterion(Embed['EmbAsym1'], Embed['EmbAsym2']) + criterion(Embed['EmbAsym2'],Embed['EmbAsym1'])

                            #TrainFpr = ComputeFPR(Embed['Hybrid1'], Embed['Hybrid2'], FPR95 * 0.9, FPR95 * 1.1)
                            # print('TrainFpr = ' + repr(TrainFpr))

            loss /= grad_accumulation_steps

            # backward + optimize
            loss.backward()

            if ((i + 1) % grad_accumulation_steps) == 0:  # Wait for several backward steps
                optimizer.step()  # Now we can do an optimizer step

                # zero the parameter gradients
                optimizer.zero_grad()

            running_loss     += loss.item()

            SchedularUpadteInterval = 200
            if (i % SchedularUpadteInterval == 0) &(i>0):  #
                running_loss /= SchedularUpadteInterval/grad_accumulation_steps
                print('running_loss: '+repr(100*running_loss)[0:4])
                scheduler.step(running_loss)







            PrintStep = 500
            if (i % PrintStep == 0):  # & (i>0):

                if i > 0:
                    running_loss_ce /= PrintStep
                    running_loss_neg /= PrintStep
                    running_loss_pos /= PrintStep

                    #print('running_loss_neg: ' + repr(100*running_loss_neg)[0:5] + ' running_loss_pos: ' + repr(100*running_loss_pos)[0:5])

                # val accuracy
                net.eval()
                StepSize = 800
                EmbVal1 = EvaluateNet(net.module.GetChannelCnn(0, CnnMode), ValSetData[:, :, :, :, 0], device,
                                      StepSize,CnnMode)
                EmbVal2 = EvaluateNet(net.module.GetChannelCnn(1, CnnMode), ValSetData[:, :, :, :, 1], device,
                                      StepSize.CnnMode)
                Dist = np.power(EmbVal1 - EmbVal2, 2).sum(1)
                ValError = FPR95Accuracy(Dist, ValSetLabels) * 100

                del EmbVal1, EmbVal2

                # estimate fpr95 threshold
                PosValIdx = np.squeeze(np.asarray(np.where(ValSetLabels == 1)))
                CurrentFPR95 = np.sort(Dist[PosValIdx])[int(0.95 * PosValIdx.shape[0])]
                if i > 0:
                    print('FPR95: ' + repr(CurrentFPR95)[0:4] + ' Loss= ' + repr(running_loss / i)[0:4])
                net.train()

                if (net.module.Mode == 'Hybrid1') | (net.module.Mode == 'Hybrid2'):
                    net.module.Mode = 'Hybrid'

                net = net.eval()

                if (i % 2000 == 0) & (i > 0):
                    #FPR95 = CurrentFPR95
                    print('FPR95 changed: ' + repr(FPR95)[0:5])

                # compute stats
                if (GeneratorMode == 'Pairwise') | (GeneratorMode == 'PairwiseRot'):

                    if i >= len(my_training_DataLoader):
                        TestDecimation1 = 1
                    else:
                        TestDecimation1 = TestDecimation;

                    # test accuracy
                    NoSamples = 0
                    TotalTestError = 0
                    for DataName in TestData:
                        EmbTest1 = EvaluateNet(net.module.GetChannelCnn(0, CnnMode),
                                               TestData[DataName]['Data'][0::TestDecimation1, :, :, :, 0], device,
                                               StepSize,CnnMode)
                        EmbTest2 = EvaluateNet(net.module.GetChannelCnn(1, CnnMode),
                                               TestData[DataName]['Data'][0::TestDecimation1, :, :, :, 1], device,
                                               StepSize,CnnMode)
                        Dist = np.power(EmbTest1 - EmbTest2, 2).sum(1)
                        TestData[DataName]['TestError'] = FPR95Accuracy(Dist, TestData[DataName]['Labels'][
                                                                              0::TestDecimation1]) * 100
                        TotalTestError += TestData[DataName]['TestError'] * TestData[DataName]['Data'].shape[0]
                        NoSamples += TestData[DataName]['Data'].shape[0]
                    TotalTestError /= NoSamples

                    del EmbTest1, EmbTest2

                    if (net.module.Mode == 'Hybrid1') | (net.module.Mode == 'Hybrid2'):
                        net.module.Mode = 'Hybrid'

                if TotalTestError < LowestError:
                    LowestError = TotalTestError

                    print('Best error found and saved: ' + repr(LowestError)[0:5])
                    filepath = ModelsDirName + BestFileName + '.pth'
                    state = {'epoch': epoch,
                             'state_dict': net.module.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'Description': Description,
                             'LowestError': LowestError,
                             'OuterBatchSize': OuterBatchSize,
                             'InnerBatchSize': InnerBatchSize,
                             'Mode': net.module.Mode,
                             'CnnMode': CnnMode,
                             'GeneratorMode': GeneratorMode,
                             'Loss': criterion.Mode,
                             'FPR95': FPR95}
                    torch.save(state, filepath)

                str = '[%d, %5d] loss: %.3f' % (epoch, i, 100 * running_loss) + ' Val Error: ' + repr(ValError)[0:6]
                if running_loss_ce > 0:
                    str += ' Rot loss: ' + repr(running_loss_ce)[0:6]

                # for DataName in TestData:
                #   str +=' ' + DataName + ': ' + repr(TestData[DataName]['TestError'])[0:6]
                str += ' FPR95 = ' + repr(FPR95)[0:6] + ' Mean: ' + repr(TotalTestError)[0:6]
                print(str)

                writer.add_scalar('Val Error', ValError, epoch * len(my_training_DataLoader) + i)
                writer.add_scalar('Test Error', TotalTestError, epoch * len(my_training_DataLoader) + i)
                writer.add_scalar('Loss', 100 * running_loss, epoch * len(my_training_DataLoader) + i)
                writer.add_scalar('FPR95', FPR95, epoch * len(my_training_DataLoader) + i)
                writer.add_text('Text', str)
                writer.close()




        # save epoch
        filepath = ModelsDirName + FileName + repr(epoch) + '.pth'
        state = {'epoch': epoch,
                 'state_dict': net.module.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'Description': Description,
                 'OuterBatchSize': OuterBatchSize,
                 'InnerBatchSize': InnerBatchSize,
                 'Mode': net.module.Mode,
                 'CnnMode': CnnMode,
                 'GeneratorMode': GeneratorMode,
                 'Loss': criterion.Mode,
                 'FPR95': FPR95}

        torch.save(state, filepath)

    print('Finished Training')