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
from my_classes import DatasetPairwiseTriplets, FPR95Accuracy1, FPR95Accuracy2
from my_classes import SingleNet, MetricLearningCnn, EvaluateNet, SiamesePairwiseSoftmax, NormalizeImages
from losses import ContrastiveLoss, TripletLoss, OnlineTripletLoss, OnlineHardNegativeMiningTripletLoss
from losses import InnerProduct, FindHardTrainingSet, FindFprTrainingSet, FPRLoss, PairwiseLoss, \
    FindFprTrainingSingleSideSet
from read_matlab_imdb import read_matlab_imdb

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

    ModelsDirName = './models2/'
    LogsDirName = './logs2/'
    Description = 'Symmetric CNN with Triplet loss, no HM'
    BestFileName = 'visnir_best'
    FileName = 'visnir_sym_triplet'
    # TestDir = '/home/keller/Dropbox/multisensor/python/data/test/'
    TestDir = 'F:\\multisensor\\test\\'
    TestDir = 'data//Vis-Nir_grid//test//'
    # TrainFile = '/home/keller/Dropbox/multisensor/python/data/Vis-Nir_Train.mat'
    TrainFile = 'f:\\multisensor\\train\\Vis-Nir_Train.hdf5'
    TrainFile = './data/Vis-Nir_grid/Vis-Nir_grid_Train.hdf5'
    TestDecimation = 10
    FPR95 = 0.8
    MaxNoImages = 600

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
    FreezeBaseCnns = False
    FreezeSymmetricCnn = False
    FreezeAsymmetricCnn = False

    AssymetricInitializationPhase = False
    grad_accumulation_steps = 1

    if False:
        # GeneratorMode = 'PairwiseRot'
        GeneratorMode = 'Pairwise'
        CnnMode = 'PairwiseSymmetric'
        # criterion           = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='Random')
        criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='Hardest')
        # criterion         = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='MostHardest', HardRatio=1.0/8)
        Description = 'PairwiseSymmetric Hardest'

        InitializeOptimizer = True
        FreezeBaseCnns = False
        LearningRate = 1e-1
        OuterBatchSize = 24
        InnerBatchSize = 36
        Augmentation["Test"] = {'Do': True}
        Augmentation["HorizontalFlip"] = True
        Augmentation["VerticalFlip"] = True
        Augmentation["Rotate90"] = True
        Augmentation["RandomCrop"] = {'Do': True, 'MinDx': 0, 'MaxDx': 0.2, 'MinDy': 0, 'MaxDy': 0.2}
        grad_accumulation_steps = 1

    if True:
        GeneratorMode = 'Pairwise'
        CnnMode = 'PairwiseAsymmetric'
#        criterion     = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='Random')
        criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='Hardest')
        InitializeOptimizer = False
        FreezeBaseCnns = False
        FreezeSymmetricCnn = False
        LearningRate = 1e-4
        OuterBatchSize = 24;
        InnerBatchSize = 32
        Augmentation["RandomCrop"] = {'Do': True, 'MinDx': 0, 'MaxDx': 0.3, 'MinDy': 0, 'MaxDy': 0.3}
        AssymetricInitializationPhase = False
        Description = 'PairwiseAsymmetric'

    if False:
        # GeneratorMode      = 'PairwiseRot'
        GeneratorMode = 'Pairwise'
        # CnnMode            = 'HybridRot'
        CnnMode = 'Hybrid'
        # criterion           = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='Random')
        # criterion          = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='MostHardest', HardRatio=1.0/8)
        # criterion           = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='Hardest')
        # criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='HardPos', HardRatio=1.0/2,PosRatio=1./2)
        criterion = PairwiseLoss()
        InitializeOptimizer = True
        OuterBatchSize = 16  # 24
        InnerBatchSize = 30 * 12  # 24
        LearningRate = 1e-1

        FreezeSymmetricCnn = False
        FreezeBaseCnns = False
        FreezeAsymmetricCnn = False

        AssymetricInitializationPhase = False

        Augmentation["Test"] = {'Do': False}
        Augmentation["HorizontalFlip"] = True
        Augmentation["VerticalFlip"] = True
        Augmentation["Rotate90"] = True
        Augmentation["RandomCrop"] = {'Do': True, 'MinDx': 0, 'MaxDx': 0.2, 'MinDy': 0, 'MaxDy': 0.2}
        grad_accumulation_steps = 1

        MaxNoImages = 600

    ContinueMode = True
    StartBestModel = False

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
        DatasetName = os.path.splitext(DatasetName)[0][0:-5]

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
            net.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            StartEpoch = checkpoint['epoch'] + 1

            if 'FPR95' in checkpoint:
                FPR95 = checkpoint['FPR95']

        FileList = glob.glob(ModelsDirName + "visnir_best.pth")
        if FileList:
            checkpoint = torch.load(FileList[-1])
            LowestError = checkpoint['LowestError']
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
        optimizer = torch.optim.Adam(net.parameters(), lr=LearningRate)
        # optimizer = optim.SGD(net.parameters(), lr=LearningRate, momentum=0.9)

    # ------------------------------------------------------------------------------------------

    # -------------------------------------  freeze layers --------------------------------------
    net.module.FreezeBackboneCnns(FreezeBaseCnns)
    net.module.FreezeSymmetricCnn(FreezeSymmetricCnn)
    net.module.FreezeAsymmetricCnn(FreezeAsymmetricCnn)
    # ------------------------------------------------------------------------------------------

    ########################################################################
    # Train the network
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    InnerProductLoss = InnerProduct()
    CeLoss = nn.CrossEntropyLoss()

    # writer.add_graph(net, images)
    for epoch in range(StartEpoch, 80):  # loop over the dataset multiple times

        running_loss = 0
        running_loss_ce = 0
        optimizer.zero_grad()
        for i, Data in enumerate(tqdm(my_training_DataLoader, 0)):

            net = net.train()

            # get the inputs
            pos1 = Data['pos1']
            pos2 = Data['pos2']

            pos1 = np.reshape(pos1, (pos1.shape[0] * pos1.shape[1], 1, pos1.shape[2], pos1.shape[3]), order='F')
            pos2 = np.reshape(pos2, (pos2.shape[0] * pos2.shape[1], 1, pos2.shape[2], pos2.shape[3]), order='F')

            if (GeneratorMode == 'PairwiseRot'):
                RotImg1 = Data['RotPos1']
                RotImg2 = Data['RotPos2']

                Labels1 = Data['RotLabel1']
                Labels2 = Data['RotLabel2']

                RotImg1 = np.reshape(RotImg1,
                                     (RotImg1.shape[0] * RotImg1.shape[1], 1, RotImg1.shape[2], RotImg1.shape[3]),
                                     order='F')
                RotImg2 = np.reshape(RotImg2,
                                     (RotImg2.shape[0] * RotImg2.shape[1], 1, RotImg2.shape[2], RotImg2.shape[3]),
                                     order='F')
                Labels1 = np.reshape(Labels1, (Labels1.shape[0] * Labels1.shape[1]), order='F')
                Labels2 = np.reshape(Labels2, (Labels2.shape[0] * Labels2.shape[1]), order='F')

                RotImg1, RotImg2, Labels1, Labels2 = RotImg1.to(device), RotImg2.to(device), Labels1.to(
                    device), Labels2.to(device)

                Embed = net(pos1, pos2, 'PairwiseSymmetricRot', RotImg1, RotImg2)

                # loss = criterion(Embed['Emb1'], Embed['Emb2']) + criterion(Embed['Emb2'], Embed['Emb1'])

                Class1to2 = net(Embed['RotUnnormalized1'], Embed['Unnormalized2'], 'SM')
                Class2to1 = net(Embed['RotUnnormalized2'], Embed['Unnormalized1'], 'SM')

                CE_Loss = CeLoss(Class1to2, Labels1) + CeLoss(Class2to1, Labels2)
                loss = CE_Loss
                running_loss_ce += CE_Loss.item()

            if GeneratorMode == 'Pairwise':

                if (CnnMode == 'PairwiseAsymmetric') | (CnnMode == 'PairwiseSymmetric'):
                    pos1, pos2 = pos1.to(device), pos2.to(device)
                    Embed = net(pos1, pos2)
                    loss = criterion(Embed['Emb1'], Embed['Emb2']) + criterion(Embed['Emb2'], Embed['Emb1'])

                if CnnMode == 'Hybrid':

                    # GPUtil.showUtilization()
                    with torch.no_grad():
                        Embed1, Embed2 = EvaluateDualNets(net, pos1, pos2, device, StepSize=400)

                    # GPUtil.showUtilization()

                    try:
                        with torch.no_grad():
                            # Idx = FindHardTrainingSet(Embed1, Embed2, HardRatio=1.0, PosRatio=1.0/10)
                            Idx = FindFprTrainingSet(Embed1, Embed2, FprValPos=FPR95 * 0.9, FprValNeg=FPR95 * 1.1)
                        del Embed1, Embed2

                        print('PosIdx: ' + repr(Idx['PosIdx'].shape[0]) + ' NegIdxA1: ' + repr(
                            Idx['NegIdxA1'].shape[0]) + ' NegIdxB1: ' + repr(Idx['NegIdxB1'].shape[0]))

                        while (Idx['PosIdx'].shape[0] + Idx['NegIdxA1'].shape[0] + Idx['NegIdxB1'].shape[
                            0]) > MaxNoImages:
                            # print('Memory error #1. #Images: ' + repr(Idx['PosIdx'].shape[0] + Idx['NegIdxA1'].shape[0] + Idx['NegIdxB1'].shape[0]))

                            Idx['PosIdx'] = Idx['PosIdx'][0:int(Idx['PosIdx'].shape[0] / 2)]

                            Idx['NegIdxA1'] = Idx['NegIdxA1'][0:int(Idx['NegIdxA1'].shape[0] / 2)]
                            Idx['NegIdxA2'] = Idx['NegIdxA2'][0:int(Idx['NegIdxA2'].shape[0] / 2)]

                            Idx['NegIdxB1'] = Idx['NegIdxB1'][0:int(Idx['NegIdxB1'].shape[0] / 2)]
                            Idx['NegIdxB2'] = Idx['NegIdxB2'][0:int(Idx['NegIdxB2'].shape[0] / 2)]

                        pos1 = pos1[np.concatenate((Idx['PosIdx'], Idx['NegIdxA1'], Idx['NegIdxB1']), 0),]
                        pos2 = pos2[np.concatenate((Idx['PosIdx'], Idx['NegIdxA2'], Idx['NegIdxB2']), 0),]
                    except:
                        test = 0

                    try:
                        Embed = net(pos1, pos2)
                    except:
                        print('Erro: No images:' + repr(pos1.shape[0]))
                        test = 0
                        continue

                    PosIdxLength = Idx['PosIdx'].shape[0]
                    NegIdxA1Length = Idx['NegIdxA1'].shape[0]

                    EmbedPos1 = Embed['Hybrid1'][:PosIdxLength, ]
                    EmbedPos2 = Embed['Hybrid2'][:PosIdxLength, ]

                    EmbedNegA1 = Embed['Hybrid1'][PosIdxLength:(PosIdxLength + NegIdxA1Length), ]
                    EmbedNegA2 = Embed['Hybrid2'][PosIdxLength:(PosIdxLength + NegIdxA1Length), ]

                    EmbedNegB1 = Embed['Hybrid1'][(PosIdxLength + NegIdxA1Length):, ]
                    EmbedNegB2 = Embed['Hybrid2'][(PosIdxLength + NegIdxA1Length):, ]

                    loss = criterion(EmbedPos1, EmbedPos2) - criterion(EmbedNegA1, EmbedNegA2) - criterion(EmbedNegB1,
                                                                                                           EmbedNegB2)

                    # loss += InnerProductLoss(Embed['EmbAsym1'], Embed['EmbSym1']) + InnerProductLoss(Embed['EmbAsym2'],Embed['EmbSym2'])
                    # criterion(Embed['EmbSym1'], Embed['EmbSym2']) + criterion(Embed['EmbSym2'],Embed['EmbSym1']) + \
                    # criterion(Embed['EmbAsym1'], Embed['EmbAsym2']) + criterion(Embed['EmbAsym2'],Embed['EmbAsym1'])

            loss /= grad_accumulation_steps

            # backward + optimize
            try:
                loss.backward()
            except:
                print('Memory error #2. #Images: ' + repr(pos1.shape[0]))
                continue

            if ((i + 1) % grad_accumulation_steps) == 0:  # Wait for several backward steps
                try:
                    optimizer.step()  # Now we can do an optimizer step
                except:
                    print('Memory error #3. #Images: ' + repr(pos1.shape[0]))
                # zero the parameter gradients
                optimizer.zero_grad()

            running_loss += loss.item()

            PrintStep = 500
            if (i % PrintStep == 0):  # & (i>0):

                if i > 0:
                    running_loss /= PrintStep / grad_accumulation_steps
                    running_loss_ce /= PrintStep
                    scheduler.step(running_loss)

                # val accuracy
                net.eval()
                StepSize = 512
                EmbVal1 = EvaluateNet(net.module.GetChannelCnn(0, CnnMode), ValSetData[:, :, :, :, 0], device,
                                      StepSize)
                EmbVal2 = EvaluateNet(net.module.GetChannelCnn(1, CnnMode), ValSetData[:, :, :, :, 1], device,
                                      StepSize)
                Dist = np.power(EmbVal1 - EmbVal2, 2).sum(1)
                ValError = FPR95Accuracy1(Dist, ValSetLabels) * 100

                del EmbVal1, EmbVal2

                # estimate fpr95 threshold
                PosValIdx = np.squeeze(np.asarray(np.where(ValSetLabels == 1)))
                CurrentFPR95 = np.sort(Dist[PosValIdx])[int(0.95 * PosValIdx.shape[0])]
                if i > 0:
                    print('FPR95: ' + repr(CurrentFPR95) + ' Loss= ' + repr(running_loss / i))
                net.train()

                if (net.module.Mode == 'Hybrid1') | (net.module.Mode == 'Hybrid2'):
                    net.module.Mode = 'Hybrid'

                net = net.eval()

                FPR95 = CurrentFPR95
                print('FPR95 changed: ' + repr(FPR95)[0:4])

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
                                               StepSize)
                        EmbTest2 = EvaluateNet(net.module.GetChannelCnn(1, CnnMode),
                                               TestData[DataName]['Data'][0::TestDecimation1, :, :, :, 1], device,
                                               StepSize)
                        Dist = np.power(EmbTest1 - EmbTest2, 2).sum(1)
                        TestData[DataName]['TestError'] = FPR95Accuracy1(Dist, TestData[DataName]['Labels'][
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