import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,show, legend
import numpy as np
import copy
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR
from Optimizers.RangerLars import RangerLars
from torch.utils import data
from torchvision.transforms import transforms
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from math import sqrt
import GPUtil
import pickle
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show

# my classes
from shaked.Models.UnifiedClassificaionAndRegressionAgeModel import UnifiedClassificaionAndRegressionAgeModel
from networks.face_classes import FaceGenerator, FaceClassification, Create_Training_Test_Sets, ComputeError, ShowRowImages, \
    imshow
from networks.face_classes import EvaluateNet, HardMining, ComputeErrorHistogram, sigmoid
from losses.losses import CascadedClassification, CenterLoss, MeanVarianceLoss, WeightedCrossEntropyLoss, MetricLearningLoss, \
    WeightedBinaryCrossEntropyLoss, AngleLoss,Heatmap1Dloss
from utils.utils import LoadModel,PrepareLoders
from utils.read_matlab_imdb import read_matlab_imdb

torch.autograd.set_detect_anomaly(True)
import warnings

warnings.filterwarnings("ignore", message="UserWarning: albumentations.augmentations.transforms.RandomResizedCrop")
from multiprocessing import Process, freeze_support

torch.backends.cudnn.deterministic = True  # needed
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    freeze_support()

    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NumGpus = 1#torch.cuda.device_count()
    torch.cuda.empty_cache()
    GPUtil.showUtilization()

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    name = torch.cuda.get_device_name(0)

    ModelsDirName = './models2/'
    LogsDirName = './logs2/'
    Description = 'DeepAge'
    BestFileName = 'face_best'
    FileName = 'visnir_sym_triplet'
    TestDir = 'F:\\multisensor\\test\\'
    TrainFile = 'f:\\deepage\\face_dataset.hdf5'

    writer = SummaryWriter(LogsDirName)
    LowestError = 1e10

    # ----------------------------     configuration   ---------------------------
    MseLoss = nn.MSELoss().to(device)
    CentLoss = CenterLoss(reduction = 'mean').to(device)
    MeanVareLoss = MeanVarianceLoss(LamdaMean=0.2, LamdaSTD=0.05,device=device).to(device)
    MetricLearnLoss = MetricLearningLoss(LabelsDistT=3,reduction = 'mean').to(device)
    CeLoss = nn.CrossEntropyLoss().to(device)
    AngularleLoss = AngleLoss().to(device)
    BCEloss = nn.BCEWithLogitsLoss().to(device)
    CELogProb = nn.NLLLoss().to(device)
    KLDiv = nn.KLDivLoss(reduction = 'batchmean').to(device)

    ShowFigure = False

    CnnMode = 'Shaked'
    CnnMode = 'Cascade'
    CnnMode = 'Classify'


    SamplingMode = 'UniformAge'
    SamplingMode = 'Random'

    # criterion=None
    criterion = nn.CrossEntropyLoss()



    InitializeOptimizer = True
    OuterBatchSize = 128#64*NumGpus
    InnerBatchSize = 1
    DuplicateNo = 1

    LearningRate = 1e-2
    MinAge = 15
    MaxAge = 80
    AgeIntareval = 1 #5
    NumLabels = (MaxAge - MinAge) / AgeIntareval + 1
    NoTestAugment = 1

    CascadeSupport = 15
    CascadeSkip  = 5
    FcK = 256
    UseCascadedLoss = {}
    UseCascadedLoss['Class']          = False
    UseCascadedLoss['MetricLearning'] = False
    UseCascadedLoss['Center']         = False
    UseCascadedLoss['Ordinal']        = False
    UseCascadedLoss['Sphere']         = False
    UseCascadedLoss['Regress']        = False
    UseCascadedLoss['MeanVar']        = False

    UseGradientNorm = True
    DropoutP = 0.5
    weight_decay = 1e-5

    StartBestModel = False
    UseBestScore   = True

    # FaceNet
    FreezeBaseCnn   = True
    FreezeFaceNetFC = True

    # age classifcation
    FreezeClassEmbeddFC        = False
    FreezeAgeClassificationFCs = False

    # emebedding layer
    FreezeEmbedding            = False

    # main ordinal
    FreezeOrdinalLayers       = False

    #heatmap
    FreezeHeatmapLayers       = True

    #age and gender
    FreezeEthnicityLayers = True
    FreezeGenderLayers    = True


    # losses
    UseAgeClassLoss     = True
    UseClassCenterLoss  = False
    UseMetricLearLoss   = False
    UseClassMeanValLoss = True
    UseAngularLoss      = False
    UseRegressionLoss   = False

    UseHeatmapLoss      = False

    # ordinal losses
    UseOridinalLoss         = False
    UseExtendedOridinalLoss = False



    #gender & ethnicity & faces
    UseClassifyFacesLoss = False
    UseGenderLoss        = False
    UseEthnicityLoss     = False
    UseAgeGenderRace     = False


    TrainingRatio = 0.8

    Heatmap10 = Heatmap1Dloss(device, NumLabes=NumLabels, sigma=10.0)
    WeightedBceLoss = WeightedBinaryCrossEntropyLoss(NumLabels,device,reduction='mean')

    if (CnnMode == 'Regress') | (CnnMode == 'Cascade'):
        criterion = CascadedClassification(NumLabels, CascadeSupport,CascadeSkip)


    # imgplot = plt.imshow(np.moveaxis(ROI, 0, 2))
    # ShowTwoRowImages(np.moveaxis(Images[0:2,], 1, 3), np.moveaxis(Images[2:4,], 1, 3))

    ContinueMode = True

    #np.random.seed(0)
    SaveTrainState = False;

    # ----------------------------- read data----------------------------------------------

    num_workers = 4
    Test_DataLoader, Train_DataLoader, Data = \
        PrepareLoders(TrainFile, TrainingRatio, SaveTrainState, MinAge, AgeIntareval, InnerBatchSize, OuterBatchSize,
                      DuplicateNo,
                      SamplingMode, num_workers, TrainTransformMode="TrainA", TestTransformMode="Test")

    # -------------------------    loading previous results   ------------------------
    net = FaceClassification(NumClasses=int(NumLabels), CascadeSupport=CascadeSupport,CascadeSkip=CascadeSkip,
                                 NumFaceClasses=Data['LinearIds'].max() + 1,K=FcK,DropoutP=DropoutP)

    #net = UnifiedClassificaionAndRegressionAgeModel(int(NumLabels), AgeIntareval, MinAge, MaxAge)

    StartEpoch = 0
    if ContinueMode:
        net, optimizer, LowestError, StartEpoch,scheduler = LoadModel(net, StartBestModel, ModelsDirName, BestFileName,
                                                            UseBestScore,LoadModel)

    net.to(device)
    if InitializeOptimizer:

        if True:
            FaceNetNames = ['FaceNet.' + x[0] for x in net.FaceNet.named_parameters()]  ## 'FaceNet.conv2d_1a.conv.weight'
            BaseParams = []
            for name, param in net.named_parameters():
                if name not in FaceNetNames:
                    BaseParams.append(param)


            optimizer = torch.optim.Adam([
                {'params': net.FaceNet.parameters(), 'lr': LearningRate, 'weight_decay': weight_decay},
                {'params': BaseParams, 'lr': LearningRate, 'weight_decay': weight_decay}
            ], lr=LearningRate)

            #optimizer = torch.optim.SGD([
            #    {'params': net.FaceNet.parameters(), 'lr': LearningRate / 10, 'weight_decay': weight_decay},
            #    {'params': BaseParams, 'lr': LearningRate, 'weight_decay': weight_decay}
            #], lr=LearningRate)


        #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad == True, net.parameters()), lr=LearningRate,weight_decay=weight_decay)
        #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad == True, net.parameters()), lr=LearningRate,weight_decay=weight_decay,momentum=0.9)
        optimizer = RangerLars(net.parameters(), lr=LearningRate)

    # ------------------------------------------------------------------------------------------






    # -------------------------------------  freeze layers --------------------------------------
    # classification
    if True:
        net.FreezeBaseCnn(FreezeBaseCnn)
        net.FreezeFaceNetFC(FreezeFaceNetFC)

        net.FreezeAgeClassificationFCs(FreezeAgeClassificationFCs)
        net.FreezeClassEmbeddFC(FreezeClassEmbeddFC)

        net.FreezeEmbedding(FreezeEmbedding)

        # ordinal
        net.FreezeOrdinalLayers(FreezeOrdinalLayers)


        #heatmap
        net.FreezeHeatmapLayers(FreezeHeatmapLayers)


        # Gender
        net.FreezeGenderLayers(FreezeGenderLayers)

        # Ethnicirt
        net.FreezeEthnicityLayers(FreezeEthnicityLayers)



    if NumGpus > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    ########################################################################
    # Train the network
    #scheduler  = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=0, verbose=True)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    TempotalError = np.array([])

    # writer.add_graph(net, images)
    for epoch in range(StartEpoch, 200):  # loop over the dataset multiple times

        print('\n\nStarting epoch: ' + repr(epoch))

        #if epoch == 0:# Warmpup
         #   scheduler = StepLR(optimizer, step_size=1, gamma=10)
        #if epoch==10:
         #   scheduler = StepLR(optimizer, step_size=5, gamma=sqrt(0.1))

        if NumGpus == 1:
            if epoch < 15:  net.FreezeBaseCnn(True)  # net.freeze_base_cnn(True)
            if epoch > 15:  net.FreezeBaseCnn(False)
            print('net.FaceNet.last_linear.weight.requires_grad: '+ repr(net.FaceNet.last_linear.weight.requires_grad))
        else:
            if epoch < 15:  net.module.FreezeBaseCnn(True)
            if epoch > 15:  net.module.FreezeBaseCnn(False)
            print('net.FaceNet.last_linear.weight.requires_grad: ' + repr(net.module.FaceNet.last_linear.weight.requires_grad))

        if epoch > StartEpoch:
            #scheduler.step(running_loss)
            scheduler.step()
            str = 'running_loss=' + repr(running_loss)[0:4]
        else:
            str = ''

        # Print Learning Rates

        str += ' LR: '
        for param_group in optimizer.param_groups:
            str += repr(param_group['lr']) + ' '
        print(str+'\n')

        running_loss = 0
        running_regression = 0
        optimizer.zero_grad()
        for i, TrainData in enumerate(tqdm(Train_DataLoader, 0)):

            net = net.train()

            # get the inputs
            Labels = TrainData['Labels']
            CurrentImages = TrainData['Images']
            Age = TrainData['Ages']
            #FaceIds = TrainData['Ids']
            #Ethnicity1 = TrainData['Race']
            #Gender1 = TrainData['Gender']

            Age = np.reshape(Age, (Age.shape[0] * Age.shape[1]), order='F')
            #Ethnicity1 = np.reshape(Ethnicity1, (Ethnicity1.shape[0] * Ethnicity1.shape[1]), order='F')
            #Gender1 = np.reshape(Gender1, (Gender1.shape[0] * Gender1.shape[1]), order='F')
            #FaceIds = np.reshape(FaceIds, (FaceIds.shape[0] * FaceIds.shape[1]), order='F')
            Labels = np.reshape(Labels, (Labels.shape[0] * Labels.shape[1]), order='F')
            CurrentImages = np.reshape(CurrentImages,
                                 (CurrentImages.shape[0] * CurrentImages.shape[1], CurrentImages.shape[2], CurrentImages.shape[3],
                                  CurrentImages.shape[4]),
                                 order='F')

            Labels, CurrentImages, Age= Labels.to(device), CurrentImages.to(device), Age.to(device)
            #Ethnicity1, Gender1, FaceIds  = Ethnicity1.to(device), Gender1.to(device), FaceIds.to(device)

            #Embed = net(CurrentImages, CnnMode, Labels=Age,DropoutP = 0.5)
            Embed = net(CurrentImages,Mode =CnnMode,Labels=Labels)
            loss = 0

            if (CnnMode == 'Classify') | (CnnMode == 'Shaked'):

                if UseClassifyFacesLoss:
                    # IdClassLoss = CeLoss(Embed['IdEmbed'], FaceIds )
                    IdClassLoss = AngularleLoss(Embed['IdEmbed'], Labels.round().long())
                    loss += IdClassLoss


                if UseAgeGenderRace:
                    AgeGenderRaceClassLoss = criterion(Embed['ClassAgeGenderRace'], torch.round(Labels).long(), )
                    loss += AgeGenderRaceClassLoss

                if UseMetricLearLoss:
                    MlLoss = MetricLearnLoss(Embed['Base'], Labels.round().long(), Mode='Hard')
                    #MlLoss = MetricLearnLoss(Embed['Base'], torch.round(Labels).long(), Mode='Random')
                    loss +=  MlLoss


                if UseAngularLoss:
                    AngLoss = AngularleLoss(Embed['Sphere'], Labels.round().long())
                    loss += AngLoss


                if UseGenderLoss:
                    GenderLoss = BCEloss(Embed['Gender'].squeeze(), Gender1.float())
                    loss += GenderLoss


                if UseEthnicityLoss:
                    EthnicityLoss = CeLoss(Embed['Ethnicity'].squeeze(), Ethnicity1)
                    loss += EthnicityLoss

                if UseRegressionLoss:
                    RegressionLoss = ((Embed['Regression'] - Age) ** 2).mean()
                    loss += RegressionLoss

                    running_regression += (Embed['Regression'] - Age).abs().mean().item()


            if (CnnMode == 'Cascade') | (CnnMode == 'Regress'):

                if NumGpus == 1:
                    loss1,AllLosses = criterion(CnnMode, Embed, Labels.round().long(), Age.long(),
                                     UseLoss=UseCascadedLoss,
                                     EmbeddingCenters=net.CascadeEmbedding,
                                     ApplyAngularLoss=net.CascadeAngularLoss)
                else:
                    loss1, AllLosses = criterion(CnnMode, Embed, Labels.round().long(), Age.long(),
                                               UseLoss=UseCascadedLoss,
                                               EmbeddingCenters=net.module.CascadeEmbedding,
                                               ApplyAngularLoss=net.module.CascadeAngularLoss)
                #loss += loss1

            if UseCascadedLoss['Regress']:
                RegressionLoss = ((Embed['ProbRegress'] - Age) ** 2).mean()
                loss += RegressionLoss

                running_regression += (Embed['ProbRegress'] - Age).abs().mean().item()

            if UseOridinalLoss:
                OrdinalLoss         = WeightedBceLoss(Embed['OrdinalClass'], Labels.round().long(), ComputeWeights=True)
                loss += OrdinalLoss

            if UseExtendedOridinalLoss:
                CurrentLabels   = torch.clamp(Labels,0,NumLabels-1)
                ExtendedOrdinalLoss = CELogProb(Embed['ExtendedOrdinalClass'].log(),torch.round(CurrentLabels).long())
                #loss += ExtendedOrdinalLoss

                ClassProbLog = F.log_softmax(Embed['Class'], 1)
                KLDivLoss = KLDiv(ClassProbLog[:,:-1],Embed['ExtendedOrdinalClass'])+  KLDiv(  Embed['ExtendedOrdinalClass'].log(), F.softmax(Embed['Class'][:,:-1],1))
                #loss += KLDivLoss

                #H = -((Embed['ExtendedOrdinalClass']*Embed['ExtendedOrdinalClass'].log()).sum(1)).mean()
                #loss += H

                MVloss = MeanVareLoss(Embed['ExtendedOrdinalClass'],CurrentLabels.round().long(),IsProbability=True)
                loss += MVloss



            if UseAgeClassLoss:
                #AgeClassLoss = CeLoss(Embed['Class'], torch.round(Labels).long(), )
                AgeClassLoss = CeLoss(Embed['ClassA'], torch.round(Labels).long(), )
                loss += AgeClassLoss

            if UseClassMeanValLoss:
                MVloss = MeanVareLoss(Embed['Class'], Labels.round().long())
                loss += MVloss




            if UseHeatmapLoss:
                HeatmapLoss = Heatmap3(Embed['HeatmapClass'], Labels.round().long())
                loss += HeatmapLoss

                HeatmapCascadeClassLoss = Heatmap1(Embed['HeatmapCascadeClass'], Labels.round().long())
                loss += HeatmapCascadeClassLoss





            if UseClassCenterLoss:

                if NumGpus == 1:
                    CenterLoss = CentLoss(Embed['ClassEmbed'], net.Embedding, Labels.round().long(), Mode='MSE')
                else:
                    CenterLoss = CentLoss(Embed['ClassEmbed'], net.module.Embedding, Labels.round().long(), Mode='MSE')
                loss += CenterLoss



            # backward + optimize
            loss.backward()

            clipping_value = 1
            if UseGradientNorm:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_value)

            optimizer.step()  # Now we can do an optimizer step

            # zero the parameter gradients
            optimizer.zero_grad()

            running_loss += loss.item()







            SchedularUpadteInterval = 100
            if (i % SchedularUpadteInterval == 0) & (i > 0):

                #str = 'Loss: ' + repr(loss.item())[0:4]
                str = 'running_loss: ' + repr(running_loss/i)[0:4]

                # TRAINING LOSS
                if CnnMode == 'Cascade':
                    #CascadeOrdinal = (Embed['CascadeOrdinal'] - Labels).abs().mean()
                    #CascadeProbMean = (Embed['CascadeProb'] - Labels).abs().mean()
                    #str += ' CascadeOrdinal: ' + repr(CascadeOrdinal.item())[0:4] #+ ' CascadeProb: ' + repr(CascadeProbMean.item())[0:4]

                    if UseCascadedLoss['MetricLearning']: str += ' Cas MetricLearning: ' + repr(AllLosses['MlLoss'].item())[0:4]
                    if UseCascadedLoss['Center']:         str += ' Cas CenterLoss: '     + repr(AllLosses['Center'].item())[0:4]
                    if UseCascadedLoss['Sphere']:         str += ' Cas SphereLoss: '     + repr(AllLosses['Sphere'].item())[0:4]
                    if UseCascadedLoss['Ordinal']:        str += ' Cas OrdinalLoss: '    + repr(AllLosses['Ordinal'].item())[0:4]
                    if UseCascadedLoss['Class']:          str += ' Cas ClassLoss: '      + repr(AllLosses['CascadeClass'].item())[0:4]
                    if UseCascadedLoss['Regress']:        str += ' Cas Regression Error: ' + repr(running_regression/i)[0:4]

                if UseAgeClassLoss:      str += ' AgeClassLoss: ' + repr(AgeClassLoss.item())[0:4]
                if UseClassCenterLoss:   str += ' CenterLoss: ' + repr(CenterLoss.item())[0:4]
                if UseClassMeanValLoss:  str += ' MeanValLoss: ' + repr(MVloss)[0:4]
                if UseMetricLearLoss:    str += ' MetricLearLoss: ' + repr(MlLoss.item())[0:4]
                if UseClassifyFacesLoss: str += ' IdClassLoss: ' + repr(IdClassLoss.item())[0:4]
                if UseAngularLoss:       str += ' AngularLoss: ' + repr(AngLoss.item())[0:4]
                if UseRegressionLoss:
                    str += ' Regression Error: ' + repr(running_regression/i)[0:4]
                if UseGenderLoss:
                    Error = ((Embed['Gender'] > 0).squeeze().int() != Gender1).float().mean()
                    str += ' GenderLoss: ' + repr(GenderLoss.item())[0:4] + ' G. Error: ' + repr(Error.item())[0:4]
                if UseEthnicityLoss:
                    Error = (Embed['Ethnicity'].argmax(1) != Ethnicity1).float().mean()
                    str += ' GenderLoss: ' + repr(EthnicityLoss.item())[0:4] + ' Eth. Error: ' + repr(Error.item())[0:4]
                if UseOridinalLoss:
                    Error = ((Embed['OrdinalClass'] > 0).sum(1) - Labels).abs().mean()
                    str += ' OridinalLoss: ' + repr(OrdinalLoss.item())[0:4] + ' Ordinal Error: ' + repr(Error.item())[0:4]
                if UseExtendedOridinalLoss:
                    str += ' Extended Ordinal Error: ' + repr(ExtendedOrdinalLoss.item())[0:4]
                if UseHeatmapLoss:
                    Error = (Embed['HeatmapClass'].argmax(1) - Labels).abs().mean()
                    str += ' Heatmap Error: ' + repr(Error.item())[0:4]

                print(str)










            PrintStep = 800
            if ((i % PrintStep == 0) or (i == len(Train_DataLoader)-1)) and (i>0):

                if (i % PrintStep == 0): running_loss /= PrintStep
                else:                    running_loss /= i

                net = net.eval()

                str = '\n [%d, %5d] loss: %.3f' % (epoch, i, 100 * running_loss)

                if ShowFigure:
                    fig, ax = plt.subplots()

                with torch.set_grad_enabled(False):

                    # val accuracy

                    if CnnMode == 'Cascade': TestMode = 'Cascade_test'
                    else:                    TestMode = CnnMode



                    AugmentEmb = []
                    for i in range(NoTestAugment):
                        AugmentEmb.append(EvaluateNet(net,Test_DataLoader,device, Mode=TestMode))

                    Emb = dict()

                    if 'Class' in AugmentEmb[0].keys():
                        Emb['Class']       = AugmentEmb[0]['Class']
                    if 'ProbRegress' in AugmentEmb[0].keys():
                        Emb['ProbRegress'] = AugmentEmb[0]['ProbRegress']
                    if 'OrdinalClass' in AugmentEmb[0].keys():
                         Emb['OrdinalClass'] = AugmentEmb[0]['OrdinalClass']
                    for i in range(1,NoTestAugment):
                        if 'Class' in AugmentEmb[0].keys():
                            Emb['Class']       += AugmentEmb[i]['Class']
                        if 'ProbRegress' in AugmentEmb[0].keys():
                            Emb['ProbRegress'] += AugmentEmb[i]['ProbRegress']
                        if 'OrdinalClass' in AugmentEmb[0].keys():
                            Emb['OrdinalClass'] += AugmentEmb[i]['OrdinalClass']

                    if 'ProbRegress' in AugmentEmb[0].keys():
                        Emb['ProbRegress'] /= NoTestAugment
                    if 'OrdinalClass' in AugmentEmb[0].keys():
                        Emb['OrdinalClass'] /= NoTestAugment
                    Emb['Labels']       = AugmentEmb[0]['Labels']
                    Emb['Ages']         = AugmentEmb[0]['Ages']


                    ClassResult = Emb['Class'].argmax(1)
                    ClassError = np.abs(ClassResult - Emb['Labels'].squeeze())*AgeIntareval

                    TempotalError =  np.append(TempotalError, ClassError.mean())

                    class_count = np.bincount(ClassError.astype(np.long))

                    if ShowFigure:
                        ax.plot(class_count, 'b-*', label='Class');

                    str += ' Class Average: ' + repr(ClassError.mean())[0:4]

                    if UseAgeGenderRace:
                        ClassResult = Emb['ClassAgeGenderRace'].argmax(1)
                        Error = np.abs(ClassResult - np.round(TestLabels.numpy()))

                        str += ' ClassAgeGenderRace Average: ' + repr(Error.mean())[0:4]

                    if UseOridinalLoss:
                        OrdinalError = ((Emb['OrdinalClass'] > 0).sum(1) - np.round(Emb['Labels'].squeeze()))*AgeIntareval
                        print('Ordinal error mean: ' + repr(OrdinalError.mean())[0:4])

                        OrdinalError = np.abs(OrdinalError)

                        # plot(Probs[idx[0],:]);show()

                        class_count = np.bincount(OrdinalError.astype(np.long))

                        if ShowFigure:
                            ax.plot(class_count, 'g-s',label='Ordinal');


                        str += ' Ordinal AverageLoss: ' + repr(OrdinalError.mean())[0:4]

                    if UseExtendedOridinalLoss:
                        OrdinalProbs = sigmoid(Emb['OrdinalClass'])
                        BaseOrdinalClassificationProbs = -(OrdinalProbs[:, 1:] - OrdinalProbs[:, 0:-1])
                        # plt.plot(BaseOrdinalClassificationProbs[0, :].detach().cpu());show()
                        ExtendOrdinalClassIdx = BaseOrdinalClassificationProbs.argmax(1)

                        TestLabels = np.clip(np.round(Emb['Labels'].squeeze()), a_min = 0, a_max = NumLabels - 1)
                        ExtendedOrdinalError = np.abs(TestLabels-ExtendOrdinalClassIdx).mean()

                        str += ' ExtendedOridinal Average: ' + repr(ExtendedOrdinalError)[0:4]


                    if UseRegressionLoss:
                        RegressionLossError = np.abs((Emb['Regression'] - Emb['Ages'].squeeze())).mean()
                        str += ' Regression Error: ' + repr(RegressionLossError )[0:4]
                        CurrentError = RegressionLossError


                    if UseHeatmapLoss:
                        HeatmapResult = Emb['HeatmapClass'].argmax(1)
                        HeatmapError = (HeatmapResult - np.round(TestLabels.numpy()))*AgeIntareval
                        str += ' Heatmap L1: ' + repr(np.abs(HeatmapError).mean())[0:4] +' Heatmap mean: ' + repr(HeatmapError.mean())[0:4]

                        class_count = np.bincount(np.abs(HeatmapError).astype(np.long))

                        if ShowFigure:
                            ax.plot(class_count, 'r-+',label='Heatmap');


                        HeatmapResult = Emb['HeatmapCascadeClass'].argmax(1)
                        HeatmapError = (HeatmapResult - np.round(TestLabels.numpy())) * AgeIntareval
                        str += ' CascadesHeatmap L1: ' + repr(np.abs(HeatmapError).mean())[0:4] + ' CascadesHeatmap mean: ' + repr(HeatmapError.mean())[0:4]

                        class_count = np.bincount(np.abs(HeatmapError).astype(np.long))
                        if ShowFigure:
                            ax.plot(class_count, 'k-*', label='CasHeatmap');

                    if UseGenderLoss:
                        GenderError = ((Emb['Gender'] > 0).squeeze().astype(int) != Gender[Data['TestSamples']]).astype(float).mean()
                        print('GenderError: ' + repr(GenderError)[0:4])

                    if UseEthnicityLoss:
                        EthnicityError = (Emb['Ethnicity'].argmax(1) != Ethnicity[Data['TestSamples']]).astype(float).mean()
                        print('EthnicityError: ' + repr(EthnicityError)[0:4])

                    # compute and print errors

                    print(str)




                    if CnnMode == 'Cascade':

                        str = ''

                        #greedy
                        if False:
                            CasslError = np.abs(np.squeeze(Emb['CascadeGreedyClass']) - TestLabels.numpy())
                            str = ' Casscade greedy: ' + repr(CasslError.mean())[0:4]


                            #ordinal
                            CassOdinalError = (np.squeeze(Emb['CascadeOrdinal']) - TestLabels.numpy())[idx]
                            CasslErrorMean  = CassOdinalError.mean()
                            CasslError      = np.abs(CassOdinalError)
                            str += ' Casscade ordinal mean: ' + repr(CasslErrorMean.mean())[0:4]
                            str += ' Casscade ordinal: ' + repr(CasslError.mean())[0:4]

                            class_count = np.bincount((CasslError).astype(np.long))

                            if ShowFigure:
                                ax.plot(class_count, 'm-^', label='Cas oridinal');




                        if UseCascadedLoss['Regress']:
                            RegressionError = np.abs(np.round(Emb['ProbRegress']).squeeze() - Emb['Ages'].squeeze()).mean()

                            str += ' Cascaded Regress Error: '      + repr(RegressionError)[0:4]

                        print(str)

                    if ShowFigure:
                        ax.legend(frameon=False, fontsize='xx-large');show()

                    if UseAgeClassLoss:                           CurrentError = ClassError.mean()
                    if UseOridinalLoss | UseExtendedOridinalLoss: CurrentError = OrdinalError.mean()
                    if UseCascadedLoss['Regress']:                CurrentError = RegressionError.mean()
                    if UseCascadedLoss['Ordinal']:                CurrentError = RegressionError.mean()


                    if (i >= (len(Train_DataLoader) - PrintStep)) | (CurrentError < LowestError):# | i>500:

                        if CurrentError < LowestError:
                            LowestError = CurrentError
                            print('Best error found and saved: ' + repr(LowestError)[0:5])
                            filepath = ModelsDirName + BestFileName + '.pth'

                        if (i >= (len(Train_DataLoader) - PrintStep)) | True:
                            # end of epoch
                            filepath = ModelsDirName + FileName + repr(epoch) + '.pth'

                        state = {'epoch': epoch,
                                 'state_dict': net.state_dict() if (NumGpus == 1) else net.module.state_dict(),
                                 'optimizer_name': type(optimizer).__name__,
                                 'optimizer': optimizer.state_dict(),
                                 'scheduler_name': type(scheduler).__name__,
                                 'scheduler': scheduler.state_dict(),
                                 'Description': Description,
                                 'LowestError': LowestError,
                                 'OuterBatchSize': OuterBatchSize,
                                 'InnerBatchSize': InnerBatchSize,
                                 'DropoutP': DropoutP,
                                 'weight_decay': weight_decay,
                                 'CnnMode ': CnnMode}
                        torch.save(state, filepath)
                        print('Saved checkpoint ' + filepath)

                    x = (epoch * len(Train_DataLoader) + i) / PrintStep
                    writer.add_text('Text', str)
                    writer.close()

                    net = net.train()

        print('\n\n')

    print('Finished Training')