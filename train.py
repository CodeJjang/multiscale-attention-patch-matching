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
warnings.filterwarnings("ignore", message="UserWarning: albumentations.augmentations.transforms.RandomResizedCrop")

def assert_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def load_datasets_paths(ds_name):
    if ds_name == 'VisNir':
        test_dir = 'F:\\multisensor\\test\\'
        train_file = 'F:\\multisensor\\train\\Vis-Nir_Train.hdf5'
    elif ds_name == 'cuhk':
        test_dir = 'D:\\multisensor\\datasets\\cuhk\\test\\'
        train_file = 'D:\\multisensor\\datasets\\cuhk\\train.hdf5'
    elif ds_name == 'vedai':
        test_dir = 'D:\\multisensor\\datasets\\vedai\\test\\'
        train_file = 'D:\\multisensor\\datasets\\vedai\\train.hdf5'
    elif ds_name == 'visnir-grid':
        test_dir = 'D:\\multisensor\\datasets\\Vis-Nir_grid\\test\\'
        train_file = 'D:\\multisensor\\datasets\\Vis-Nir_grid\\train.hdf5'
    elif ds_name == 'brown-liberty':
        test_dir = 'D:\\multisensor\\datasets\\brown\\patchdata\\test_yos_not\\'
        train_file = 'D:\\multisensor\\datasets\\brown\\patchdata\\liberty_full_for_multisensor.hdf5'
    elif ds_name == 'brown-notredame':
        test_dir = 'D:\\multisensor\\datasets\\brown\\patchdata\\test_lib_yos\\'
        train_file = 'D:\\multisensor\\datasets\\brown\\patchdata\\notredame_full_for_multisensor.hdf5'
    elif ds_name == 'brown-yosemite':
        test_dir = 'D:\\multisensor\\datasets\\brown\\patchdata\\test_lib_not\\'
        train_file = 'D:\\multisensor\\datasets\\brown\\patchdata\\yosemite_full_for_multisensor.hdf5'
    elif ds_name == 'hpatches-liberty':
        test_dir = 'D:\\multisensor\\datasets\\hpatches-benchmark\\data\\hpatches-release-multisensor\\data_v2.hdf5'
        train_file = 'D:\\multisensor\\datasets\\brown\\patchdata\\liberty_full_for_multisensor.hdf5'
    elif ds_name == 'hpatches-all-brown':
        test_dir = 'D:\\multisensor\\datasets\\hpatches-benchmark\\data\\hpatches-release-multisensor\\data_v2.hdf5'
        train_file = 'D:\\multisensor\\datasets\\brown\\patchdata\\full_for_multisensor.hdf5'
    return train_file, test_dir


def load_test_datasets(TestDir):
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
    return TestData


def evaluate_test(TestData, TestDecimation1, CnnMode, device, StepSize):
    NoSamples = 0
    TotalTestError = 0
    for DataName in TestData:
        EmbTest = EvaluateDualNets(net, TestData[DataName]['Data'][0::TestDecimation1, :, :, :, 0],
                                   TestData[DataName]['Data'][0::TestDecimation1, :, :, :, 1], CnnMode, device,
                                   StepSize)

        Dist = np.power(EmbTest['Emb1'] - EmbTest['Emb2'], 2).sum(1)
        TestData[DataName]['TestError'] = FPR95Accuracy(Dist, TestData[DataName]['Labels'][
                                                              0::TestDecimation1]) * 100
        TotalTestError += TestData[DataName]['TestError'] * TestData[DataName]['Data'].shape[0]
        NoSamples += TestData[DataName]['Data'].shape[0]
    TotalTestError /= NoSamples

    del EmbTest
    return TotalTestError


def evaluate_hpatch(TestData, TestDecimation1, CnnMode, device, StepSize, splits, taskdir, extra_data):
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
    verification_map, verification_data = results_verification(CnnMode, splits['full'], eval)
    print('Verification mAP:', verification_map * 100)
    eval = eval_matching(descriptors, splits['full'])
    matching_map, matching_data = results_matching(CnnMode, splits['full'], eval)
    print('Matching mAP:', matching_map * 100)
    eval = eval_retrieval(descriptors, splits['full'], taskdir)
    retrieval_map, retrieval_data = results_retrieval(CnnMode, splits['full'], eval)
    print('Retrieval mAP:', retrieval_map * 100)
    extra_data['verification'] = {'mAP': verification_map, 'data': verification_data}
    extra_data['matching'] = {'mAP': matching_map, 'data': matching_data}
    extra_data['retrieval'] = {'mAP': retrieval_map, 'data': retrieval_data}
    return 1 - verification_map, 1 - matching_map, 1 - retrieval_map


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#"cuda:0"
    NumGpus = torch.cuda.device_count()
    torch.cuda.empty_cache()
    GPUtil.showUtilization()
    print(device)
    name = torch.cuda.get_device_name(0)

    ModelsDirName = './artifacts/symmetric_enc_transformer_test_2/models/'
    LogsDirName = './artifacts/symmetric_enc_transformer_test/logs/'
    Description = 'Symmetric CNN with Triplet loss, no HM'
    BestFileName = 'best_model'
    FileName = 'model_epoch_'
    ds_name = 'VisNir'
    TrainFile, TestDir = load_datasets_paths(ds_name)
    TestDecimation = 1
    FPR95 = 0.8

    assert_dir(ModelsDirName)
    assert_dir(LogsDirName)

    scaler = MyGradScaler()

    writer = SummaryWriter(LogsDirName)
    LowestError = 1e10

    # ----------------------------     configuration   ---------------------------
    Augmentation = {}

    assymetric_init = False

    TestMode = False
    use_validation = False
    FreezeSymmetricBlock = False

    torch.manual_seed(0)
    np.random.seed(0)
    #torch.set_deterministic(True)


    GeneratorMode = 'Pairwise'
    CnnMode = 'SymmetricAttention'
    NegativeMiningMode = 'Random'
    criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode=NegativeMiningMode,device=device)
    Description = 'Symmetric Hardest'

    InitializeOptimizer = True
    UseWarmUp           = True

    StartBestModel      = False
    UseBestScore        = False

    LearningRate = 1e-1

    weight_decay = 0
    DropoutP = 0.5

    OuterBatchSize = 4*12
    InnerBatchSize = 2*12
    Augmentation["Test"] = {'Do': False}
    Augmentation["HorizontalFlip"] = True
    Augmentation["Rotate90"] = True
    Augmentation["VerticalFlip"] = False
    Augmentation["HorizontalFlip"] = True
    Augmentation["Test"] = False

    PrintStep = 100
    FreezeSymmetricCnn  = False
    FreezeSymmetricBlock = False

    FreezeAsymmetricCnn = True




    StartBestModel = False
    UseBestScore   = False







    # ----------------------------- read data----------------------------------------------
    Data = read_matlab_imdb(TrainFile)
    TrainingSetData = Data['Data']
    TrainingSetLabels = np.squeeze(Data['Labels'])
    TrainingSetSet = np.squeeze(Data['Set'])
    del Data


    ValSetData = []
    TrainIdx = np.squeeze(np.asarray(np.where(TrainingSetSet == 1)))
    if use_validation:
        ValIdx = np.squeeze(np.asarray(np.where(TrainingSetSet == 3)))

        # VALIDATION data
        ValSetLabels = torch.from_numpy(TrainingSetLabels[ValIdx])

        ValSetData = TrainingSetData[ValIdx, :, :, :].astype(np.float32)
        ValSetData[:, :, :, :, 0] -= ValSetData[:, :, :, :, 0].mean()
        ValSetData[:, :, :, :, 1] -= ValSetData[:, :, :, :, 1].mean()
        ValSetData = torch.from_numpy(NormalizeImages(ValSetData))

    # TRAINING data
    TrainingSetData = np.squeeze(TrainingSetData[TrainIdx,])
    TrainingSetData = np.squeeze(TrainingSetData[TrainIdx,])
    TrainingSetLabels = TrainingSetLabels[TrainIdx]

    # define generators
    Training_Dataset = DatasetPairwiseTriplets(TrainingSetData, TrainingSetLabels, InnerBatchSize, Augmentation, GeneratorMode)
    Training_DataLoader = MultiEpochsDataLoader(Training_Dataset, batch_size=OuterBatchSize, shuffle=True,
                                                num_workers=8, pin_memory=True)


    TestData = load_test_datasets(TestDir)

    net = MetricLearningCnn(CnnMode,DropoutP)
    optimizer = torch.optim.Adam(net.parameters(), lr=LearningRate)


    StartEpoch = 0

    net,optimizer,LowestError,StartEpoch,scheduler,LodedNegativeMiningMode = LoadModel(net, StartBestModel, ModelsDirName, BestFileName, UseBestScore, device)
    print('LodaedNegativeMiningMode: ' + LodedNegativeMiningMode)

    # -------------------------------------  freeze layers --------------------------------------
    # net.FreezeSymmetricCnn(FreezeSymmetricCnn)
    # net.FreezeSymmetricBlock(FreezeSymmetricBlock)
    #
    # net.FreezeAsymmetricCnn(FreezeAsymmetricCnn)
    # ------------------------------------------------------------------------------------------

    # -------------------- Initialization -----------------------
    if assymetric_init:
        net.netAS1 = copy.deepcopy(net.module.netS)
        net.netAS2 = copy.deepcopy(net.module.netS)


    if NumGpus > 1:
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
        # WarmUpEpochs = 4
        WarmUpEpochs = 8
        # WarmUpEpochs = 4
        scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=1, total_epoch=WarmUpEpochs,
                                                    after_scheduler= StepLR(optimizer, step_size=3, gamma=0.1))
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

        #print('\n' + colored('Gain = ' + repr(net.module.Gain.item())[0:6], 'cyan', attrs=['reverse', 'blink']))
        #print('\n' + colored('Gain1 = ' +repr(net.module.Gain1.item())[0:6], 'cyan', attrs=['reverse', 'blink']))
        #print('\n' + colored('Gain2 = ' +repr(net.module.Gain2.item())[0:6], 'cyan', attrs=['reverse', 'blink']))


        #warmup
        if InitializeOptimizer and (epoch - StartEpoch < WarmUpEpochs) and UseWarmUp:
            print(colored('\n Warmup step #' + repr(epoch - StartEpoch), 'green', attrs=['reverse', 'blink']))
            #print('\n Warmup step #' + repr(epoch - StartEpoch))
            scheduler_warmup.step()
        else:
            if epoch > StartEpoch:
                print('CurrentError=' + repr(ValError)[0:8])

                if type(scheduler).__name__ == 'StepLR':
                    scheduler.step()

                if type(scheduler).__name__ == 'ReduceLROnPlateau':
                    # scheduler.step(ValError)
                    scheduler.step(TotalTestError)
        running_loss = 0
        #scheduler_warmup.step(epoch-StartEpoch,running_loss)

        str = '\n LR: '
        for param_group in optimizer.param_groups:
            str += repr(param_group['lr']) + ' '
        print(colored(str, 'blue', attrs=['reverse', 'blink']))

        print('FreezeSymmetricCnn = ' + repr(FreezeSymmetricCnn) + '\nFreezeAsymmetricCnn = '+repr(FreezeAsymmetricCnn) + '\n')
        print('NegativeMiningMode = ' + criterion.Mode)
        print('CnnMode = '+CnnMode + '\nGeneratorMode = ' + GeneratorMode)

        Case1 = (criterion.Mode == 'Random') and (optimizer.param_groups[0]['lr'] <= (LearningRate/1e3 + 1e-8)) \
                and (epoch-StartEpoch>WarmUpEpochs)
        Case2 = (CnnMode == 'Hybrid') and (criterion.Mode == 'Hardest') and (optimizer.param_groups[0]['lr'] <= (LearningRate/1e3 +1e-8)) \
                and (FreezeSymmetricCnn==True)
        if Case1 or Case2:
            if Case1:
                #print('Switching Random->Hardest')
                print(colored('Switching Random->Hardest', 'green', attrs=['reverse', 'blink']))
                criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode = 'Hardest',device=device)

                LearningRate = 1e-1
                optimizer = torch.optim.Adam(
                    [{'params': filter(lambda p: p.requires_grad == True, net.parameters()), 'lr': LearningRate,
                      'weight_decay': weight_decay},
                     {'params': filter(lambda p: p.requires_grad == False, net.parameters()), 'lr': 0,
                      'weight_decay': 0}],
                    lr=0, weight_decay=0.00)

                #start with warmup
                scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=1, total_epoch=WarmUpEpochs)
                StartEpoch = epoch

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

            # zero the parameter gradients
            optimizer.zero_grad()

            net = net.train()

            # get the inputs
            pos1 = Data['pos1']
            pos2 = Data['pos2']

            pos1 = np.reshape(pos1, (pos1.shape[0] * pos1.shape[1], 1, pos1.shape[2], pos1.shape[3]), order='F')
            pos2 = np.reshape(pos2, (pos2.shape[0] * pos2.shape[1], 1, pos2.shape[2], pos2.shape[3]), order='F')


            pos1, pos2 = pos1.to(device), pos2.to(device)

            Embed = net(pos1, pos2,DropoutP=DropoutP)

            loss = criterion(Embed['Emb1'], Embed['Emb2']) + criterion(Embed['Emb2'], Embed['Emb1'])


            scaler.scale(loss).backward()
            clipping_value = 1
            scaler.step(optimizer)
            scaler.update()

            running_loss     += loss.item()

            SchedularUpadteInterval = 200
            if (i % SchedularUpadteInterval == 0) &(i>0):
                print('running_loss: '+repr(running_loss/i)[0:8])


            if (((i % PrintStep == 0) or (i * InnerBatchSize >= len(Training_DataLoader) - 1)) and (i > 0)) or TestMode:

                if i > 0:
                    running_loss     /=i
                    running_loss_neg /= i
                    running_loss_pos /= i


                # val accuracy
                StepSize = 800
                net.eval()

                if len(ValSetData) > 0:
                    Emb = EvaluateDualNets(net, ValSetData[:, :, :, :, 0], ValSetData[:, :, :, :, 1],CnnMode,device, StepSize)

                    Dist = np.power(Emb['Emb1'] - Emb['Emb2'], 2).sum(1)
                    ValError = FPR95Accuracy(Dist, ValSetLabels) * 100
                    del Emb
                    # estimate fpr95 threshold
                    PosValIdx = np.squeeze(np.asarray(np.where(ValSetLabels == 1)))
                    CurrentFPR95 = np.sort(Dist[PosValIdx])[int(0.95 * PosValIdx.shape[0])]
                    if i > 0:
                        print('FPR95: ' + format(CurrentFPR95, ".2e") + ' Loss= ' + repr(running_loss)[0:6])
                else:
                    ValError = 0



                print('FPR95 changed: ' + repr(FPR95)[0:5])

                # compute stats


                if i >= len(Training_DataLoader):
                    TestDecimation1 = 1
                else:
                    TestDecimation1 = TestDecimation

                # test accuracy
                extra_data = {}
                if 'hpatch' not in ds_name:
                    TotalTestError = evaluate_test(TestData, TestDecimation1, CnnMode, device, StepSize)
                else:
                    verification_err, matching_err, retrieval_err = evaluate_hpatch(TestData, TestDecimation1, CnnMode, device, StepSize, hpatch_splits, hpatch_taskdir, extra_data)
                    TotalTestError = verification_err

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

                #if (ValError < LowestError):
                if TotalTestError < LowestError:
                    LowestError = TotalTestError
                    # LowestError = ValError

                    print(colored('Best error found and saved: ' + repr(TotalTestError)[0:5], 'red', attrs=['reverse', 'blink']))
                    #print('Best error found and saved: ' + repr(LowestError)[0:5])
                    filepath = ModelsDirName + BestFileName + '.pth'
                    torch.save(state, filepath)
                    save_best_model_stats(ModelsDirName, epoch, TotalTestError, TestData, extra_data)


                str = '[%d, %5d] loss: %.3f' % (epoch, i, 100 * running_loss) + ' Val Error: ' + repr(ValError)[0:6]


                # for DataName in TestData:
                #   str +=' ' + DataName + ': ' + repr(TestData[DataName]['TestError'])[0:6]
                str += ' FPR95 = ' + repr(FPR95)[0:6] + ' Mean: ' + repr(TotalTestError)[0:6]
                print(str)

                if True:
                    writer.add_scalar('Val Error', ValError, epoch * len(Training_DataLoader) + i)
                    if 'hpatch' not in ds_name:
                        writer.add_scalar('Test Error', TotalTestError, epoch * len(Training_DataLoader) + i)
                    else:
                        writer.add_scalar('Test Verification Error', verification_err, epoch * len(Training_DataLoader) + i)
                        writer.add_scalar('Test Matching Error', matching_err,
                                          epoch * len(Training_DataLoader) + i)
                        writer.add_scalar('Test Retrieval Error', retrieval_err,
                                          epoch * len(Training_DataLoader) + i)
                    writer.add_scalar('Loss', 100 * running_loss, epoch * len(Training_DataLoader) + i)
                    writer.add_scalar('FPR95', FPR95, epoch * len(Training_DataLoader) + i)
                    writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch * len(Training_DataLoader) + i)
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