import argparse
import glob
import os
import warnings
from pathlib import Path

import GPUtil
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from termcolor import colored
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from tqdm import tqdm

from datasets.DatasetPairwiseTriplets import DatasetPairwiseTriplets
from networks.losses import OnlineHardNegativeMiningTripletLoss
from networks.PatchMatchingNetwork import PatchMatchingNetwork
from util.read_hdf5_data import read_hdf5_data
from util.utils import load_model, MultiEpochsDataLoader, MyGradScaler, save_best_model_stats, FPR95Accuracy, \
    normalize_image, evaluate_network
from util.warmup_scheduler import GradualWarmupSchedulerV2

warnings.filterwarnings("ignore", message="UserWarning: albumentations.augmentations.transforms.RandomResizedCrop")


def assert_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def load_datasets_paths(ds_name, ds_path):
    if ds_name == 'visnir':
        test_dir = os.path.join(ds_path, 'test\\')
        train_file = os.path.join(ds_path, 'train\\train.hdf5')
    elif ds_name == 'cuhk':
        test_dir = os.path.join(ds_path, 'en_etal\\cuhk\\test\\')
        train_file = os.path.join(ds_path, 'en_etal\\cuhk\\train.hdf5')
    elif ds_name == 'vedai':
        test_dir = os.path.join(ds_path, 'en_etal\\vedai\\test\\')
        train_file = os.path.join(ds_path, 'en_etal\\vedai\\train.hdf5')
    elif ds_name == 'visnir-grid':
        test_dir = os.path.join(ds_path, 'en_etal\\visnir\\test\\')
        train_file = os.path.join(ds_path, 'en_etal\\visnir\\train.hdf5')
    elif ds_name == 'ubc-liberty':
        test_dir = os.path.join(ds_path, 'ubc\\test_yos_not\\')
        train_file = os.path.join(ds_path, 'ubc\\liberty_full.hdf5')
    elif ds_name == 'ubc-notredame':
        test_dir = os.path.join(ds_path, 'ubc\\test_lib_yos\\')
        train_file = os.path.join(ds_path, 'ubc\\notredame_full.hdf5')
    elif ds_name == 'ubc-yosemite':
        test_dir = os.path.join(ds_path, 'ubc\\test_lib_not\\')
        train_file = os.path.join(ds_path, 'ubc\\yosemite_full.hdf5')
    return train_file, test_dir


def load_test_datasets(TestDir):
    FileList = glob.glob(TestDir + "*.hdf5")
    TestData = dict()
    for File in FileList:
        path, DatasetName = os.path.split(File)
        DatasetName = os.path.splitext(DatasetName)[0]

        Data = read_hdf5_data(File)

        x = Data['Data'].astype(np.float32)
        TestLabels = torch.from_numpy(np.squeeze(Data['Labels']))
        del Data

        x[:, :, :, :, 0] -= x[:, :, :, :, 0].mean()
        x[:, :, :, :, 1] -= x[:, :, :, :, 1].mean()

        x = normalize_image(x)
        x = torch.from_numpy(x)

        TestData[DatasetName] = dict()
        TestData[DatasetName]['Data'] = x
        TestData[DatasetName]['Labels'] = TestLabels
        del x
    return TestData


def evaluate_test(TestData, device, StepSize):
    NoSamples = 0
    TotalTestError = 0
    for DataName in TestData:
        EmbTest = evaluate_network(net, TestData[DataName]['Data'][:, :, :, :, 0],
                                   TestData[DataName]['Data'][:, :, :, :, 1], device,
                                   StepSize)

        Dist = np.power(EmbTest['Emb1'] - EmbTest['Emb2'], 2).sum(1)
        TestData[DataName]['TestError'] = FPR95Accuracy(Dist, TestData[DataName]['Labels'][:]) * 100
        TotalTestError += TestData[DataName]['TestError'] * TestData[DataName]['Data'].shape[0]
        NoSamples += TestData[DataName]['Data'].shape[0]
    TotalTestError /= NoSamples

    del EmbTest
    return TotalTestError


def parse_args():
    parser = argparse.ArgumentParser(description='Train models for multimodal patch matching.')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--artifacts', default='./artifacts', help='artifacts path')
    parser.add_argument('--exp-name', default='symmetric_enc_transformer_test_4', help='experiment name')
    parser.add_argument('--evaluate-every', type=int, default=50, help='evaluate network and print steps')
    parser.add_argument('--skip-validation', type=bool, const=True, default=False,
                        help='whether to skip validation evaluation', nargs='?')
    parser.add_argument('--continue-from-checkpoint', type=bool, const=True, default=False,
                        nargs='?', help='whether to continue training from checkpoint')
    parser.add_argument('--continue-from-best-score', type=bool, const=True, default=False,
                        nargs='?', help='whether to use best score when continuing training')
    parser.add_argument('--continue-from-best-model', type=bool, const=True, default=True,
                        nargs='?', help='whether to continue training using best model')
    parser.add_argument('--batch-size', type=int, default=40, help='batch size')
    parser.add_argument('--inner-batch-size', type=int, default=24, help='inner batch size of positive pairs')
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--weight-decay', type=float, default=0, help='weight decay')
    parser.add_argument('--dataset-name', default='visnir', help='dataset name')
    parser.add_argument('--dataset-path', default='visnir', help='dataset name')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NumGpus = torch.cuda.device_count()
    torch.cuda.empty_cache()
    GPUtil.showUtilization()
    print('Using', device)
    name = torch.cuda.get_device_name(0)

    ModelsDirName = os.path.join(args.artifacts, args.exp_name, 'models')
    LogsDirName = os.path.join(args.artifacts, args.exp_name, 'logs')
    Description = 'Symmetric CNN with Triplet loss and transformer encoder'
    BestFileName = 'best_model'
    FileName = 'model_epoch_'
    TrainFile, TestDir = load_datasets_paths(args.dataset_name, args.dataset_path)

    assert_dir(ModelsDirName)
    assert_dir(LogsDirName)

    scaler = MyGradScaler()

    writer = SummaryWriter(LogsDirName)

    # ----------------------------     configuration   ---------------------------
    Augmentation = {}

    TestMode = False
    skip_validation = args.skip_validation

    GeneratorMode = 'Pairwise'
    NegativeMiningMode = 'Random'
    criterion = OnlineHardNegativeMiningTripletLoss(margin=1, mode=NegativeMiningMode, device=device)

    InitializeOptimizer = True
    UseWarmUp = True

    LearningRate = args.lr

    weight_decay = args.weight_decay
    dropout = args.dropout

    OuterBatchSize = args.batch_size
    InnerBatchSize = args.inner_batch_size
    Augmentation["Test"] = {'Do': False}
    Augmentation["HorizontalFlip"] = True
    Augmentation["Rotate90"] = True
    Augmentation["VerticalFlip"] = False
    Augmentation["RandomCrop"] = {'Do': False}
    Augmentation["Test"] = False

    PrintStep = args.evaluate_every

    StartBestModel = args.continue_from_best_model
    UseBestScore = args.continue_from_best_score

    # ----------------------------- read data----------------------------------------------
    Data = read_hdf5_data(TrainFile)
    TrainingSetData = Data['Data']
    TrainingSetLabels = np.squeeze(Data['Labels'])
    TrainingSetSet = np.squeeze(Data['Set'])
    del Data

    ValSetData = []
    TrainIdx = np.squeeze(np.asarray(np.where(TrainingSetSet == 1)))
    if not skip_validation:
        ValIdx = np.squeeze(np.asarray(np.where(TrainingSetSet == 3)))

        # VALIDATION data
        ValSetLabels = torch.from_numpy(TrainingSetLabels[ValIdx])

        ValSetData = TrainingSetData[ValIdx, :, :, :].astype(np.float32)
        ValSetData[:, :, :, :, 0] -= ValSetData[:, :, :, :, 0].mean()
        ValSetData[:, :, :, :, 1] -= ValSetData[:, :, :, :, 1].mean()
        ValSetData = torch.from_numpy(normalize_image(ValSetData))

    # TRAINING data
    TrainingSetData = np.squeeze(TrainingSetData[TrainIdx,])
    TrainingSetData = np.squeeze(TrainingSetData[TrainIdx,])
    TrainingSetLabels = TrainingSetLabels[TrainIdx]

    # define generators
    Training_Dataset = DatasetPairwiseTriplets(TrainingSetData, TrainingSetLabels, InnerBatchSize, Augmentation,
                                               GeneratorMode)
    Training_DataLoader = MultiEpochsDataLoader(Training_Dataset, batch_size=OuterBatchSize, shuffle=True,
                                                num_workers=8, pin_memory=True)

    TestData = load_test_datasets(TestDir)

    net = PatchMatchingNetwork()
    optimizer = torch.optim.Adam(net.parameters(), lr=LearningRate)

    StartEpoch = 0
    LowestError = 1e5
    if args.continue_from_checkpoint:
        net, optimizer, LowestError, StartEpoch, scheduler, loaded_negative_mining_mode = load_model(net,
                                                                                                     StartBestModel,
                                                                                                     ModelsDirName,
                                                                                                     BestFileName,
                                                                                                     UseBestScore,
                                                                                                     device)

    if NumGpus > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device)

    if InitializeOptimizer:
        optimizer = torch.optim.Adam(
            [{'params': filter(lambda p: p.requires_grad == True, net.parameters()), 'lr': LearningRate,
              'weight_decay': weight_decay},
             {'params': filter(lambda p: p.requires_grad == False, net.parameters()), 'lr': 0, 'weight_decay': 0}],
            lr=0, weight_decay=0.00)

    ########################################################################
    # Train the network
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    if UseWarmUp:
        WarmUpEpochs = 8
        scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=1, total_epoch=WarmUpEpochs,
                                                    after_scheduler=StepLR(optimizer, step_size=3, gamma=0.1))
    else:
        WarmUpEpochs = 0

    CeLoss = nn.CrossEntropyLoss()

    for epoch in range(StartEpoch, 1000):  # loop over the dataset multiple times

        running_loss_pos = 0
        running_loss_neg = 0
        optimizer.zero_grad()

        # warmup
        if InitializeOptimizer and (epoch - StartEpoch < WarmUpEpochs) and UseWarmUp:
            print(colored('\n Warmup step #' + repr(epoch - StartEpoch), 'green', attrs=['reverse', 'blink']))
            # print('\n Warmup step #' + repr(epoch - StartEpoch))
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
        # scheduler_warmup.step(epoch-StartEpoch,running_loss)

        str = '\n LR: '
        for param_group in optimizer.param_groups:
            str += repr(param_group['lr']) + ' '
        print(colored(str, 'blue', attrs=['reverse', 'blink']))

        print('NegativeMiningMode = ' + criterion.mode)
        print('GeneratorMode = ' + GeneratorMode)

        finished_warmup = (criterion.mode == 'Random') and (
                    optimizer.param_groups[0]['lr'] <= (LearningRate / 1e3 + 1e-8)) \
                          and (epoch - StartEpoch > WarmUpEpochs)
        if finished_warmup:
            print(colored('Switching Random->Hardest', 'green', attrs=['reverse', 'blink']))
            criterion = OnlineHardNegativeMiningTripletLoss(margin=1, mode='Hardest', device=device)

            LearningRate = 1e-1
            optimizer = torch.optim.Adam(
                [{'params': filter(lambda p: p.requires_grad == True, net.parameters()), 'lr': LearningRate,
                  'weight_decay': weight_decay},
                 {'params': filter(lambda p: p.requires_grad == False, net.parameters()), 'lr': 0,
                  'weight_decay': 0}],
                lr=0, weight_decay=0.00)

            # start with warmup
            scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=1, total_epoch=WarmUpEpochs)
            StartEpoch = epoch

            if type(scheduler).__name__ == 'StepLR':
                scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

            if type(scheduler).__name__ == 'ReduceLROnPlateau':
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

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

            Embed = net(pos1, pos2, dropout=dropout)

            loss = criterion(Embed['Emb1'], Embed['Emb2']) + criterion(Embed['Emb2'], Embed['Emb1'])

            scaler.scale(loss).backward()
            clipping_value = 1
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            SchedularUpadteInterval = 200
            if (i % SchedularUpadteInterval == 0) & (i > 0):
                print('running_loss: ' + repr(running_loss / i)[0:8])

            if (((i % PrintStep == 0) or (i * InnerBatchSize >= len(Training_DataLoader) - 1)) and (i > 0)) or TestMode:

                if i > 0:
                    running_loss /= i
                    running_loss_neg /= i
                    running_loss_pos /= i

                # val accuracy
                StepSize = 800
                net.eval()

                if len(ValSetData) > 0:
                    Emb = evaluate_network(net, ValSetData[:, :, :, :, 0], ValSetData[:, :, :, :, 1], device, StepSize)

                    Dist = np.power(Emb['Emb1'] - Emb['Emb2'], 2).sum(1)
                    ValError = FPR95Accuracy(Dist, ValSetLabels) * 100
                    del Emb
                else:
                    ValError = 0

                # test accuracy
                TotalTestError = evaluate_test(TestData, device, StepSize)

                state = {'epoch': epoch,
                         'state_dict': net.module.state_dict(),
                         'optimizer_name': type(optimizer).__name__,
                         # 'optimizer': optimizer.state_dict(),
                         'optimizer': optimizer,
                         'scheduler_name': type(scheduler).__name__,
                         # 'scheduler': scheduler.state_dict(),
                         'scheduler': scheduler,
                         'Description': Description,
                         'LowestError': LowestError,
                         'OuterBatchSize': OuterBatchSize,
                         'InnerBatchSize': InnerBatchSize,
                         'NegativeMiningMode': criterion.mode,
                         'GeneratorMode': GeneratorMode,
                         'Loss': criterion.mode}

                if TotalTestError < LowestError:
                    LowestError = TotalTestError

                    print(colored('Best error found and saved: ' + repr(TotalTestError)[0:5], 'red',
                                  attrs=['reverse', 'blink']))
                    filepath = os.path.join(ModelsDirName, BestFileName + '.pth')
                    torch.save(state, filepath)
                    save_best_model_stats(ModelsDirName, epoch, TotalTestError, TestData)

                str = '[%d, %5d] loss: %.3f' % (epoch, i, 100 * running_loss) + ' Val Error: ' + repr(ValError)[0:6]

                str += 'Mean: ' + repr(TotalTestError)[0:6]
                print(str)

                if True:
                    writer.add_scalar('Val Error', ValError, epoch * len(Training_DataLoader) + i)
                    writer.add_scalar('Test Error', TotalTestError, epoch * len(Training_DataLoader) + i)
                    writer.add_scalar('Loss', 100 * running_loss, epoch * len(Training_DataLoader) + i)
                    writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'],
                                      epoch * len(Training_DataLoader) + i)
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
