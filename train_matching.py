import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import copy
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import argparse
from datasets.HDF5Dataset import HDF5Dataset
from networks.MetricLearningCNN import MetricLearningCNN
from losses import ContrastiveLoss, TripletLoss, OnlineTripletLoss, OnlineHardNegativeMiningTripletLoss
from losses import InnerProduct, FindFprTrainingSet, FPRLoss, PairwiseLoss, HardTrainingLoss
from multiprocessing import freeze_support
from read_matlab_imdb import read_matlab_imdb
from losses import Compute_FPR_HardNegatives, ComputeFPR
from utils import get_torch_device
from datasets.PairwiseTriplets import PairwiseTriplets
import warnings
from utils import NormalizeImages, FPR95Accuracy, EvaluateNet

warnings.filterwarnings("ignore", message="UserWarning: albumentations.augmentations.transforms.RandomResizedCrop")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def load_test_datasets(test_path, batch_size):
    test_files = glob.glob(os.path.join(test_path, "*.hdf5"))
    test_loaders = {}
    for test_file in test_files:
        path, dataset_name = os.path.split(test_file)
        dataset_name = os.path.splitext(dataset_name)[0]

        print(f'Loading test dataset {dataset_name}...')
        test_dataset = HDF5Dataset(dataset_name, path, test_file)
        test_loaders[dataset_name] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    return test_loaders


def evaluate_validation(net, val_data, val_labels, device, batch_size, cnn_mode):
    net.eval()
    # StepSize = 512
    embedding1 = EvaluateNet(net.module.GetChannelCnn(0, cnn_mode), val_data[:, :, :, :, 0], device,
                             batch_size)
    embedding2 = EvaluateNet(net.module.GetChannelCnn(1, cnn_mode), val_data[:, :, :, :, 1], device,
                             batch_size)
    dist = np.power(embedding1 - embedding2, 2).sum(1)
    val_error = FPR95Accuracy(dist, val_labels) * 100

    # plt.hist(dist[np.where(val_labels==0)[0]], 10)
    # plt.hist(dist[np.where(val_labels==1)[0]], 10)

    # estimate fpr95 threshold
    positives_indices = np.squeeze(np.asarray(np.where(val_labels == 1)))
    curr_FPR95 = np.sort(dist[positives_indices])[int(0.95 * positives_indices.shape[0])]
    return curr_FPR95, val_error


def evaluate_test(net, test_loaders, device, cnn_mode, test_decimation, generator_mode):
    if (generator_mode == 'Pairwise') | (generator_mode == 'PairwiseRot'):

        test_samples_amount = 0
        test_error = 0
        net.eval()
        for dataset in test_loaders:
            test_loader = test_loaders[dataset]
            max_samples = int(len(test_loader) / test_decimation)
            seen_samples = 0
            embeddings1 = []
            embeddings2 = []
            labels = []
            for _, (data, batch_labels) in enumerate(
                    tqdm(test_loader, total=max_samples, desc=f'Test {test_loader.dataset.dataset_name}', position=1,
                         leave=False)):
                batch_size = data.shape[0]
                embedding1 = EvaluateNet(net.module.GetChannelCnn(0, cnn_mode),
                                         data[:, :, :, :, 0], device,
                                         batch_size)
                embedding2 = EvaluateNet(net.module.GetChannelCnn(1, cnn_mode),
                                         data[:, :, :, :, 1], device,
                                         batch_size)
                embeddings1.append(embedding1)
                embeddings2.append(embedding2)
                labels.append(batch_labels)
                seen_samples += batch_size
                if seen_samples >= max_samples:
                    break
            embedding1 = np.concatenate(embeddings1)
            embedding2 = np.concatenate(embeddings2)
            labels = np.concatenate(labels)
            dist = np.power(embedding1 - embedding2, 2).sum(1)
            test_error = FPR95Accuracy(dist, labels) * 100
            test_error += test_error * len(test_loader)
            test_samples_amount += len(test_loader)
        test_error /= test_samples_amount

        if (net.module.Mode == 'Hybrid1') | (net.module.Mode == 'Hybrid2'):
            net.module.Mode = 'Hybrid'

        return test_error


def train(train_loader, val_data, val_labels, test_loaders, net, optimizer, criterion, scheduler, device,
          start_epoch, grad_accumulation_steps, generator_mode, cnn_mode, FPR95, fpr_hard_negatives, fpr_max_images_num,
          batch_size, pairwise_triplets_batch_size, tb_writer, test_decimation, models_dir_name, best_file_name,
          out_fname, architecture_description, evaluate_every, lowest_error, skip_validation, skip_test):
    for epoch in range(start_epoch, 80):
        running_loss = 0
        running_loss_ce = 0
        running_loss_pos = 0
        running_loss_neg = 0
        optimizer.zero_grad()

        for batch_num, Data in enumerate(tqdm(train_loader, desc='Train', position=0), 1):
            net.train()

            # get the inputs
            pos1 = Data['pos1']
            pos2 = Data['pos2']

            pos1 = np.reshape(pos1, (pos1.shape[0] * pos1.shape[1], 1, pos1.shape[2], pos1.shape[3]), order='F')
            pos2 = np.reshape(pos2, (pos2.shape[0] * pos2.shape[1], 1, pos2.shape[2], pos2.shape[3]), order='F')

            if generator_mode == 'Pairwise':

                if (cnn_mode == 'PairwiseAsymmetric') | (cnn_mode == 'PairwiseSymmetric'):

                    if fpr_hard_negatives:
                        pos1, pos2 = Compute_FPR_HardNegatives(net, pos1, pos2, device, FprValPos=FPR95,
                                                               FprValNeg=1.5 * FPR95, MaxNoImages=fpr_max_images_num)

                    pos1, pos2 = pos1.to(device), pos2.to(device)
                    embeddings = net(pos1, pos2)
                    loss = criterion(embeddings['Emb1'], embeddings['Emb2']) + criterion(embeddings['Emb2'],
                                                                                         embeddings['Emb1'])

                if cnn_mode == 'Hybrid':

                    # GPUtil.showUtilization()
                    if fpr_hard_negatives:

                        Embed = Compute_FPR_HardNegatives(net, pos1, pos2, device, FprValPos=0.9 * FPR95,
                                                          FprValNeg=1.1 * FPR95, MaxNoImages=fpr_max_images_num)

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
                            loss -= neg_loss1
                            neg_loss = neg_loss1.item()

                        # loss -= PairwiseLoss(EmbedNegA['EmbAsym1'], EmbedNegA['EmbAsym2'])

                        # del EmbedNegA

                        if (Embed['NegIdxB1'].nelement() > 1) & (Embed['NegIdxB1'].shape[0] > 1):
                            Embed['NegIdxB1'], Embed['NegIdxB2'] = Embed['NegIdxB1'].to(device), Embed['NegIdxB2'].to(
                                device)
                            EmbedNegB = net(Embed['NegIdxB1'], Embed['NegIdxB2'])

                            neg_loss2 = PairwiseLoss(EmbedNegB['Hybrid1'], EmbedNegB['Hybrid2'])
                            loss -= neg_loss2
                            neg_loss += neg_loss2.item()

                        del Embed, EmbedPos

                        running_loss_neg += neg_loss
                        running_loss_pos += pos_loss
                    else:
                        pos1, pos2 = pos1.to(device), pos2.to(device)

                        # loss  = HardTrainingLoss(net, pos1, pos2,PosRatio=0.5,HardRatio=0.5,T=1,device=device)
                        # loss += HardTrainingLoss(net, pos2, pos1, PosRatio=0.5, HardRatio=0.5, T=1, device=device)

                        # Embed = net(pos1, pos2, p=0.3)
                        # loss = criterion(Embed['Hybrid1'], Embed['Hybrid2']) + criterion(Embed['Hybrid2'],Embed['Hybrid1'])
                        # loss += InnerProductLoss(Embed['EmbAsym1'], Embed['EmbSym1']) + InnerProductLoss(Embed['EmbAsym2'],Embed['EmbSym2'])
                        # loss +=criterion(Embed['EmbSym1'], Embed['EmbSym2']) + criterion(Embed['EmbSym2'],Embed['EmbSym1'])
                        # loss +=criterion(Embed['EmbAsym1'], Embed['EmbAsym2']) + criterion(Embed['EmbAsym2'],Embed['EmbAsym1'])

                        # TrainFpr = ComputeFPR(Embed['Hybrid1'], Embed['Hybrid2'], FPR95 * 0.9, FPR95 * 1.1)
                        # print('TrainFpr = ' + repr(TrainFpr))

            loss /= grad_accumulation_steps

            # backward + optimize
            loss.backward()

            if ((batch_num + 1) % grad_accumulation_steps) == 0:  # Wait for several backward steps
                optimizer.step()  # Now we can do an optimizer step

                # zero the parameter gradients
                optimizer.zero_grad()

            running_loss += loss.item()

            if (batch_num % evaluate_every == 0):
                running_loss /= evaluate_every / grad_accumulation_steps
                running_loss_ce /= evaluate_every
                scheduler.step(running_loss)
                running_loss_neg /= evaluate_every
                running_loss_pos /= evaluate_every

                tqdm.write('running_loss_neg: ' + repr(100 * running_loss_neg)[0:5] + ' running_loss_pos: ' + repr(
                    100 * running_loss_pos)[0:5])

                print_val_fpr = ''
                if not skip_validation:
                    curr_FPR95, val_error = evaluate_validation(net, val_data, val_labels, device, batch_size, cnn_mode)
                    print_val_fpr = 'FPR95: ' + repr(curr_FPR95) + ' '

                loss = running_loss / batch_num
                tqdm.write(print_val_fpr + 'Loss: ' + repr(loss))

                if (net.module.Mode == 'Hybrid1') | (net.module.Mode == 'Hybrid2'):
                    net.module.Mode = 'Hybrid'

                if (batch_num % 2000 == 0) and (batch_num > 0):
                    # FPR95 = curr_FPR95
                    tqdm.write('FPR95 changed: ' + repr(FPR95)[0:5])

                val_err_str = ''
                if not skip_validation:
                    val_err_str = ' Val Error: ' + repr(val_error)[0:6]
                str = '[%d, %5d] loss: %.3f' % (epoch, batch_num, 100 * running_loss) + val_err_str
                if running_loss_ce > 0:
                    str += ' Rot loss: ' + repr(running_loss_ce)[0:6]

                # for dataset in test_loaders:
                #   str +=' ' + dataset + ': ' + repr(test_loaders[dataset]['TestError'])[0:6]
                str += ' FPR95 = ' + repr(FPR95)[0:6]
                tqdm.write(str)

                if not skip_validation:
                    tb_writer.add_scalar('Val Error', val_error, epoch * len(train_loader) + batch_num)
                tb_writer.add_scalar('Loss', 100 * running_loss, epoch * len(train_loader) + batch_num)
                tb_writer.add_scalar('FPR95', FPR95, epoch * len(train_loader) + batch_num)
                tb_writer.add_text('Text', str)
                tb_writer.close()

        if not skip_test:
            test_error = evaluate_test(net, test_loaders, device, cnn_mode, test_decimation, generator_mode)
            if not test_error:
                raise Exception('Test error after evaluation test is None')

            tb_writer.add_scalar('Test Error', test_error, epoch * len(train_loader) + batch_num)
            tqdm.write('Mean Test Error: ' + repr(test_error)[0:6])
            if test_error < lowest_error:
                lowest_error = test_error

                tqdm.write(f'Best test error found and saved: {repr(lowest_error)[0:5]}')
                filepath = os.path.join(models_dir_name, best_file_name + '.pth')
                state = {'epoch': epoch,
                         'state_dict': net.module.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'Description': architecture_description,
                         'LowestError': lowest_error,
                         'OuterBatchSize': batch_size,
                         'InnerBatchSize': pairwise_triplets_batch_size,
                         'Mode': net.module.Mode,
                         'CnnMode': cnn_mode,
                         'GeneratorMode': generator_mode,
                         'Loss': criterion.Mode,
                         'FPR95': FPR95}
                torch.save(state, filepath)
        else:
            # save epoch
            filepath = os.path.join(models_dir_name, f'{out_fname}_{epoch}.pth')
            state = {'epoch': epoch,
                     'state_dict': net.module.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'Description': architecture_description,
                     'OuterBatchSize': batch_size,
                     'InnerBatchSize': pairwise_triplets_batch_size,
                     'Mode': net.module.Mode,
                     'CnnMode': cnn_mode,
                     'GeneratorMode': generator_mode,
                     'Loss': criterion.Mode,
                     'FPR95': FPR95}

            torch.save(state, filepath)

    print('Finished training')


def load_trainval_dataset(trainval_dir, augmentations, generator_mode, batch_size, pairwise_triplets_batch_size):
    trainval_imdb = read_matlab_imdb(trainval_dir)
    trainval_data = trainval_imdb['Data']
    trainval_labels = np.squeeze(trainval_imdb['Labels'])
    trainval_set = np.squeeze(trainval_imdb['Set'])

    # ShowTwoImages(trainval_data[0:3, :, :, 0])
    # ShowTwoRowImages(trainval_data[0:3, :, :, 0], trainval_data[0:3, :, :, 1])

    train_indices = np.squeeze(np.asarray(np.where(trainval_set == 1)))
    val_indices = np.squeeze(np.asarray(np.where(trainval_set == 3)))

    # get val data
    val_labels = torch.from_numpy(trainval_labels[val_indices])
    val_data = trainval_data[val_indices].astype(np.float32)

    # ShowTwoRowImages(np.squeeze(val_data[0:4, :, :, :, 0]), np.squeeze(val_data[0:4, :, :, :, 1]))
    # val_data = torch.from_numpy(val_data).float().cpu()
    val_data[:, :, :, :, 0] -= val_data[:, :, :, :, 0].mean()
    val_data[:, :, :, :, 1] -= val_data[:, :, :, :, 1].mean()
    val_data = torch.from_numpy(NormalizeImages(val_data))

    # train data
    train_data = np.squeeze(trainval_data[train_indices,])
    train_labels = trainval_labels[train_indices]

    # define generators
    train_dataset = PairwiseTriplets(train_data, train_labels, pairwise_triplets_batch_size, augmentations,
                                     generator_mode)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    return train_loader, val_data, val_labels


def load_checkpoint(models_dir_name, continue_from_best_model, net, optimizer):
    print('Continuing from checkpoint')
    if continue_from_best_model:
        print('Loading best model')
        model_checkpoints = glob.glob(os.path.join(models_dir_name, "visnir_best.pth"))
    else:
        print('Loading last model')
        model_checkpoints = glob.glob(os.path.join(models_dir_name, "visnir*"))

    if len(model_checkpoints) == 0:
        raise Exception(f"Tried continuing from checkpoint but found nothing in {models_dir_name}")

    model_checkpoints.sort(key=os.path.getmtime)
    checkpoint_path = model_checkpoints[-1]
    checkpoint = torch.load(checkpoint_path)
    print(f'{checkpoint_path} loaded')

    net.load_state_dict(checkpoint['state_dict'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])

    start_epoch = checkpoint['epoch'] + 1

    FPR95 = None
    if 'FPR95' in checkpoint:
        FPR95 = checkpoint['FPR95']
        print('Loaded FPR95 = ' + repr(FPR95))

    lowest_error = checkpoint['LowestError']
    print(f'lowest_error: {lowest_error}')
    return start_epoch, lowest_error, FPR95


def assymetric_init(net):
    net.module.netAS1 = copy.deepcopy(net.module.netS)
    net.module.netAS2 = copy.deepcopy(net.module.netS)


def parse_args():
    parser = argparse.ArgumentParser(description='Train models for multimodal patch matching.')
    parser.add_argument('--models', help='models path')
    parser.add_argument('--logs', help='logs path')
    parser.add_argument('--test', help='test data path')
    parser.add_argument('--trainval', help='trainval data path')
    parser.add_argument('--evaluate-every', type=int, default=1000, help='evaluate network and print steps')
    parser.add_argument('--skip-validation', type=bool, default=False, help='whether to skip validation evaluation')
    parser.add_argument('--skip-test', type=bool, default=False, help='whether to skip test evaluation')
    parser.add_argument('--continue-from-checkpoint', type=bool, default=True,
                        help='whether to continue training from checkpoint')
    parser.add_argument('--continue-from-best-model', type=bool, default=True,
                        help='whether to continue training using best model')
    parser.add_argument('--batch-size', type=int, default=24, help='batch size')
    parser.add_argument('--pairs-triplets-batch-size', type=int, default=6, help='batch size of pairs/triplets')
    parser.add_argument('--lr', type=float, default=1e-4, help='batch size of pairs/triplets')

    return parser.parse_args()


def main():
    freeze_support()
    args = parse_args()
    device = get_torch_device()
    models_dir_name = args.models
    tb_writer = SummaryWriter(args.logs)
    architecture_description = 'Symmetric CNN with Triplet loss, no HM'
    best_file_name = 'visnir_best'
    out_fname = 'visnir_sym_triplet'
    # TrainFile = '/home/keller/Dropbox/multisensor/python/data/Vis-Nir_Train.mat'
    # TrainFile = './data/Vis-Nir_grid/Vis-Nir_grid_Train.hdf5'
    test_decimation = 1000
    FPR95 = 0.8
    fpr_max_images_num = 400
    fpr_hard_negatives = False

    lowest_error = 1e10

    # ----------------------------     configuration   ---------------------------
    augmentations = {}
    augmentations["HorizontalFlip"] = False
    augmentations["VerticalFlip"] = False
    augmentations["Rotate90"] = True
    augmentations["Test"] = {'Do': True}
    augmentations["RandomCrop"] = {'Do': False, 'MinDx': 0, 'MaxDx': 0.2, 'MinDy': 0, 'MaxDy': 0.2}

    # default values
    freeze_symmetric_cnn = True
    freeze_asymmetric_cnn = True

    grad_accumulation_steps = 1

    if args.skip_validation and args.skip_test:
        raise Exception('Cannot skip both validation and test')

    if True:
        # generator_mode = 'PairwiseRot'
        generator_mode = 'Pairwise'
        cnn_mode = 'PairwiseSymmetric'
        # criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='Random')
        criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='Hardest')
        # criterion         = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='MostHardest', HardRatio=1.0/8)
        architecture_description = 'PairwiseSymmetric Hardest'

        lr = args.lr
        batch_size = args.batch_size
        pairwise_triplets_batch_size = args.pairs_triplets_batch_size
        augmentations["Test"] = {'Do': False}
        augmentations["HorizontalFlip"] = True
        augmentations["VerticalFlip"] = True
        augmentations["Rotate90"] = True
        augmentations["RandomCrop"] = {'Do': True, 'MinDx': 0, 'MaxDx': 0.2, 'MinDy': 0, 'MaxDy': 0.2}
        grad_accumulation_steps = 1

        freeze_symmetric_cnn = False
        freeze_asymmetric_cnn = True

        fpr_hard_negatives = False

    if False:
        generator_mode = 'Pairwise'
        cnn_mode = 'PairwiseAsymmetric'
        # criterion     = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='Random')
        criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='Hardest')
        # criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='HardPos', HardRatio=1.0/2, PosRatio=1. / 2)

        freeze_symmetric_cnn = True
        freeze_asymmetric_cnn = False

        lr = args.lr
        batch_size = args.batch_size
        pairwise_triplets_batch_size = args.pairs_triplets_batch_size

        augmentations["Test"] = {'Do': False}
        augmentations["HorizontalFlip"] = True
        augmentations["VerticalFlip"] = True
        augmentations["Rotate90"] = True
        augmentations["RandomCrop"] = {'Do': True, 'MinDx': 0, 'MaxDx': 0.2, 'MinDy': 0, 'MaxDy': 0.2}

        architecture_description = 'PairwiseAsymmetric'
    if False:
        # generator_mode      = 'PairwiseRot'
        generator_mode = 'Pairwise'
        # cnn_mode            = 'HybridRot'
        cnn_mode = 'Hybrid'
        # criterion           = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='Random')
        # criterion           = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='Hardest')
        criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='HardPos', HardRatio=1.0 / 2, PosRatio=1. / 2)
        # HardestCriterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='Hardest')
        batch_size = args.batch_size  # was 16
        pairwise_triplets_batch_size = args.pairs_triplets_batch_size  # was 48
        lr = args.lr  # 1e-2

        fpr_hard_negatives = False

        freeze_symmetric_cnn = True
        freeze_asymmetric_cnn = False

        augmentations["Test"] = {'Do': False}
        augmentations["HorizontalFlip"] = True
        augmentations["VerticalFlip"] = True
        augmentations["Rotate90"] = True
        augmentations["RandomCrop"] = {'Do': True, 'MinDx': 0, 'MaxDx': 0.2, 'MinDy': 0, 'MaxDy': 0.2}
        grad_accumulation_steps = 1

        fpr_max_images_num = 400

        test_decimation = 20

    train_loader, val_data, val_labels = load_trainval_dataset(args.trainval, augmentations, generator_mode, batch_size,
                                                               pairwise_triplets_batch_size)
    test_loaders = load_test_datasets(args.test, batch_size)

    net = MetricLearningCNN(cnn_mode)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0)
    start_epoch = 0

    if args.continue_from_checkpoint:
        start_epoch, lowest_error, loaded_FPR95 = load_checkpoint(models_dir_name, args.continue_from_best_model, net,
                                                                  optimizer)
        if loaded_FPR95:
            FPR95 = loaded_FPR95

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)
    net.to(device)

    if 'asymmetric' in cnn_mode.lower():
        assymetric_init(net)

    if False:
        BaseCnnParams = net.module.BaseCnnParams()
        HeadCnnParams = net.module.HeadCnnParams()
        optimizer = torch.optim.Adam(
            [{'params': BaseCnnParams, 'weight_decay': 0}, {'params': HeadCnnParams, 'weight_decay': 0}],
            lr=lr)

        optimizer = optim.SGD([{'params': net.module.netS.parameters(), 'lr': 1e-5},
                               {'params': net.module.netAS1.parameters(), 'lr': 1e-5},
                               {'params': net.module.netAS2.parameters(), 'lr': 1e-5},
                               {'params': net.module.fc1.parameters(), 'lr': 1e-4},
                               {'params': net.module.fc2.parameters(), 'lr': 1e-4},
                               {'params': net.module.fc3.parameters(), 'lr': 1e-4}],
                              lr=lr, momentum=0.0, weight_decay=0.00)  # momentum=0.9

    if freeze_symmetric_cnn:
        net.module.freeze_symmetric_cnn()
    if freeze_asymmetric_cnn:
        net.module.freeze_asymmetric_cnn()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    train(train_loader, val_data, val_labels, test_loaders, net, optimizer, criterion, scheduler, device,
          start_epoch, grad_accumulation_steps, generator_mode, cnn_mode, FPR95, fpr_hard_negatives, fpr_max_images_num,
          batch_size, pairwise_triplets_batch_size, tb_writer, test_decimation, models_dir_name, best_file_name,
          out_fname, architecture_description, args.evaluate_every, lowest_error, args.skip_validation,
          args.skip_test)


if __name__ == '__main__':
    main()
