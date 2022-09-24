import argparse
import math
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
from datasets.SelfSupervisionPairwiseDataset import SelfSupervisionPairwiseDataset
from networks.MultiscaleTransformerEncoder import MultiscaleTransformerEncoder
from networks.losses import OnlineHardNegativeMiningTripletLoss
from util.read_hdf5_data import read_hdf5_data
from util.utils import load_model, MultiEpochsDataLoader, MyGradScaler, save_best_model_stats, evaluate_test, \
    load_test_datasets, evaluate_validation, load_validation_set
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
        train_file = os.path.join(ds_path, 'ubc\\liberty_train_full.hdf5')
    elif ds_name == 'ubc-notredame':
        test_dir = os.path.join(ds_path, 'ubc\\test_lib_yos\\')
        train_file = os.path.join(ds_path, 'ubc\\notredame_train_full.hdf5')
    elif ds_name == 'ubc-yosemite':
        test_dir = os.path.join(ds_path, 'ubc\\test_lib_not\\')
        train_file = os.path.join(ds_path, 'ubc\\yosemite_train_full.hdf5')
    return train_file, test_dir


def create_optimizer(net, lr_rate, weight_decay):
    return torch.optim.Adam(
        [{'params': filter(lambda p: p.requires_grad == True, net.parameters()), 'lr': lr_rate,
          'weight_decay': weight_decay},
         {'params': filter(lambda p: p.requires_grad == False, net.parameters()), 'lr': 0,
          'weight_decay': 0}],
        lr=0, weight_decay=0)


def train(net, train_dataloader, start_epoch, device, warmup_epochs, generator_mode, lr_rate, weight_decay,
          writer, evaluate_net_steps, models_dir, best_file_name, outer_batch_size, inner_batch_size,
          optimizer, scheduler, scheduler_warmup, criterion, lowest_err, arch_desc, test_data, val_data, val_labels,
          epochs, scheduler_patience):
    scaler = MyGradScaler()

    for epoch in range(start_epoch, epochs):
        optimizer.zero_grad()
        is_warmup_phase = epoch - start_epoch < warmup_epochs

        if is_warmup_phase:
            print('\n', colored('Warmup step #' + repr(epoch - start_epoch), 'green', attrs=['reverse', 'blink']))
            scheduler_warmup.step()
        else:
            if epoch > start_epoch:
                if type(scheduler).__name__ == 'StepLR':
                    scheduler.step()

                if type(scheduler).__name__ == 'ReduceLROnPlateau':
                    scheduler.step(total_test_err)
        running_loss = 0

        log = 'LR: '
        for param_group in optimizer.param_groups:
            log += repr(param_group['lr']) + ' '
        print('\n', colored(log, 'blue', attrs=['reverse', 'blink']))

        print('negative_mining_mode: ' + criterion.mode)
        print('generator_mode: ' + generator_mode)

        should_mine_hard_negatives = criterion.mode == 'Random' and \
                                     optimizer.param_groups[0]['lr'] <= (lr_rate / 1e3 + 1e-8) and \
                                     not is_warmup_phase
        if should_mine_hard_negatives:
            print(colored('Switching Random->Hardest', 'green', attrs=['reverse', 'blink']))
            criterion = OnlineHardNegativeMiningTripletLoss(margin=1, mode='Hardest', device=device)

            optimizer = create_optimizer(net, lr_rate, weight_decay)

            scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=1, total_epoch=warmup_epochs)
            start_epoch = epoch

            if type(scheduler).__name__ == 'StepLR':
                scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

            if type(scheduler).__name__ == 'ReduceLROnPlateau':
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=scheduler_patience,
                                              verbose=True)

        bar = tqdm(train_dataloader, 0, leave=False, total=math.ceil((len(train_dataloader) - 1) / inner_batch_size))
        for batch_num, data in enumerate(bar):

            # zero the parameter gradients
            optimizer.zero_grad()

            net = net.train()

            # get the inputs
            pos1 = data['pos1']
            pos2 = data['pos2']

            pos1 = np.reshape(pos1, (pos1.shape[0] * pos1.shape[1], 1, pos1.shape[2], pos1.shape[3]), order='F')
            pos2 = np.reshape(pos2, (pos2.shape[0] * pos2.shape[1], 1, pos2.shape[2], pos2.shape[3]), order='F')

            pos1, pos2 = pos1.to(device), pos2.to(device)

            emb = net(pos1, pos2)

            loss = criterion(emb['Emb1'], emb['Emb2']) + criterion(emb['Emb2'], emb['Emb1'])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            # if epoch >= 90:
            #     evaluate_net_steps = 20
            if (batch_num % evaluate_net_steps == 0 or batch_num * inner_batch_size >= len(train_dataloader) - 1) and \
                    batch_num > 0:

                if batch_num > 0:
                    running_loss /= batch_num

                net.eval()
                val_err = 0
                if len(val_data) > 0:
                    val_err = evaluate_validation(net, val_data, val_labels, device)

                # test accuracy
                total_test_err = evaluate_test(net, test_data, device)

                state = {'epoch': epoch,
                         'state_dict': net.module.state_dict(),
                         'optimizer_name': type(optimizer).__name__,
                         'optimizer': optimizer,
                         'scheduler_name': type(scheduler).__name__,
                         'scheduler': scheduler,
                         'arch_desc': arch_desc,
                         'lowest_err': lowest_err,
                         'outer_batch_size': outer_batch_size,
                         'inner_batch_size': inner_batch_size,
                         'negative_mining_mode': criterion.mode,
                         'generator_mode': generator_mode,
                         'loss': criterion.mode}

                if total_test_err < lowest_err:
                    lowest_err = total_test_err

                    print('\n', colored('Best error found and saved: ' + repr(total_test_err)[0:5], 'red',
                                        attrs=['reverse', 'blink']))
                    filepath = os.path.join(models_dir, best_file_name + '.pth')
                    torch.save(state, filepath)
                    save_best_model_stats(models_dir, epoch, total_test_err, test_data)

                log = '[%d, %5d] Loss: %.3f' % (epoch, batch_num, 100 * running_loss) + ' Val Error: ' + repr(val_err)[
                                                                                                         0:6]
                log += ' Test Error: ' + repr(total_test_err)[0:6]
                print(log)

                writer.add_scalar('Val Error', val_err, epoch * len(train_dataloader) + batch_num)
                writer.add_scalar('Test Error', total_test_err, epoch * len(train_dataloader) + batch_num)
                writer.add_scalar('Loss', 100 * running_loss, epoch * len(train_dataloader) + batch_num)
                writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'],
                                  epoch * len(train_dataloader) + batch_num)
                writer.add_text('Log', log)
                writer.close()

                # save epoch
                filepath = models_dir + 'model_epoch_' + repr(epoch) + '.pth'
                torch.save(state, filepath)

            if (batch_num * inner_batch_size) > (len(train_dataloader) - 1):
                bar.clear()
                bar.close()
                break

    print('Finished Training')


def parse_args():
    parser = argparse.ArgumentParser(description='Train models for multimodal patch matching.')
    parser.add_argument('--epochs', type=int, default=90, help='epochs')
    parser.add_argument('--artifacts', default='./artifacts', help='artifacts path')
    parser.add_argument('--exp-name', default='symmetric_enc_transformer_test_4', help='experiment name')
    parser.add_argument('--evaluate-every', type=int, default=100, help='evaluate network and print steps')
    parser.add_argument('--skip-validation', type=bool, const=True, default=False,
                        help='whether to skip validation evaluation', nargs='?')
    parser.add_argument('--skip-test', type=bool, const=True, default=False,
                        help='whether to skip test evaluation', nargs='?')
    parser.add_argument('--continue-from-checkpoint', type=bool, const=True, default=False,
                        nargs='?', help='whether to continue training from checkpoint')
    parser.add_argument('--continue-from-best-score', type=bool, const=True, default=False,
                        nargs='?', help='whether to use best score when continuing training')
    parser.add_argument('--continue-from-best-model', type=bool, const=True, default=True,
                        nargs='?', help='whether to continue training using best model')
    parser.add_argument('--batch-size', type=int, default=48, help='batch size')
    parser.add_argument('--inner-batch-size', type=int, default=24, help='inner batch size of positive pairs')
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--weight-decay', type=float, default=0, help='weight decay')
    parser.add_argument('--dataset-name', default='visnir', help='dataset name')
    parser.add_argument('--dataset-path', default='visnir', help='dataset name')
    parser.add_argument('--warmup-epochs', type=int, default=14, help='warmup epochs')
    parser.add_argument('--scheduler-patience', type=int, default=6, help='scheduler patience epochs')
    parser.add_argument('--desc-dim', type=int, default=128, help='descriptor dimensions')
    parser.add_argument('--ssl', type=bool, const=True, default=False, help='whether to perform ssl training',
                        nargs='?')

    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpus_num = torch.cuda.device_count()
    torch.cuda.empty_cache()
    GPUtil.showUtilization()
    print('Using', device)

    models_dir = os.path.join(args.artifacts, args.exp_name, 'models')
    logs_dirname = os.path.join(args.artifacts, args.exp_name, 'logs')
    arch_desc = 'Symmetric CNN with Triplet loss and transformer encoder'
    best_file_name = 'best_model'

    train_file, test_dir = load_datasets_paths(args.dataset_name, args.dataset_path)

    assert_dir(models_dir)
    assert_dir(logs_dirname)

    start_best_model = args.continue_from_best_model
    use_best_score = args.continue_from_best_score
    writer = SummaryWriter(logs_dirname)
    generator_mode = 'Pairwise'
    negative_mining_mode = 'Random'
    skip_validation = args.skip_validation
    skip_test = args.skip_test
    lr_rate = args.lr
    weight_decay = args.weight_decay
    dropout = args.dropout
    outer_batch_size = args.batch_size
    inner_batch_size = args.inner_batch_size
    epochs = args.epochs
    scheduler_patience = args.scheduler_patience
    augmentations = {
        "Test": False,
        "HorizontalFlip": True,
        "Rotate90": True,
        "VerticalFlip": False,
        "RandomCrop": {'Do': False}
    }
    evaluate_net_steps = args.evaluate_every

    data = read_hdf5_data(train_file)
    train_data = data['Data']
    train_labels = np.squeeze(data['Labels'])
    train_split = np.squeeze(data['Set'])
    del data

    val_data = []
    val_labels = []
    train_indices = np.squeeze(np.asarray(np.where(train_split == 1)))
    if not skip_validation:
        val_data, val_labels = load_validation_set(train_data, train_split, train_labels)

    train_data = np.squeeze(train_data[train_indices,])
    train_labels = train_labels[train_indices]

    if not args.ssl:
        train_dataset = DatasetPairwiseTriplets(train_data, train_labels, inner_batch_size, augmentations,
                                                generator_mode)
    else:
        print("Training SSL...")
        train_dataset = SelfSupervisionPairwiseDataset(train_data, inner_batch_size, augmentations)
    train_dataloader = MultiEpochsDataLoader(train_dataset, batch_size=outer_batch_size, shuffle=True,
                                             num_workers=8, pin_memory=True)

    test_data = None
    if not skip_test:
        test_data = load_test_datasets(test_dir)

    net = MultiscaleTransformerEncoder(dropout, encoder_dim=args.desc_dim)
    optimizer = create_optimizer(net, lr_rate, weight_decay)
    start_epoch = 0
    lowest_err = 1e10
    if args.continue_from_checkpoint:
        net, optimizer, lowest_err, start_epoch, scheduler, loaded_negative_mining_mode = load_model(net,
                                                                                                     start_best_model,
                                                                                                     models_dir,
                                                                                                     best_file_name,
                                                                                                     use_best_score,
                                                                                                     device)

    if gpus_num > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        net = nn.DataParallel(net)
    net.to(device)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=scheduler_patience, verbose=True)

    warmup_epochs = args.warmup_epochs
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                                after_scheduler=StepLR(optimizer, step_size=3, gamma=0.1))

    criterion = OnlineHardNegativeMiningTripletLoss(margin=1, mode=negative_mining_mode, device=device)

    train(net, train_dataloader, start_epoch, device, warmup_epochs, generator_mode, lr_rate, weight_decay,
          writer, evaluate_net_steps, models_dir, best_file_name, outer_batch_size, inner_batch_size,
          optimizer, scheduler, scheduler_warmup, criterion, lowest_err, arch_desc, test_data, val_data, val_labels,
          epochs, scheduler_patience)


if __name__ == '__main__':
    main()
