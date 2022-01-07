import glob
import json
import os

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from util.read_hdf5_data import read_hdf5_data


def load_model(net, start_best_model, models_dirname, best_filename, use_best_score, device, load_epoch=None):
    scheduler = None
    optimizer = None

    lowest_err = 1e5

    negative_mining_mode = 'Random'

    if start_best_model:
        flist = glob.glob(models_dirname + best_filename + '.pth')
    else:
        flist = glob.glob(models_dirname + "model*")

    if flist:
        flist.sort(key=os.path.getmtime)

        if load_epoch is not None:
            model_path = models_dirname + 'model_epoch_%s.pth' % load_epoch
            print('%s loaded' % model_path)
            checkpoint = torch.load(model_path)
        else:
            print(flist[-1] + ' loaded')
            checkpoint = torch.load(flist[-1])

        if ('lowest_err' in checkpoint.keys()) and use_best_score:
            lowest_err = checkpoint['lowest_err']

        if 'negative_mining_mode' in checkpoint.keys():
            negative_mining_mode = checkpoint['negative_mining_mode']

        net_dict = net.state_dict()
        checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if
                                    (k in net_dict) and (net_dict[k].shape == checkpoint['state_dict'][k].shape)}

        net.load_state_dict(checkpoint['state_dict'], strict=False)

        if 'optimizer_name' in checkpoint.keys():
            optimizer = torch.optim.Adam(net.parameters())
            try:
                optimizer = checkpoint['optimizer']
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda(device)
            except Exception as e:
                print(e)
                print('Optimizer loading error')

        if ('scheduler_name' in checkpoint.keys()) and (optimizer != None):

            try:
                if checkpoint['scheduler_name'] == 'ReduceLROnPlateau':
                    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, verbose=True)

                scheduler = checkpoint['scheduler']
            except Exception as e:
                print(e)
                print('Optimizer loading error')

        start_epoch = checkpoint['epoch'] + 1
    else:
        print('Weights file not loaded')
        optimizer = None
        start_epoch = 0

    print('lowest_err: ' + repr(lowest_err)[0:6])

    return net, optimizer, lowest_err, start_epoch, scheduler, negative_mining_mode


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class MyGradScaler:
    def __init__(self):
        pass

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass


def save_best_model_stats(dir, epoch, test_err, test_data):
    content = {
        'Test error': test_err,
        'Epoch': epoch
    }
    for test_set in test_data:
        if isinstance(test_data[test_set], dict):
            content[f'Test set {test_set} error'] = test_data[test_set]['TestError']
    fpath = os.path.join(dir, 'visnir_best_model_stats.json')
    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=4)


def FPR95Accuracy(dist_mat, labels):
    pos_indices = np.squeeze(np.asarray(np.where(labels == 1)))
    neg_indices = np.squeeze(np.asarray(np.where(labels == 0)))

    neg_dists = dist_mat[neg_indices]
    pos_dists = np.sort(dist_mat[pos_indices])

    thresh = pos_dists[int(0.95 * pos_dists.shape[0])]

    fp = sum(neg_dists < thresh)

    return fp / float(neg_dists.shape[0])


def FPR95Threshold(PosDist):
    PosDist = PosDist.sort(dim=-1, descending=False)[0]
    Val = PosDist[int(0.95 * PosDist.shape[0])]

    return Val


def normalize_image(x):
    return x / (255.0 / 2)


def evaluate_network(net, data1, data2, device, step_size=800):
    with torch.no_grad():

        for k in range(0, data1.shape[0], step_size):

            a = data1[k:(k + step_size), :, :, :]
            b = data2[k:(k + step_size), :, :, :]

            # a, b = a.to(device), b.to(device)
            x = net(a, b)

            if k == 0:
                keys = list(x.keys())
                emb = dict()
                for key in keys:
                    emb[key] = np.zeros(tuple([data1.shape[0]]) + tuple(x[key].shape[1:]), dtype=np.float32)

            for key in keys:
                emb[key][k:(k + step_size)] = x[key].cpu()

    return emb


def load_test_datasets(test_dir):
    file_list = glob.glob(test_dir + "*.hdf5")
    test_data = dict()
    for f in file_list:
        path, dataset_name = os.path.split(f)
        dataset_name = os.path.splitext(dataset_name)[0]

        data = read_hdf5_data(f)

        x = data['Data'].astype(np.float32)
        test_labels = torch.from_numpy(np.squeeze(data['Labels']))
        del data

        x[:, :, :, :, 0] -= x[:, :, :, :, 0].mean()
        x[:, :, :, :, 1] -= x[:, :, :, :, 1].mean()

        x = normalize_image(x)
        x = torch.from_numpy(x)

        test_data[dataset_name] = dict()
        test_data[dataset_name]['Data'] = x
        test_data[dataset_name]['Labels'] = test_labels
        del x
    return test_data


def load_validation_set(train_data, train_split, train_labels):
    val_indices = np.squeeze(np.asarray(np.where(train_split == 3)))

    # VALIDATION data
    val_labels = torch.from_numpy(train_labels[val_indices])

    val_data = train_data[val_indices, :, :, :].astype(np.float32)
    val_data[:, :, :, :, 0] -= val_data[:, :, :, :, 0].mean()
    val_data[:, :, :, :, 1] -= val_data[:, :, :, :, 1].mean()
    val_data = torch.from_numpy(normalize_image(val_data))

    return val_data, val_labels


def evaluate_test(net, test_data, device, step_size=800):
    samples_amount = 0
    total_test_err = 0
    for dataset_name in test_data:
        dataset = test_data[dataset_name]
        emb = evaluate_network(net, dataset['Data'][:, :, :, :, 0], dataset['Data'][:, :, :, :, 1], device, step_size)

        dist = np.power(emb[0] - emb[1], 2).sum(1)
        dataset['TestError'] = FPR95Accuracy(dist, dataset['Labels']) * 100
        total_test_err += dataset['TestError'] * dataset['Data'].shape[0]
        samples_amount += dataset['Data'].shape[0]
    total_test_err /= samples_amount

    del emb
    return total_test_err


def evaluate_validation(net, val_data, val_labels, device):
    val_emb = evaluate_network(net, val_data[:, :, :, :, 0], val_data[:, :, :, :, 1], device)

    dist = np.power(val_emb['Emb1'] - val_emb['Emb2'], 2).sum(1)
    val_err = FPR95Accuracy(dist, val_labels) * 100
    return val_err
