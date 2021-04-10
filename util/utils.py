import glob
import json
import os

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


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


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


class PairSelector:
    """
    Implementation should return indices of positive pairs and negative pairs that will be passed to compute
    Contrastive Loss
    return positive_pairs, negative_pairs
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class AllPositivePairSelector(PairSelector):
    """
    Discards embeddings and generates all possible pairs given labels.
    If balance is True, negative pairs are a random sample to match the number of positive samples
    """

    def __init__(self, balance=True):
        super(AllPositivePairSelector, self).__init__()
        self.balance = balance

    def get_pairs(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]
        if self.balance:
            negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]

        return positive_pairs, negative_pairs


class HardNegativePairSelector(PairSelector):
    """
    Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
    matching the number of positive pairs.
    """

    def __init__(self, cpu=True):
        super(HardNegativePairSelector, self).__init__()
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)

        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]

        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[:len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


class HardNegativesTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(HardNegativesTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[
                    torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                              negative_selection_fn=hardest_negative,
                                                                                              cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                             negative_selection_fn=random_hard_negative,
                                                                                             cpu=cpu)


def SemihardNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                               negative_selection_fn=lambda
                                                                                                   x: semihard_negative(
                                                                                                   x, margin),
                                                                                               cpu=cpu)


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


def FPR95Accuracy(Dist, Labels):
    PosIdx = np.squeeze(np.asarray(np.where(Labels == 1)))
    NegIdx = np.squeeze(np.asarray(np.where(Labels == 0)))

    NegDist = Dist[NegIdx]
    PosDist = np.sort(Dist[PosIdx])

    Val = PosDist[int(0.95 * PosDist.shape[0])]

    FalsePos = sum(NegDist < Val);

    FPR95Accuracy = FalsePos / float(NegDist.shape[0])

    return FPR95Accuracy


def FPR95Threshold(PosDist):
    PosDist = PosDist.sort(dim=-1, descending=False)[0]
    Val = PosDist[int(0.95 * PosDist.shape[0])]

    return Val


def normalize_image(x):
    return x / (255.0 / 2)


def evaluate_network(net, data1, data2, device, step_size):
    with torch.no_grad():

        for k in range(0, data1.shape[0], step_size):

            a = data1[k:(k + step_size), :, :, :]
            b = data2[k:(k + step_size), :, :, :]

            a, b = a.to(device), b.to(device)
            x = net(a, b)

            if k == 0:
                keys = list(x.keys())
                emb = dict()
                for key in keys:
                    emb[key] = np.zeros(tuple([data1.shape[0]]) + tuple(x[key].shape[1:]), dtype=np.float32)

            for key in keys:
                emb[key][k:(k + step_size)] = x[key].cpu()

    return emb
