from itertools import combinations
import numpy as np
import torch
import GPUtil
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def get_torch_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpus_num = torch.cuda.device_count()
    torch.cuda.empty_cache()
    GPUtil.showUtilization()

    print(f'Using: {device}')
    if device.type != "cpu":
        print(f'{gpus_num} GPUs available')

    return device


def NormalizeImages(x):
    # Result = (x/255.0-0.5)/0.5
    Result = x / (255.0 / 2)
    return Result


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


def EvaluateNet(net, data, device, batch_size):
    with torch.no_grad():
        for idx in range(0, data.shape[0], batch_size):

            # ShowTwoRowImages(x[0:3, :, :, 0], x[0:3, :, :, 1])
            batch = data[idx:(idx + batch_size), :, :, :]  # - my_training_Dataset.VisAvg

            # ShowTwoRowImages(a[0:10,0,:,:], b[0:10,0,:,:])
            batch = batch.to(device)
            batch_embeddings = net(batch)

            if idx == 0:
                embeddings = np.zeros((data.shape[0], batch_embeddings.shape[1]), dtype=np.float32)

            embeddings[idx:(idx + batch_size)] = batch_embeddings.cpu()

    return embeddings


def EvaluateDualNets(net, Data1, Data2, device, StepSize, p=0):
    with torch.no_grad():

        for k in range(0, Data1.shape[0], StepSize):

            # ShowTwoRowImages(x[0:3, :, :, 0], x[0:3, :, :, 1])
            a = Data1[k:(k + StepSize), :, :, :]  # - my_training_Dataset.VisAvg
            b = Data2[k:(k + StepSize), :, :, :]  # - my_training_Dataset.VisAvg

            # ShowTwoRowImages(a[0:10,0,:,:], b[0:10,0,:,:])
            a, b = a.to(device), b.to(device)
            x = net(a, b, p=p)

            if k == 0:
                keys = list(x.keys())
                Emb = dict()
                for key in keys:
                    Emb[key] = np.zeros((Data1.shape[0], x[key].shape[1]), dtype=np.float32)

            for key in keys:
                Emb[key][k:(k + StepSize)] = x[key].cpu()

    return Emb


def ComputeAllErros(TestData, net, device, StepSize):
    Errors = dict()
    Loss = 0
    for DataName in TestData:
        EmbTest1 = EvaluateNet(net.module.GetChannelCnn(0), TestData[DataName]['Data'][:, :, :, :, 0], device, StepSize)
        EmbTest2 = EvaluateNet(net.module.GetChannelCnn(1), TestData[DataName]['Data'][:, :, :, :, 1], device, StepSize)
        Dist = np.power(EmbTest1 - EmbTest2, 2).sum(1)
        Errors['TestError'] = FPR95Accuracy(Dist, TestData[DataName]['Labels'])
        Loss += Errors['TestError']

    Errors['Mean'] /= len(TestData)


def FPR95Accuracy(dist, labels):
    positive_indices = np.squeeze(np.asarray(np.where(labels == 1)))
    negative_indices = np.squeeze(np.asarray(np.where(labels == 0)))

    negative_dist = dist[negative_indices]
    positive_dist = np.sort(dist[positive_indices])

    recall_thresh = positive_dist[int(0.95 * positive_dist.shape[0])]

    fp = sum(negative_dist < recall_thresh)

    negatives = float(negative_dist.shape[0])
    if negatives == 0:
        return 0
    return fp / negatives


def FPR95Threshold(PosDist):
    PosDist = PosDist.sort(dim=-1, descending=False)[0]
    Val = PosDist[int(0.95 * PosDist.shape[0])]

    return Val


def ShowRowImages(image_data):
    fig = plt.figure(figsize=(1, image_data.shape[0]))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(1, image_data.shape[0]),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    # for ax, im in zip(grid, image_data):
    for ax, im in zip(grid, image_data):
        # Iterating over the grid returns the Axes.
        ax.imshow(im, cmap='gray')
    plt.show()


def ShowTwoRowImages(image_data1, image_data2):
    fig = plt.figure(figsize=(2, image_data1.shape[0]))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(2, image_data1.shape[0]),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    # for ax, im in zip(grid, image_data):
    for ax, im in zip(grid, image_data1):
        # Iterating over the grid returns the Axes.
        if im.shape[0] == 1:
            ax.imshow(im, cmap='gray')
        if im.shape[0] == 3:
            ax.imshow(im)

    for i in range(image_data2.shape[0]):
        # Iterating over the grid returns the Axes.
        if im.shape[0] == 1:
            grid[i + image_data1.shape[0]].imshow(image_data2[i], cmap='gray')
        if im.shape[0] == 3:
            grid[i + image_data1.shape[0]].imshow(image_data2[i])
    plt.show()


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

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
