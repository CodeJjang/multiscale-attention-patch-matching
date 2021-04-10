import torch
import torch.nn as nn
import torch.nn.functional as F


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)

        losses = F.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


class PairwiseLoss(nn.Module):
    def __init__(self):
        super(PairwiseLoss, self).__init__()

        self.mode = 'FPR'

    @staticmethod
    def forward(pos1, pos2):
        if (pos1.nelement() == 0) | (pos2.nelement() == 0):
            return 0

        losses = (pos1 - pos2).pow(2).sum(1).pow(.5)

        return losses.mean()


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)


class OnlineHardNegativeMiningTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, mode, margin_ratio=1, pos_ratio=1, neg_pow=1, pos_pow=1, device=None):
        super(OnlineHardNegativeMiningTripletLoss, self).__init__()
        self.margin = margin
        self.mode = mode
        self.margin_ratio = margin_ratio
        self.pos_ratio = pos_ratio
        self.pos_pow = pos_pow
        self.neg_pow = neg_pow
        self.device = device

    def forward(self, emb1, emb2):

        if self.mode == 'Random':
            neg_idx = torch.randint(high=emb1.shape[0], size=(emb1.shape[0],), device=self.device)
            ap_distances = (emb1 - emb2).pow(2).sum(1)
            an_distances = (emb1 - emb2[neg_idx, :]).pow(2).sum(1)
            margin = ap_distances - an_distances

        if (self.mode == 'Hardest') | (self.mode == 'HardPos'):
            sim_matrix = torch.mm(emb1, emb2.transpose(0, 1))
            sim_matrix -= 1000000000 * torch.eye(n=sim_matrix.shape[0], m=sim_matrix.shape[1], device=self.device)
            neg_idx = torch.argmax(sim_matrix, axis=1)  # find negative with highest similarity

        if self.mode == 'Hardest':
            ap_distances = (emb1 - emb2).pow(2).sum(1)
            an_distances = (emb1 - emb2[neg_idx, :]).pow(2).sum(1)

            margin = ap_distances - an_distances

        if self.mode == 'HardPos':
            ap_distances = (emb1 - emb2).pow(2).sum(1)
            an_distances = (emb1 - emb2[neg_idx, :]).pow(2).sum(1)

            # get LARGEST positive distances
            pos_idx = ap_distances.argsort(dim=-1, descending=True)  # sort positive distances
            pos_idx = pos_idx[0:int(self.pos_ratio * pos_idx.shape[0])]  # retain only self.pos_ratio of the positives 

            margin = ap_distances[pos_idx] - an_distances[pos_idx]

            # hard examples first: sort margin
            idx = margin.argsort(dim=-1, descending=True)

            # retain a subset of hard examples
            idx = idx[0:int(self.margin_ratio * idx.shape[0])]  # retain some of the examples

            margin = margin[idx]

        losses = F.relu(margin + self.margin)
        idx = torch.where(losses > 0)[0]

        if idx.size()[0] > 0:
            losses = losses[idx].mean()

            if torch.isnan(losses):
                print('Found nan in loss ')
        else:
            losses = 0

        return losses


class InnerProduct(nn.Module):

    def __init__(self):
        super(InnerProduct, self).__init__()

    @staticmethod
    def forward(emb1, emb2):
        loss = (emb1 * emb2).abs().sum(1)
        return loss.mean()


def find_fpr_training_set(emb1, emb2, FprValPos, FprValNeg):
    with torch.no_grad():

        sim_matrix = torch.mm(emb1, emb2.transpose(0, 1))
        sim_matrix -= 1000000000 * torch.eye(n=sim_matrix.shape[0], m=sim_matrix.shape[1])
        neg_idx = torch.argmax(sim_matrix, axis=1)

        # compute DISTANCES
        ap_distances = (emb1 - emb2).pow(2).sum(1)
        an_distances = (emb1 - emb2[neg_idx, :]).pow(2).sum(1)

        # get positive distances ABOVE fpr
        pos_idx = torch.squeeze(torch.where(ap_distances > FprValPos)[0])

        # sort array: LARGEST distances first
        pos_idx = pos_idx[ap_distances[pos_idx].argsort(dim=-1, descending=True)]

        # get negative distances BELOW fpr
        neg_idx1 = torch.squeeze(torch.where(an_distances < FprValNeg)[0])

        if (neg_idx1.nelement() > 1):
            neg_idx1 = neg_idx1[an_distances[neg_idx1].argsort(dim=-1, descending=False)]

        neg_idx2 = neg_idx[neg_idx1]

        res = dict()
        res['pos_idx'] = pos_idx
        res['NegIdxA1'] = neg_idx1
        res['NegIdxA2'] = neg_idx2

        neg_idx = torch.argmax(sim_matrix, axis=0)
        an_distances = (emb1[neg_idx, :] - emb2).pow(2).sum(1)

        neg_idx2 = torch.squeeze(torch.where(an_distances < FprValNeg)[0])
        if neg_idx2.nelement() > 1:
            neg_idx2 = neg_idx2[an_distances[neg_idx2].argsort(dim=-1, descending=False)]
        neg_idx1 = neg_idx[neg_idx2]
        res['NegIdxB1'] = neg_idx1
        res['NegIdxB2'] = neg_idx2

    return res


class FPRLoss(nn.Module):

    def __init__(self, ):
        super(FPRLoss, self).__init__()

    @staticmethod
    def forward(anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)

        losses = distance_positive.mean() - distance_negative.mean()

        return losses
