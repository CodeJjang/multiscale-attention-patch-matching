import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored

from network.my_classes import EvaluateDualNets,FPR95Threshold

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

        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)

        losses = F.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()



class PairwiseLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self):
        super(PairwiseLoss, self).__init__()

        self.Mode = 'FPR'

    def forward(self, Embed1, Embed2):

        if (Embed1.nelement()==0) | (Embed2.nelement()==0):
            return 0

        losses = (Embed1 - Embed2).pow(2).sum(1).pow(.5)

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

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)






def HardTrainingLoss(net, pos1, pos2,PosRatio,MarginRatio,T,device):

    net.eval()
    with torch.no_grad():
        emb1, emb2 = EvaluateDualNets(net, pos1, pos2, device, StepSize=400)

        # Dist  = sim_matrix(emb1,emb2).cpu().detach()
        Similarity = torch.mm(emb1, emb2.transpose(0, 1)).cpu()
        Similarity -= 1000000000 * torch.eye(n =Similarity.shape[0],m= Similarity.shape[1])
        NegIdx = torch.argmax(Similarity, axis=1).detach().cpu()


        #compute distances
        ap_distances = (emb1 - emb2).pow(2).sum(1)  # .pow(.5)
        an_distances = (emb1 - emb2[NegIdx, :]).pow(2).sum(1)

        # get LARGEST positive distances
        PosIdx = ap_distances.argsort(dim=-1, descending=True).cpu()
        PosIdx = PosIdx[0:int(PosRatio * PosIdx.shape[0])]

        #update NegIdx
        NegIdx = NegIdx[PosIdx]

        margin = ap_distances[PosIdx] - an_distances[PosIdx]

        # hard examples first
        Idx = margin.argsort(dim=-1, descending=True)

        # retain a subset of hard examples
        Idx = Idx[0:int(MarginRatio * Idx.shape[0])]

        PosIdx = PosIdx[Idx]
        NegIdx = NegIdx[Idx]

        AdditionalNegIdx = np.setdiff1d(NegIdx,PosIdx)
        NegIdx = np.concatenate((PosIdx,AdditionalNegIdx),0)

        del emb1, emb2

        pos1 = pos1[PosIdx]
        pos2 = pos2[NegIdx]



    #compute margin
    net.train()
    pos1, pos2 = pos1.to(device), pos2.to(device)
    Embed = net(pos1, pos2)

    if net.module.Mode == 'Hybrid':
        emb1 =  Embed['Hybrid1']
        emb2 =  Embed['Hybrid2']

        del Embed['EmbSym1'],Embed['EmbSym2'],Embed['EmbAsym1'],Embed['EmbAsym2']


    # Dist  = sim_matrix(emb1,emb2).cpu().detach()
    with torch.no_grad():
        Similarity = torch.mm(emb1, emb2.transpose(0, 1)).cpu()
        Similarity -= 1000000000 * torch.eye(n=Similarity.shape[0], m=Similarity.shape[1])
        NegIdx = torch.argmax(Similarity, axis=1).detach().cpu()

    ap_distances = (emb1 - emb2[0:emb1.shape[0], :]).pow(2).sum(1)  # .pow(.5)
    an_distances = (emb1 - emb2[NegIdx, :]).pow(2).sum(1)

    margin = ap_distances - an_distances

    margin = margin[torch.where(margin + T > 0)[0]]
    losses = (margin + T).mean()

    return losses






class OnlineHardNegativeMiningTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin,Mode,MarginRatio=1,PosRatio=1,NegPow=1,PosPow=1,device=None):
        super(OnlineHardNegativeMiningTripletLoss, self).__init__()
        self.margin    = margin
        self.Mode      = Mode
        self.MarginRatio = MarginRatio
        self.PosRatio  = PosRatio
        self.PosPow    = PosPow
        self.NegPow    = NegPow
        self.device    = device


    def forward(self, emb1, emb2):


        if self.Mode == 'Random':
            NegIdx = torch.randint(high=emb1.shape[0], size=(emb1.shape[0],),device=self.device)
            ap_distances = (emb1 - emb2).pow(2).sum(1)  # .pow(.5)
            an_distances = (emb1 - emb2[NegIdx, :]).pow(2).sum(1)  # .pow(.5)
            margin = ap_distances - an_distances


        if (self.Mode == 'Hardest')  | (self.Mode == 'HardPos'):
            #Dist  = sim_matrix(emb1,emb2).cpu().detach()
            Similarity = torch.mm(emb1, emb2.transpose(0, 1))
            Similarity -= 1000000000*torch.eye(n=Similarity.shape[0],m=Similarity.shape[1],device=self.device)
            NegIdx = torch.argmax(Similarity, axis=1) #find negative with highest similarity




        if (self.Mode == 'Hardest') :

            #ap_distances = (emb1 - emb2[0:emb1.shape[0],:]).pow(2).sum(1)  # .pow(.5)
            ap_distances = (emb1 - emb2).pow(2).sum(1)  # .pow(.5)
            an_distances = (emb1 - emb2[NegIdx,:]).pow(2).sum(1)

            margin = ap_distances - an_distances




        if (self.Mode == 'HardPos'):

            #ap_distances = (emb1 - emb2[0:emb1.shape[0],:,]).pow(2).sum(1)  # .pow(.5)
            ap_distances = (emb1 - emb2).pow(2).sum(1)  # .pow(.5)
            an_distances = (emb1 - emb2[NegIdx, :]).pow(2).sum(1)

            #get LARGEST positive distances
            PosIdx = ap_distances.argsort(dim=-1, descending=True)#sort positive distances
            PosIdx = PosIdx[0:int(self.PosRatio * PosIdx.shape[0])]#retain only self.PosRatio of the positives 

            NegIdx=NegIdx[PosIdx]

            margin = ap_distances[PosIdx] - an_distances[PosIdx]

            # hard examples first: sort margin
            Idx = margin.argsort(dim=-1, descending=True)

            # retain a subset of hard examples
            Idx = Idx[0:int(self.MarginRatio * Idx.shape[0])]#retain some of the examples

            margin = margin[Idx]

        losses = F.relu(margin + self.margin)
        idx = torch.where(losses>0)[0]

        if idx.size()[0]>0:
            losses = losses[idx].mean()

            if torch.isnan(losses):
                print('Found nan in loss ')
        else:
            losses = 0# losses.sum()
            #print(colored('\n No margin samples', 'magenta', attrs=['reverse', 'blink']))

        #return losses, idx.size()[0]/margin.shape[0]
        return losses





class InnerProduct(nn.Module):

    def __init__(self):
        super(InnerProduct, self).__init__()

    def forward(self, emb1, emb2):

        Loss = (emb1*emb2).abs().sum(1)
        return Loss.mean()









def FindFprTrainingSet(emb1, emb2,FprValPos,FprValNeg,MaxImgNo=0):

    with torch.no_grad():

        # Dist  = sim_matrix(emb1,emb2).cpu().detach()
        Similarity = torch.mm(emb1, emb2.transpose(0, 1))
        Similarity -= 1000000000 * torch.eye(n =Similarity.shape[0],m= Similarity.shape[1])
        NegIdx = torch.argmax(Similarity, axis=1)


        #compute DISTANCES
        ap_distances = (emb1 - emb2).pow(2).sum(1)  # .pow(.5)
        an_distances = (emb1 - emb2[NegIdx, :]).pow(2).sum(1)

        FPR95 = FPR95Threshold(ap_distances)
        #FprValPos = FPR95
        #FprValNeg = FPR95

        #plt.hist(ap_distances, 10);
        #plt.hist(an_distances, 10);

        # get positive distances ABOVE fpr
        PosIdx = torch.squeeze(torch.where(ap_distances > FprValPos)[0])
        #plt.hist(ap_distances, 10);
        #plt.hist(an_distances, 10);

        #sort array: LARGEST distances first
        PosIdx = PosIdx[ap_distances[PosIdx].argsort(dim=-1, descending=True)]

        # get negative distances BELOW fpr
        NegIdx1 = torch.squeeze(torch.where(an_distances < FprValNeg)[0])

        if (NegIdx1.nelement() > 1):
            NegIdx1 = NegIdx1[an_distances[NegIdx1].argsort(dim=-1, descending=False)]

        NegIdx2 = NegIdx[NegIdx1]


        Result = dict()
        Result['PosIdx']   = PosIdx
        Result['NegIdxA1'] = NegIdx1
        Result['NegIdxA2'] = NegIdx2



        NegIdx = torch.argmax(Similarity, axis=0)
        an_distances = (emb1[NegIdx, :] - emb2).pow(2).sum(1)

        NegIdx2 = torch.squeeze(torch.where(an_distances < FprValNeg)[0])
        if (NegIdx2.nelement() > 1):
            NegIdx2 = NegIdx2[an_distances[NegIdx2].argsort(dim=-1, descending=False)]
        NegIdx1 = NegIdx[NegIdx2]
        Result['NegIdxB1'] = NegIdx1
        Result['NegIdxB2'] = NegIdx2

    return Result












class FPRLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self,):
        super(FPRLoss, self).__init__()


    def forward(self, anchor, positive, negative):

        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)

        losses = distance_positive.mean() - distance_negative.mean()

        return losses




def Compute_FPR_HardNegatives(net, pos1, pos2, device,FprValPos,FprValNeg,MaxNoImages):

    with torch.no_grad():
        Embed = EvaluateDualNets(net, pos1, pos2, device, StepSize=400)

    # GPUtil.showUtilization()


    with torch.no_grad():
        Idx = FindFprTrainingSet( torch.from_numpy(Embed['Emb1']),  torch.from_numpy(Embed['Emb2']), FprValPos, FprValNeg)
    del Embed

    #print('PosIdx: ' + repr(Idx['PosIdx'].shape[0]) + ' NegIdxA1: ' + repr(
     #   Idx['NegIdxA1'].shape[0]) + ' NegIdxB1: ' + repr(Idx['NegIdxB1'].shape[0]))

    while (Idx['PosIdx'].nelement() + Idx['NegIdxA1'].nelement() + Idx['NegIdxB1'].nelement()) > MaxNoImages:
        # print('Memory error #1. #Images: ' + repr(Idx['PosIdx'].shape[0] + Idx['NegIdxA1'].shape[0] + Idx['NegIdxB1'].shape[0]))

        Idx['PosIdx'] = Idx['PosIdx'][0:int(Idx['PosIdx'].nelement() / 2)]

        if (Idx['NegIdxA1'].nelement() > 1):
            if (Idx['NegIdxA1'].shape[0] > 1):
                Idx['NegIdxA1'] = Idx['NegIdxA1'][0:int(Idx['NegIdxA1'].nelement() / 2)]
                Idx['NegIdxA2'] = Idx['NegIdxA2'][0:int(Idx['NegIdxA2'].nelement() / 2)]

        if (Idx['NegIdxB1'].nelement() > 1):
            if (Idx['NegIdxB1'].shape[0] > 1):
                Idx['NegIdxB1'] = Idx['NegIdxB1'][0:int(Idx['NegIdxB1'].nelement() / 2)]
                Idx['NegIdxB2'] = Idx['NegIdxB2'][0:int(Idx['NegIdxB2'].nelement() / 2)]

    Result = {}
    Result['PosIdx1'] = pos1[Idx['PosIdx']]
    Result['PosIdx2'] = pos2[Idx['PosIdx']]

    Result['NegIdxA1'] = pos1[Idx['NegIdxA1']]
    Result['NegIdxA2'] = pos2[Idx['NegIdxA2']]

    Result['NegIdxB1'] = pos1[Idx['NegIdxB1']]
    Result['NegIdxB2'] = pos2[Idx['NegIdxB2']]

    return Result








def ComputeFPR(emb1, emb2,FprValPos,FprValNeg):

    with torch.no_grad():

        # Dist  = sim_matrix(emb1,emb2).cpu().detach()
        Similarity = torch.mm(emb1, emb2.transpose(0, 1)).cpu()
        Similarity -= 1000000000 * torch.eye(n =Similarity.shape[0],m= Similarity.shape[1])
        NegIdx = torch.argmax(Similarity, axis=1)


        #compute DISTANCES
        ap_distances = (emb1 - emb2).pow(2).sum(1)  # .pow(.5)
        an_distances = (emb1 - emb2[NegIdx, :]).pow(2).sum(1)

        # get negative distances BELOW fpr
        NegIdx1 = torch.where(an_distances < FprValNeg)[0]

        Result = NegIdx1.shape[0]/an_distances.shape[0]

    return Result
