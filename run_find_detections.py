import torch
import numpy as np
import cv2
from torch import device
import matplotlib.pyplot as plt
from networks.MetricLearningCNN import MetricLearningCNN
from utils import NormalizeImages
import warnings

warnings.filterwarnings("ignore", message="UserWarning: albumentations.augmentations.transforms.RandomResizedCrop")
from multiprocessing import Process, freeze_support


def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type

    # return boxes[pick].astype("int")
    return pick


def non_max_suppression_points(Points, Radius, overlapThresh):
    Boxes = np.zeros((Points.shape[0], 4))

    Boxes[:, 0:2] = Points[:, 0:2] - Radius  # x1,y1
    Boxes[:, 2:4] = Points[:, 0:2] + Radius  # x2,y2

    idx = non_max_suppression_fast(Boxes, overlapThresh)

    # Result = Boxes[:,0:2]+Radius

    return np.asarray(idx)


def Detect_Features_Points(net, Img, NumberPoints, Radius, UseNonMaxima, Visualization=False):
    # Recorder saving inputs to all submodules
    recorder = torchfunc.hooks.recorders.ForwardPre()

    recorder.modules(net)

    # Push example image through network
    with torch.no_grad():
        Embed = net(Img)

    # plt.figure()
    # plt.imshow(np.squeeze(a0))

    # inverse loop on layers
    for k in range(len(recorder.data) - 1, 0, -1):

        Data = np.squeeze(recorder.data[k][0].cpu().numpy())
        # print(k-1,net[k - 1],Data.shape)

        if k == (len(recorder.data) - 1):
            Energy = np.abs(Data).sum(0)

            EnergyIdx = np.flip(np.argsort(Energy.ravel()))
            idx = np.unravel_index(EnergyIdx, Energy.shape)
            FeaturePoints = np.squeeze(np.dstack(idx))
            # Energy[FeaturePoints[0, 0], FeaturePoints[0, 1]]

            # FeaturePoints = FeaturePoints[0:int(FeaturePoints.shape[0]/2),:]
            # EnergyIdx     = EnergyIdx[    0:int(EnergyIdx.shape[0]/2)]

            FeaturePoints = FeaturePoints[0:NumberPoints, :]
            EnergyIdx = EnergyIdx[0:NumberPoints]

            if UseNonMaxima:
                overlapThresh = 0
                max_suppression_idx = non_max_suppression_points(FeaturePoints, Radius, overlapThresh)
                FeaturePoints = FeaturePoints[max_suppression_idx, :]
                EnergyIdx = EnergyIdx[max_suppression_idx]

            # sort detection energy again
            idx = np.flip(np.argsort(Energy.ravel()[EnergyIdx]))
            # FeaturePoints = FeaturePoints[idx, :]
            # FeaturePoints = FeaturePoints[0:NumberPoints, :]

            if Visualization:
                Pts = FeaturePoints.astype(np.float)
                Pts[:, 0] *= Img.shape[2] / Data.shape[1]  # y
                Pts[:, 1] *= Img.shape[3] / Data.shape[2]  # x
                plt.plot(Pts[:, 1], Pts[:, 0], 'o', color='black');  # plot(x, y)
                plt.show()

            continue

        if isinstance(net[k - 1], nn.Conv2d):
            Pad = net[k - 1].padding
            Dilation = net[k - 1].dilation
            Stride = net[k - 1].stride
            KernelSize = net[k - 1].kernel_size

            if (Stride[0] == 1) & (Dilation[0] == 1) & (Pad[0] == int(KernelSize[0] / 2)) & (
                    Pad[1] == int(KernelSize[0] / 2)):
                continue

            if Dilation[0] > 1:
                FeaturePoints += int((KernelSize[0] - 1) / 2) * (Dilation[0] - 1)

            if Stride[0] > 1:
                for m in range(0, FeaturePoints.shape[0]):
                    StartX = FeaturePoints[m, 1] * Stride[0]
                    EndX = StartX + Stride[0]

                    StartY = FeaturePoints[m, 0] * Stride[1]
                    EndY = StartY + Stride[1]

                    Pool = np.abs(Data[:, StartY:EndY, StartX:EndX]).max(0)
                    MaxIdx = np.unravel_index(Pool.argmax(), Pool.shape)

                    FeaturePoints[m, 0] = StartY + MaxIdx[1]
                    FeaturePoints[m, 1] = StartX + MaxIdx[0]

                if Visualization:
                    plt.plot(FeaturePoints[:, 1], FeaturePoints[:, 0], 'o', color='black');  # plot(x, y)
                    plt.show()

    return FeaturePoints


device: device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NumGpus = torch.cuda.device_count()

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)
name = torch.cuda.get_device_name(0)

ModelsDirName = './best models/'
ModelsDirName = './models/'
BestFileName = 'visnir_best_hybrid.pth'
# TestDir = '/home/keller/Dropbox/multisensor/python/data/test/'
# TestDir = 'F:\\multisensor\\test\\'
TestDir = './data/images/'
# TestDir = 'F:\\multisensor\\test\\'  # error 1.1
# TrainFile = '/home/keller/Dropbox/multisensor/python/data/Vis-Nir_Train.mat'
TestDecimation = 1

CnnMode = 'PairwiseAsymmetric'
CnnMode = 'PairwiseSymmetric'
CnnMode = 'Hybrid'

# Load all datasets


RgbImgName = '0001_rgb.tiff'
NirImgName = '0001_nir.tiff'
NumberPoints = 500

RgbIm = np.array(cv2.imread(TestDir + RgbImgName))
NirIm = np.array(cv2.imread(TestDir + NirImgName))

RgbIm = RgbIm.mean(2).astype(np.float32)
NirIm = NirIm.mean(2).astype(np.float32)

RgbIm = np.reshape(RgbIm, (1, 1, RgbIm.shape[0], RgbIm.shape[1]), order='F').astype(np.float32)
NirIm = np.reshape(NirIm, (1, 1, NirIm.shape[0], NirIm.shape[1]), order='F').astype(np.float32)

# -------------------------    loading previous results   ------------------------
checkpoint = torch.load(ModelsDirName + 'visnir_sym_triplet40.pth')
net = MetricLearningCNN(checkpoint['Mode'])
net.to(device)
net.load_state_dict(checkpoint['state_dict'])
net.to(device)

net = net.eval()

i = 0

a0 = RgbIm
b0 = NirIm

a = a0 - a0.mean()
b = b0 - b0.mean()
a = torch.from_numpy(NormalizeImages(a));
b = torch.from_numpy(NormalizeImages(b));

a, b = a.to(device), b.to(device)

netS = net.netS.block
netAS1 = net.netAS1.block
netAS2 = net.netAS2.block

net1 = netAS1
net2 = netAS2
NumberPoints = 5000
Radius = 5
UseNonMaxima = True
Visualization = False

Img = a
FeaturePointsA = Detect_Features_Points(net1, Img, NumberPoints, Radius, UseNonMaxima, Visualization)
# plt.plot(FeaturePointsA[:,1],FeaturePointsA[:,0], 'o', color='black');# plot(x, y)

Img = b
FeaturePointsB = Detect_Features_Points(net2, Img, NumberPoints, Radius, UseNonMaxima, Visualization)

f, axarr = plt.subplots(1, 2)
axarr[0].imshow(np.squeeze(a0))
axarr[0].plot(FeaturePointsA[:, 1], FeaturePointsA[:, 0], 'o', color='red');  # plot(x, y)
axarr[1].imshow(np.squeeze(b0))
axarr[1].plot(FeaturePointsB[:, 1], FeaturePointsB[:, 0], 'o', color='red');  # plot(x, y)
plt.show()

aa = 6
