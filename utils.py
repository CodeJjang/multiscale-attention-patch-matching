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
