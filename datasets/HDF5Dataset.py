from torch.utils.data import Dataset
import numpy as np
import h5py
import torch

from utils import NormalizeImages


class HDF5Dataset(Dataset):

    def __init__(self, h5_fpath):
        with h5py.File(h5_fpath, 'r') as h5_file:
            data = h5_file['Data']
            labels = h5_file['Labels']
            self.train = data
            self.labels = labels
            self.length = len(data)

    def __getitem__(self, index):
        data = self.train[str(index)]
        label = self.labels[str(index)]

        data = data.astype(np.float32)
        label = torch.from_numpy(np.squeeze(label))

        data[:, :, :, :, 0] -= data[:, :, :, :, 0].mean()
        data[:, :, :, :, 1] -= data[:, :, :, :, 1].mean()

        data = NormalizeImages(data)
        data = torch.from_numpy(data)
        return data, label

    def __len__(self):
        return self.length
