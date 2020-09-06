from torch.utils.data import Dataset
import numpy as np
import h5py
import torch
import os
import psutil
import pickle
from utils import NormalizeImages


class HDF5Dataset(Dataset):

    def __init__(self, dataset_name, path, h5_fpath):
        self.h5_fpath = h5_fpath
        self.dataset_name = dataset_name
        means_cache_name = os.path.join(path, f'{dataset_name}_mean_image.p')
        if os.path.isfile(means_cache_name):
            print(f'Loading dataset {dataset_name} means from cache.')
            self.means = pickle.load(open(means_cache_name, 'rb'))
        else:
            print(f'Dataset {dataset_name} has no means cache file, calculating dataset means...')
            self.means = self.get_mean_image(h5_fpath)
            pickle.dump(self.means, open(means_cache_name, 'wb'))

    def __getitem__(self, index):
        with h5py.File(self.h5_fpath, 'r') as h5_file:
            data = h5_file['Data']
            labels = h5_file['Labels']
            data = data[index]
            label = labels[index]

            data = data.astype(np.float32)
            label = torch.from_numpy(np.array(label, dtype=np.float32))

            data -= self.means

            data = NormalizeImages(data)
            data = torch.from_numpy(data)
            return data, label

    def __len__(self):
        with h5py.File(self.h5_fpath, 'r') as h5_file:
            data = h5_file['Data']
            return len(data)

    def get_mean_image(self, fpath):
        """
        Returns the mean_image of a xs dataset.
        :param str fpath: Filepath of the data upon which the mean_image should be calculated.
        :return: ndarray xs_mean: mean_image of the x dataset.
        """
        with h5py.File(fpath, "r") as h5_file:
            # last shape indicates whether it's a pair or a triplet dataset
            last_shape = h5_file['Data'].shape[-1]
            means = []

            # check available memory and divide the mean calculation in steps
            total_memory = 0.5 * psutil.virtual_memory().available  # In bytes. Take 1/2 of what is available, just to make sure.
            filesize = os.path.getsize(fpath)
            steps = int(np.ceil(filesize / total_memory))
            n_rows = h5_file['Data'].shape[0]
            stepsize = int(n_rows / float(steps))

            for pair_idx in range(last_shape):

                xs_mean_arr = None
                for i in range(steps):
                    if xs_mean_arr is None:  # create xs_mean_arr that stores intermediate mean_temp results
                        xs_mean_arr = np.zeros((steps,) + h5_file['Data'].shape[1:-1], dtype=np.float64)

                    if i == steps - 1:  # for the last step, calculate mean till the end of the file
                        data_slice = h5_file['Data'][i * stepsize: n_rows, :, :, :, pair_idx]
                        xs_mean_temp = np.mean(data_slice, axis=0, dtype=np.float64, keepdims=True)
                    else:
                        data_slice = h5_file['Data'][i * stepsize: (i + 1) * stepsize, :, :, :, pair_idx]
                        xs_mean_temp = np.mean(data_slice, axis=0, dtype=np.float64, keepdims=True)
                    xs_mean_arr[i] = xs_mean_temp

                xs_mean = np.mean(xs_mean_arr, axis=0, dtype=np.float64, keepdims=True).astype(np.float32)

                means.append(xs_mean)

            return np.array(means).reshape(xs_mean.shape + (last_shape,))
