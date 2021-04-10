import glob
import os
from multiprocessing import freeze_support

import h5py
import numpy as np
import torch

from util import read_hdf5_data

if __name__ == '__main__':
    freeze_support()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    test_dir = './data/Vis-Nir_grid/'
    load_all_test_sets = True
    train_file = './data/brown/patchdata_64x64.h5'

    convert_train_files = True
    convert_test_files = True
    convert_patch_files = True

    if convert_patch_files:
        data = read_hdf5_data(train_file)
        data['liberty'] = np.reshape(data['liberty'], (data['liberty'].shape[0], 1, 64, 64), order='F')
        data['notredame'] = np.reshape(data['notredame'], (data['notredame'].shape[0], 1, 64, 64), order='F')
        data['yosemite'] = np.reshape(data['yosemite'], (data['yosemite'].shape[0], 1, 64, 64), order='F')

        with h5py.File('patchdata1' + '.h5', 'w') as f:
            f.create_dataset('liberty', data=data['liberty'])
            f.create_dataset('notredame', data=data['notredame'])
            f.create_dataset('yosemite', data=data['yosemite'])

    if convert_train_files:
        path, dataset_name = os.path.split(train_file)
        dataset_name = os.path.splitext(train_file)[0]

        data = read_hdf5_data(train_file)
        training_set_data = np.transpose(data['Data'], (0, 3, 2, 1))
        training_set_labels = np.squeeze(data['Labels'])
        training_set_splits = np.squeeze(data['Set'])

        training_set_data = np.reshape(training_set_data, (
        training_set_data.shape[0], 1, training_set_data.shape[1], training_set_data.shape[2],
        training_set_data.shape[3]), order='F')
        training_set_labels = 2 - training_set_labels

        with h5py.File(dataset_name + '.hdf5', 'w') as f:
            f.create_dataset('Data', data=training_set_data, compression='gzip', compression_opts=9)
            f.create_dataset('Labels', data=training_set_labels, compression='gzip', compression_opts=9)
            f.create_dataset('Set', data=training_set_splits, compression='gzip', compression_opts=9)

    if convert_test_files:

        # Load all datasets
        file_list = glob.glob(test_dir + "*.mat")

        if load_all_test_sets == False:
            file_list = [file_list[0]]

        file_list = ['./data/Vis-Nir_grid/Vis-Nir_grid_Test.mat']

        test_data = dict()
        for f in file_list:
            path, dataset_name = os.path.split(f)
            dataset_name = os.path.splitext(dataset_name)[0]

            print(f)
            data = read_hdf5_data(f)

            x = np.transpose(data['testData'], (0, 3, 2, 1))
            TestLabels = torch.from_numpy(2 - data['testLabels'])

            x = np.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3]), order='F')
            with h5py.File(path + '/' + dataset_name[:-5] + '.hdf5', 'w') as f:
                f.create_dataset('Data', data=x, compression='gzip', compression_opts=9)
                f.create_dataset('Labels', data=TestLabels, compression='gzip', compression_opts=9)
