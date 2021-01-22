import numpy as np
import h5py

datasets = ['liberty', 'notredame', 'yosemite']
test_datasets_path = 'D:\\multisensor\\datasets\\brown\\patchdata\\evaluate_%s_64x64.h5'
train_datasets_path = 'D:\\multisensor\\datasets\\brown\\patchdata\\full_evaluate_%s_64x64.h5'
new_test_file_path = 'D:\\multisensor\\datasets\\brown\\patchdata\\%s_test_for_multisensor.hdf5'
new_train_file_path = 'D:\\multisensor\\datasets\\brown\\patchdata\\%s_full_for_multisensor.hdf5'
new_full_train_file_path = 'D:\\multisensor\\datasets\\brown\\patchdata\\full_for_multisensor.hdf5'


def transform_dimensions(arr):
    samples = arr.shape[0]
    arr = arr.reshape(samples, 1, 64, 64).reshape(int(samples / 2), 2, 64, 64).transpose(0, 2, 3, 1)
    arr = np.expand_dims(arr, 1)
    assert arr.shape == (int(samples / 2), 1, 64, 64, 2)
    return arr


def save_results(fpath, data, labels, set_labels):
    with h5py.File(fpath, 'w') as f:
        f.create_dataset('Data', data=data)
        f.create_dataset('Labels', data=labels)
        f.create_dataset('Set', data=set_labels)

def convert_single_ds():
    convert_train = True
    convert_test = False
    for dataset in datasets:
        if convert_test:
            with h5py.File(test_datasets_path % dataset, 'r') as f:
                pos = transform_dimensions(np.array(f.get('50000/match')))
                neg = transform_dimensions(np.array(f.get('50000/non-match')))
                data = np.concatenate((pos, neg))
                labels = np.concatenate((np.full(pos.shape[0], 1), np.full(neg.shape[0], 0)))
                set_labels = np.full(labels.shape, 1)
            save_results(new_test_file_path % dataset, data, labels, set_labels)
        if convert_train:
            with h5py.File(train_datasets_path % dataset, 'r') as f:
                pos = transform_dimensions(np.array(f.get('250000/match')))
                neg = transform_dimensions(np.array(f.get('250000/non-match')))
                data = np.concatenate((pos, neg))
                labels = np.concatenate((np.full(pos.shape[0], 1), np.full(neg.shape[0], 0)))
                set_labels = np.full(labels.shape, 1)
            save_results(new_train_file_path % dataset, data, labels, set_labels)

def convert_all_ds_train():
    ds_size = 500000
    with h5py.File(new_full_train_file_path, 'w') as f:
        data = f.create_dataset('Data', (ds_size * 3, 1, 64, 64, 2), maxshape=(None, 1, 64, 64, 2))
        labels = f.create_dataset('Labels', (ds_size * 3,), maxshape=(None,))
        set = f.create_dataset('Set', (ds_size * 3,), maxshape=(None,))
        for i, ds in enumerate(datasets):
            with h5py.File(new_train_file_path % ds, 'r') as dsf:
                data[i * ds_size: (i+1) * ds_size] = np.array(dsf.get('Data'), np.float32)
                labels[i * ds_size: (i + 1) * ds_size] = np.array(dsf.get('Labels'), np.float32)
                # set[i * ds_size: (i + 1) * ds_size] = dsf.get('Set')
def main():
    # convert_single_ds()
    convert_all_ds_train()

main()
