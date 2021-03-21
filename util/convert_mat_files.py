import numpy as np
import h5py

train_mat_file_path = 'D:\\multisensor\\datasets\\Vis-Nir_grid\\Vis-Nir_grid_Train.mat'
new_train_file_path = 'D:\\multisensor\\datasets\\Vis-Nir_grid\\train.hdf5'

test_mat_file_path = 'D:\\multisensor\\datasets\\Vis-Nir_grid\\Vis-Nir_grid_Test.mat'
new_test_file_path = 'D:\\multisensor\\datasets\\Vis-Nir_grid\\test.hdf5'

with h5py.File(train_mat_file_path, 'r') as f:
    train_data = np.array(f.get('images/data'))
    train_labels = np.logical_not(np.array(f.get('images/labels')) - 1).astype(np.float64)
    train_set = np.array(f.get('images/set'))
    train_data = train_data.transpose(0, 2, 3, 1).reshape(train_data.shape[0], 1, train_data.shape[2], train_data.shape[3], train_data.shape[1])
with h5py.File(new_train_file_path, 'w') as f:
    f.create_dataset('Data', data=train_data)
    f.create_dataset('Labels', data=train_labels)
    f.create_dataset('Set', data=train_set)

with h5py.File(test_mat_file_path, 'r') as f:
    test_data = np.array(f.get('testData'))
    test_labels = np.logical_not(np.array(f.get('testLabels')) - 1).astype(np.float64)
    test_data = test_data.transpose(0, 2, 3, 1).reshape(test_data.shape[0], 1, test_data.shape[2], test_data.shape[3], test_data.shape[1])
with h5py.File(new_test_file_path, 'w') as f:
    f.create_dataset('Data', data=test_data)
    f.create_dataset('Labels', data=test_labels)
