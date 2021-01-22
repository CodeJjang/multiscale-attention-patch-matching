import numpy as np
import h5py
import os
from joblib import Parallel, delayed
import multiprocessing
from hpatches.utils.load_dataset import load_dataset
from hpatches.utils.hpatch import hpatch_sequence
import pickle

test_datasets_path = 'D:\\multisensor\\datasets\\hpatches-benchmark\\data\\hpatches-release'
new_train_file_path = 'D:\\multisensor\\datasets\\hpatches-benchmark\\data\\hpatches-release-multisensor\\data_v2.hdf5'


def main():
    seqs = load_dataset(test_datasets_path)
    with h5py.File(new_train_file_path, 'w') as handle:
        for seq_name, seq_data in seqs.items():
            for t in hpatch_sequence.itr:
                data = getattr(seq_data, t)
                handle.create_dataset(f'{seq_name}/{t}', data=data)
        #pickle.dump(seqs, handle, protocol=pickle.HIGHEST_PROTOCOL)


main()
