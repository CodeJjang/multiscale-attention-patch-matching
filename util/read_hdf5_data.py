import h5py
import numpy as np


def read_hdf5_data(fname):
    with h5py.File(fname, 'r') as f:

        keys = list(f.keys())

        if len(keys) == 1:
            data = f[keys[0]]
            res = np.squeeze(np.array(data[()]))
        else:
            i = 0
            res = dict()
            for v in keys:
                res[v] = np.array(f[keys[i]])
                i += 1

    return res
