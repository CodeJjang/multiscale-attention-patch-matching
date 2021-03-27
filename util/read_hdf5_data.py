import h5py
import numpy as np

def read_hdf5_data(fname):
    f = h5py.File(fname, 'r')

    VarNames = list(f.keys())

    if len(VarNames) == 1:
        data = f[VarNames[0]]
        Result = np.squeeze(np.array(data[()]))
    else:
        i=0
        Result = dict()  # initiate empty dictionary
        for v in VarNames:
            Result[v] =  np.array(f[VarNames[i]])
            i=i+1

    f.close()

    return Result





