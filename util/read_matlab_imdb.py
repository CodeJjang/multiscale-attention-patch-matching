import scipy.io as sio
import h5py
import numpy as np

def read_matlab_imdb(fname):
    f = h5py.File(fname, 'r')

    VarNames = list(f.keys())

    if len(VarNames) == 1:
        imdb = f[VarNames[0]];
        Result = np.squeeze(np.array(imdb[()]))
    else:
        i=0
        Result = dict()  # initiate empty dictionary
        for v in VarNames:
            Result[v] =  np.array(f[VarNames[i]])
            i=i+1

    f.close()

    return Result





