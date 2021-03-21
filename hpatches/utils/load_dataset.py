import numpy as np
import h5py
import os
from joblib import Parallel, delayed
import multiprocessing
from hpatches.utils.hpatch import hpatch_sequence
import torch
from network.generator import NormalizeImages


def load_dataset(dataset_path):
    t = [x[0] for x in os.walk(dataset_path)][1::]
    try:
        len(t) == 116
    except:
        print("%r does not seem like a valid HPatches descriptor root folder." % (dataset_path))
    # seqs = Parallel(n_jobs=multiprocessing.cpu_count()) \
    #     (delayed(hpatch_sequence)(f) for f in t)
    seqs = []
    for f in t:
        print('Loading %s' % f)
        seq = hpatch_sequence(f)
        for t in hpatch_sequence.itr:
            data = getattr(seq, t)
            data = data - np.mean(data, axis=0)
            data = NormalizeImages(data)
            data = torch.from_numpy(data)
            setattr(seq, t, data)
        seqs.append(seq)
    seqs = dict((l.name, l) for l in seqs)
    print('Image sequence files loaded.')
    return seqs
