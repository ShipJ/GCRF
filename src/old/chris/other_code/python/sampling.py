from __future__ import division

import pandas as pd
import numpy as np
import sys

import config

np.random.seed(1984)

def random_splits(ix, pr_train, n):
    size = np.round(ix.shape[0] * pr_train)
    if n == 1:
        np.random.shuffle(ix)
        return ix[:size], ix[size:]
    train_ix = np.zeros((n, size), dtype=np.int)
    test_ix = np.zeros((n, ix.shape[0]-size), dtype=np.int)
    for i in range(n):
        np.random.shuffle(ix)
        train_ix[i] = ix[:size]
        test_ix[i] = ix[size:]
    return train_ix, test_ix

if __name__ == "__main__":
    conf = config.get(sys.argv[1])

    dhs = pd.read_csv(conf['dhs_fn'])
    for p in range(5, 100, 5):
        train_ix, test_ix = random_splits(dhs.clust_id.values, p/100, 1000)
        np.save(conf['train_ix_fn']('dhs', p), train_ix)
        np.save(conf['test_ix_fn']('dhs', p), test_ix)
