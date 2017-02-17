from __future__ import division

import sys
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

import config


loc_tag = sys.argv[1]
conf = config.get(loc_tag)
dhs = pd.read_csv(conf['dhs_fn'])
dhs.set_index('clust_id', inplace=True)
points = dhs[['x','y']]

for p in range(5, 100, 5):
    train_ix = np.load(conf['train_ix_fn']('dhs', p))
    test_ix = np.load(conf['test_ix_fn']('dhs', p))
    wealth_lags = np.zeros((len(train_ix), len(dhs)))
    poverty_lags = np.zeros((len(train_ix), len(dhs)))
    # class_lags = np.zeros((len(train_ix), len(dhs)))
    for i in range(train_ix.shape[0]):
        train_points = points.ix[train_ix[i]].values
        W = 1/(cdist(train_points, points.values)**2)
        W[np.isinf(W)] = 0
        W = W/W.sum(axis=0)

        train_y = dhs.ix[train_ix[i]]['z_median'].values
        wealth_lags[i] = (W.T * train_y).sum(axis=1)
        train_y = dhs.ix[train_ix[i]]['poverty_rate'].values
        poverty_lags[i] = (W.T * train_y).sum(axis=1)
        # train_y = (dhs.ix[train_ix[i]]['poverty_rate'].values >= .25).astype(np.int)
        # class_lags[i] = (W.T * train_y).sum(axis=1)

    np.save(conf['lags_fn']('dhs', 'wealth', p), wealth_lags)
    np.save(conf['lags_fn']('dhs', 'poverty', p), poverty_lags)
    # np.save(conf['lags_fn']('dhs', 'clf', p), class_lags)









#
