"""
This module computes an adjacency matrix of activity given the
list of time-stamped files generated from process_raw_cdr.py

It uses COO (coordinate format) matrix operations/functionality
which is considerably more efficient than simple matrix ops

The 'save' functions are commented out to prevent overriding
"""

import pandas as pd
import numpy as np
import os
import scipy.sparse as sparse
from src.config import config


def adj_matrix(source, country):
    """
    From time-stamped data files, compute adjacency matrix: runtime {3m, 15m}

    :param source: string - file path to raw data.
    :param country: str - country code.

    :return: None.
    """

    constants = config.get_constants(country)
    num_towers = constants['num_towers']

    # A 'coo' matrix is a coordinate based sparse matrix (efficient/fast)
    coo_vol = sparse.coo_matrix((num_towers, num_towers))
    coo_dur = sparse.coo_matrix((num_towers, num_towers))

    for f in os.listdir(source):
        print "Reading: %s" % f
        cdr = pd.read_csv((source+f), usecols=['source', 'target', 'activity', 'duration']).as_matrix()
        # -1: cdr where origin/target cell tower unknown
        missing = np.where(cdr == -1)
        # Convert to imaginary cell tower ID (this is not useful, but aids processing)
        cdr[missing] = num_towers-1
        coo_vol += sparse.coo_matrix((cdr[:, 2], (cdr[:, 0], cdr[:, 1])), shape=(num_towers, num_towers))
        coo_dur += sparse.coo_matrix((cdr[:, 3], (cdr[:, 0], cdr[:, 1])), shape=(num_towers, num_towers))

    # Sparse to dense matrix (i.e. full adjacency matrix)
    coo_vol = coo_vol.todense().reshape(num_towers, num_towers)
    coo_dur = coo_dur.todense().reshape(num_towers, num_towers)

    return coo_vol, coo_dur


if __name__ == '__main__':
    country = config.get_country()
    source = '../../../../data/interim/%s/cdr/timestamp/' % country
    target = '../../../../data/processed/%s/cdr/staticmetrics' % country

    vol, dur = adj_matrix(source, country)

    # np.savetxt(target+'/adj_matrix_vol.csv', vol, delimiter=',')
    # np.savetxt(target+'/adj_matrix_dur.csv', dur, delimiter=',')
