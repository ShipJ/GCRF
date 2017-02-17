import pandas as pd
import numpy as np
import os
import scipy.sparse as sparse
from src.config import config


def adj_matrix(source, target, country):
    """
    From time-stamped data files, compute adjacency matrix: ~

    :param source: string - file path to raw data.
    :param target: string - file path to save processed data.
    :param country: str - country code.

    :return: None.
    """

    constants = pd.DataFrame(pd.read_csv('../../data/processed/%s/constants.txt' % country))
    num_towers = constants['num_towers'].iloc[0]

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

    # Sparse to dense matrix (i.e. adjacency matrix)
    np.savetxt(target+'/adj_matrix_vol.csv', coo_vol.todense().reshape(num_towers, num_towers), delimiter=',')
    np.savetxt(target+'/adj_matrix_dur.csv', coo_dur.todense().reshape(num_towers, num_towers), delimiter=',')


if __name__ == '__main__':
    country = config.get_country()
    source = '../../data/interim/%s/cdr/timestamp/' % country
    target = '../../data/processed/%s/cdr/StaticMetrics' % country
    adj_matrix(source, target, country)
