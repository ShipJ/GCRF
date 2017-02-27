import pandas as pd
import numpy as np
import os
import scipy.sparse as sparse
from src.config import config


def adj_matrix(source, target, country):
    """
    From time-stamped data files, compute temporal adjacency matrix: runtime {3m, 15m}

    :param source: string - file path to raw data.
    :param target: string - file path to save processed data.
    :param country: str - country code.

    :return: None.
    """

    constants = config.get_constants(country)
    num_towers = constants['num_towers']
    hours = constants['hours']

    temporal_adj_vol = []
    temporal_adj_dur = []

    i = 0
    for f in os.listdir(source):
        print "Reading: %s" % f

        # A 'coo' matrix is a coordinate based sparse matrix (efficient/fast)
        coo_vol = sparse.coo_matrix((num_towers, num_towers))
        coo_dur = sparse.coo_matrix((num_towers, num_towers))

        cdr = pd.read_csv((source+f), usecols=['source', 'target', 'activity']).as_matrix()
        # -1: cdr where origin/target cell tower unknown
        missing = np.where(cdr == -1)
        # Convert to imaginary cell tower ID (this is not useful, but aids processing)
        cdr[missing] = num_towers-1
        temporal_adj_vol.append(sparse.coo_matrix((cdr[:, 2], (cdr[:, 0], cdr[:, 1])), shape=(num_towers, num_towers)))
        temporal_adj_dur.append(sparse.coo_matrix((cdr[:, 3], (cdr[:, 0], cdr[:, 1])), shape=(num_towers, num_towers)))

    print temporal_adj_vol
    print len(temporal_adj_vol)

    # # Sparse to dense matrix (i.e. adjacency matrix)
    # np.savetxt(target+'/adj_matrix_vol.csv', coo_vol.todense().reshape(num_towers, num_towers), delimiter=',')
    # np.savetxt(target+'/adj_matrix_dur.csv', coo_dur.todense().reshape(num_towers, num_towers), delimiter=',')


if __name__ == '__main__':
    country = config.get_country()
    source = '../../data/interim/%s/cdr/timestamp/' % country
    target = '../../data/processed/%s/cdr/staticmetrics' % country
    adj_matrix(source, target, country)
