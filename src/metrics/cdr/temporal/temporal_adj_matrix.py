import pandas as pd
import numpy as np
import os
import scipy.sparse as sparse
from src.config import config
from src.metrics.cdr.static import cdr_metrics
import sys

def volatility(activity_time):
    return np.mean(activity_time, axis=0)


def adj_matrix(source, country):
    """
    From time-stamped data files, compute temporal adjacency matrix: runtime {3m, 15m}

    :param source: string - file path to raw data.
    :param country: str - country code.

    :return: None.
    """

    constants = config.get_constants(country)
    num_towers = constants['num_towers']
    num_hours = constants['num_hours']

    i = 0

    activity_time = np.zeros((num_towers, 6, num_hours))
    for f in os.listdir(source):
        if f not in ['data.txt', '.DS_Store']:
            print "Reading: %s" % f

            cdr = pd.read_csv((source+f), usecols=['source', 'target', 'activity', 'duration']).as_matrix()
            # -1: cdr where origin/target cell tower unknown
            missing = np.where(cdr == -1)
            # Convert to imaginary cell tower ID (this is not useful, but aids processing)
            cdr[missing] = num_towers-1

            coo_vol_i = sparse.coo_matrix((cdr[:, 2], (cdr[:, 0], cdr[:, 1])), shape=(num_towers, num_towers))
            coo_dur_i = sparse.coo_matrix((cdr[:, 3], (cdr[:, 0], cdr[:, 1])), shape=(num_towers, num_towers))

            coo_vol_i = coo_vol_i.todense().reshape(num_towers, num_towers)
            coo_dur_i = coo_dur_i.todense().reshape(num_towers, num_towers)

            activity = cdr_metrics.activity(num_towers, coo_vol_i, coo_dur_i)[['Vol', 'Vol_in', 'Vol_out',
                                                                              'Dur', 'Dur_in', 'Dur_out']]
            activity_time[:, :, i] = activity
            i += 1
    vola = volatility(activity_time)

    volatil = pd.DataFrame()
    volatil['CellTowerID'] = np.array(range(num_towers))
    volatil['Vol_volatil'] = vola[:, 0]
    volatil['Vol_in'] = vola[:, 1]
    volatil['Vol_out'] = vola[:, 2]
    volatil['Dur'] = vola[:, 3]
    volatil['Dur_in'] = vola[:, 4]
    volatil['Dur_out'] = vola[:, 5]

    return volatil



if __name__ == '__main__':
    country = config.get_country()
    source = '../../../../data/interim/%s/cdr/timestamp/' % country
    target = '../../../../data/processed/%s/cdr/temporalmetrics' % country

    volatility = adj_matrix(source, country)
    volatility.to_csv(target+'/volatility.csv', index=None)
