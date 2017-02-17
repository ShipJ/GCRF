import pandas as pd
import numpy as np
import os
import scipy.sparse as sparse


def adj_matrix(source, target, country):
    """
    From time-stamped data files, compute adjacency matrix

    :param source: string - file path to raw data.
    :param target: string - file path to save processed data.
    :param country: str - country code.

    :return: None.
    """

    constants = pd.DataFrame(pd.read_csv('../../data/processed/%s/constants.txt' % country))
    num_towers = constants['num_towers'].iloc[0]

    adj_matrix_vol = np.zeros((num_towers, num_towers))
    adj_matrix_dur = np.zeros((num_towers, num_towers))

    for f in os.listdir(source):
        print "Reading: %s" % f
        cdr = pd.read_csv((source+f), usecols=['source', 'target', 'activity', 'duration']).as_matrix()
        # -1: cdr where origin/target cell tower unknown
        missing = np.where(cdr == -1)
        # Convert to imaginary cell tower ID (this is not useful, but aids processing)
        cdr[missing] = num_towers-1

        i, j, vol, dur = cdr[:, 0], cdr[:, 1], cdr[:, 2], cdr[:, 3]
        sparse_adj_vol = sparse.lil_matrix((num_towers, num_towers))
        sparse_adj_dur = sparse.lil_matrix((num_towers, num_towers))

        for i, j, v, d in zip(i, j, vol, dur):
            sparse_adj_vol[i, j] = v
            sparse_adj_dur[i, j] = d

        adj_matrix_vol += sparse_adj_vol.todense()
        adj_matrix_dur += sparse_adj_vol.todense()

    np.savetxt(target+'adj_matrix_vol.csv', adj_matrix_vol, delimiter=',')
    np.savetxt(target+'adj_matrix_dur.csv', adj_matrix_dur, delimiter=',')


def get_country():
    """
    Ask user for country code.

    :return: str - country for which there is data.
    """
    print "Process data for which country? ['sen': Senegal, 'civ': Ivory Coast]: "
    input_country = raw_input()
    if input_country == 'sen':
        country = 'sen'
    elif input_country == 'civ':
        country = 'civ'
    else:
        print "Please type the country abbreviation (lower case): "
        return get_country()
    return country


if __name__ == '__main__':
    country = get_country()
    source = '../../data/interim/%s/cdr/timestamp/' % country
    target = '../../data/processed/%s/cdr/StaticMetrics' % country
    adj_matrix(source, target, country)
