"""
This module computes an adjacency matrix of activity given the
list of time-stamped files generated from process_raw_cdr.py

It uses COO (coordinate format) matrix operations/functions
which is considerably more efficient than simple matrix ops

It computes an adjacency matrix for all data, but computes a
similar matrix using only 'working hours' data.
"""

import pandas as pd
import numpy as np
import os
import scipy.sparse as sparse
from src.config import config


def hour_select(all_hours, to_examine):
    """
    Use either 'all' of the data, or only data within a certain time-period, working hours for example.
    :param all_hours: the list of all time-stamped filenames
    :param to_examine: should give parameter 'all' or 'working'
    :return:
    """
    if to_examine == 'all':
        hours_to_examine = all_hours

    elif to_examine == 'working':
        hours_to_examine = []
        for f in all_hours:
            third_dash = f.rfind('-')
            a, b = int(list(f)[third_dash+1]), int(list(f)[third_dash+2])
            if (a == 0 and b in range(6, 10)) or (a == 1 and b in range(10)) or (a == 2 and b in range(2)):
                hours_to_examine.append(f)
    else:
        hours_to_examine = []
    return hours_to_examine


def adj_matrix(source, country, hours_of_day):
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

    all_hours = [i for i in os.listdir(source) if not i.startswith('.')]

    hours_to_examine = hour_select(all_hours, hours_of_day)

    for f in hours_to_examine:
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
    PATH = config.get_dir()

    source = PATH+'/interim/%s/cdr/timestamp/' % country
    target = PATH+'/processed/%s/cdr/adjacency' % country

    try:
        # All hours in the data set
        vol_all, dur_all = adj_matrix(source, country, hours_of_day='all')
        np.savetxt(target+'/adj_matrix_vol_all.csv', vol_all, delimiter=',')
        np.savetxt(target+'/adj_matrix_dur_all.csv', dur_all, delimiter=',')

        # Working/daylight hours only: 06:00 to 21:00
        vol_working, dur_working = adj_matrix(source, country, hours_of_day='working')
        np.savetxt(target + '/adj_matrix_vol_working.csv', vol_working, delimiter=',')
        np.savetxt(target + '/adj_matrix_dur_working.csv', dur_working, delimiter=',')

    except ValueError:
        print 'Looks like your data is missing from interim/%s/cdr/timestamp.' % country
