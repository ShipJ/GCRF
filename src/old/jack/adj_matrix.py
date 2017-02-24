# Compute the adjacency matrix, degree vector, q_matrix and log(q_matrix) for data activity metrics

# Input: CDR temporal data sets
# Output: Adjacency matrix containing the total volume of activity between pairs of cell towers

# Takes ~ 2 hours for IC, ~ 4 hours for Sen

import numpy as np
import math
import os.path
import sys

if __name__ == "__main__":

    # country = sys.argv[1]
    country = 'Senegal'
    path = '/Users/JackShipway/Desktop/UCLProject/Data/%s/CDR' % country

    # Set known data set values (number of towers, hourly duration of data)
    if country == 'Senegal':
        num_bts, hours = 1669, 8760
    elif country == 'IvoryCoast':
        num_bts, hours = 1240, 3360
    else:
        num_bts, hours = 10000, 100000

    # # Set initial time-stamp
    # m, w, d, h = 0, 0, 0, 0
    # # Initialise adjacency matrix w. no. cell towers
    # adj_matrix = np.zeros((num_bts, num_bts))
    #
    # # Cycle through all files, update adjacency matrix
    # for hour in range(hours):
    #     f = path+'/Data/Month_%s/Week_%s/Day_%s/Hour_%s/graph.csv' % (m, w, d, h)
    #
    #     if os.path.isfile(f):
    #         print 'Reading Data Set: %s' % hour, '--- ' 'm:', m, 'w:', w, 'd:', d, 'h:', h
    #         cdr = np.genfromtxt(f, delimiter=',', usecols=(1, 2, 3), skiprows=1)
    #
    #         if cdr.size > 3:
    #             active_towers = np.array(np.unique(np.concatenate([cdr[:, 0], cdr[:, 1]])))
    #             for cell_tower in active_towers:
    #                 active_data = cdr[cdr[:, 0] == cell_tower]
    #                 for i in range(len(active_data)):
    #                     adj_matrix[cell_tower, active_data[i, 1]] += active_data[i, 2]
    #
    #         # Increment CDR directory
    #         h = int(math.fmod(h + 1, 24))
    #         if h == 0:
    #             d = int(math.fmod(d + 1, 7))
    #             if d == 0:
    #                 w = int(math.fmod(w + 1, 4))
    #                 if w == 0:
    #                     m += 1

    adj_matrix = np.genfromtxt(path+'/staticmetrics/Other/adj_matrix.csv')
    wherenan = np.where(np.isnan(adj_matrix))
    adj_matrix[wherenan] = 0
    total_activity = np.genfromtxt(path+'/staticmetrics/Activity/total_activity.csv', delimiter=',')
    q_matrix = np.array(adj_matrix / total_activity[:, 1, None])

    # Compute degree vector
    deg_vector = np.zeros(num_bts)
    for i in range(num_bts):
        out_deg = np.where(adj_matrix[i, :] != 0)
        in_deg = np.where(adj_matrix[:, i] != 0)
        self_deg = 0 if adj_matrix[i, i] == 0 else 1
        deg_vector[i] = len(np.union1d(out_deg[0], in_deg[0])) - self_deg

    # Compute log(q_matrix)
    where_nan = np.isnan(q_matrix)
    q_matrix[where_nan] = 0
    def f(x):
        return x * np.log10(x)
    f = np.vectorize(f)
    log_q_matrix = f(q_matrix)

    # np.savetxt(path+'/staticmetrics/Other/adj_matrix.csv', adj_matrix, delimiter=',')
    np.savetxt(path+'/staticmetrics/Other/degree_vector.csv', deg_vector, delimiter=',')
    np.savetxt(path+'/staticmetrics/Other/q_matrix.csv', q_matrix, delimiter=',')
    np.savetxt(path+'/staticmetrics/Other/log_q_matrix.csv', log_q_matrix, delimiter=',')
