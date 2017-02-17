import numpy as np
import pandas as pd
import networkx as nx
import os
import math
import sys

# path = '/Users/JackShipway/Desktop/UCLProject/Data/Senegal/CDR/Data'
# m, w, d, h = 0, 0, 0, 0
#
# adj_matrix = np.zeros((1668, 1668))
# adj_matrix_time = []
#
# for hour in range(8760):
#     file = '/Month_%s/Week_%s/Day_%s/Hour_%s/graph.csv' % (m, w, d, h)
#
#     if os.path.isfile(path+file):
#         print 'Reading Hour: %s' % hour, (m, w, d, h)
#         data = pd.read_csv((path+file), usecols=['source', 'target', 'volume']).as_matrix()
#         G = nx.Graph()
#         G.add_nodes_from(range(1668))
#         G.add_weighted_edges_from(data)
#         del data
#
#         adj_t_i = nx.to_numpy_matrix(G)
#         G.clear()
#         adj_matrix += adj_t_i
#         # adj_matrix_time.append(adj_t_i)
#         del adj_t_i
#
#         # Increment CDR directory
#         day = 0
#         h = int(math.fmod(h + 1, 24))
#         if h == 0:
#             day += 1
#             d = int(math.fmod(d + 1, 7))
#             if d == 0:
#                 w = int(math.fmod(w + 1, 4))
#                 if w == 0:
#                     m += 1
#
# np.savetxt('adj_matrix.csv', adj_matrix, delimiter=',')
# # np.savetxt('adj_matrix_time.csv', adj_matrix_time, delimiter=',')

path = '/Users/JackShipway/Desktop/UCLProject/Data/IvoryCoast/CDR/Data'
m, w, d, h = 0, 0, 0, 0

list = range(1239)
list.append(-1)

adj_matrix = np.zeros((1240, 1240))

for hour in range(3360):
    file = '/Month_%s/Week_%s/Day_%s/Hour_%s/graph.csv' % (m, w, d, h)

    if os.path.isfile(path+file):
        print 'Reading Hour: %s' % hour, (m, w, d, h)
        data = pd.read_csv((path+file), usecols=['source', 'target', 'weight']).as_matrix()
        G = nx.Graph()
        G.add_nodes_from(list)
        G.add_weighted_edges_from(data)
        del data

        adj_t_i = nx.to_numpy_matrix(G)
        G.clear()
        adj_matrix += adj_t_i
        del adj_t_i

        # Increment CDR directory
        day = 0
        h = int(math.fmod(h + 1, 24))
        if h == 0:
            day += 1
            d = int(math.fmod(d + 1, 7))
            if d == 0:
                w = int(math.fmod(w + 1, 4))
                if w == 0:
                    m += 1

np.savetxt('adj_matrix_civ.csv', adj_matrix, delimiter=',')
