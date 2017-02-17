import numpy as np
import pandas as pd
import networkx as nx
import os
import math
import time

# Start timer
t0 = time.time()

path = '/Users/JackShipway/Desktop/UCLProject/Data/Senegal/CDR/Data'
m, w, d, h = 0, 0, 0, 0

G = nx.Graph()
G.add_nodes_from(range(0, 1668))
adj_matrix_time = np.zeros((1668, 1668, 8760))

for hour in range(8760):

    file = '/Month_%s/Week_%s/Day_%s/Hour_%s/graph.csv' % (m, w, d, h)

    if os.path.isfile(path+file):
        print 'Reading Hour: %s' % hour, (m, w, d, h)

        data = pd.read_csv((path+file), usecols=['source', 'target', 'volume']).as_matrix()
        G.add_weighted_edges_from(data)
        adj_matrix_time[:, :, hour] = nx.to_numpy_matrix(G)

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

np.savetxt('adj_matrix_time', adj_matrix_time, delimiter=',')

# End timer
t1 = time.time()
# Time elapsed
print "\nRuntime: %s seconds, %s minutes, %s hours" % (t1-t0, (t1-t0)/60, (t1-t0)/3600)