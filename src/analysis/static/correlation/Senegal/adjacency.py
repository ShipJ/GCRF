import pandas as pd
import numpy as np
import math
import os.path

if __name__ == "__main__":

    path = '/Users/JackShipway/Desktop/UCLProject/Project1-Health/Data/Senegal/CDR/Temporal'
    m, w, d, h = 0, 0, 0, 0

    adj_matrix = np.zeros((1668, 1668))

    for hour in range(8760):
        file = '/Month_%s/Week_%s/Day_%s/Hour_%s/graph.csv' % (m, w, d, h)

        if os.path.isfile(path+file):
            print "Reading Hour: %s" % hour, (m, w, d, h)

            data = pd.read_csv((path + file),
                               usecols=["source", "target", "volume", "duration"]).as_matrix()

            active_towers = np.array(np.unique(np.concatenate([data[:, 0], data[:, 1]])))
            for cell_tower in active_towers:
                active_data = data[data[:, 0] == cell_tower]
                for i in range(len(active_data)):
                    adj_matrix[cell_tower, active_data[i, 1]] += active_data[i, 2]

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

    np.savetxt('adj_matrix_4.csv', adj_matrix, delimiter=',')
