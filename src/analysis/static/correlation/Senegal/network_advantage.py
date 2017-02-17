import pandas as pd
import numpy as np
import math
import os.path

if __name__ == "__main__":

    # deg_matrix = pd.DataFrame(pd.read_csv()).as_matrix()


    ''' Normalised Entropy'''
    q_matrix = pd.DataFrame(pd.read_csv("q_matrix.csv")).as_matrix()
    print q_matrix


    # norm_entropy = np.zeros(1668)
    #
    # for index, row in q_matrix.iterrows():
    #
    #
    #     norm_entropy[i] = (-1*q_sum) / np.log(deg_matrix[i])
    #
    # # normalise by population - groupby adm_region, then divide through, will take some thought though




    # ''' Median Degree '''
    # # degree of each cell tower after removing all links below a certain threshold, which is the median in this case
    # med_matrix = pd.DataFrame(pd.read_csv()).as_matrix()
    # links_removed = deg_matrix - med_matrix
    # med_degree = np.zeros(1668)
    #
    # for i in range(len(links_removed)):
    #     deg_in = np.count_nonzero(links_removed[:, i])
    #     deg_out = np.count_nonzero(links_removed[i, :])
    #     deg_self = np.count_nonzero(links_removed[i, i])
    #
    #     med_degree[i] = (deg_in + deg_out) - deg_self
    #
    # # Normalise by population (same as above)



