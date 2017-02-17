# Compute the gravity residual metric for each data set
# -- this measures a weighted metric that takes into account the population of two given regions, and normalises
# that by teh squared distance between them.

# Input: CDR temporal data sets
# Output: Adjacency matrix containing the total volume of activity between pairs of cell towers

import pandas as pd
import numpy as np
import sys

if __name__ == "__main__":

    # country = sys.argv[1]
    country = 'IvoryCoast'
    path = '/Users/JackShipway/Desktop/UCLProject/Data/%s/CDR' % country

    # Set known data set values (number of towers, hourly duration of data)
    if country == 'Senegal':
        num_bts, hours = 1668, 8760
    elif country == 'IvoryCoast':
        num_bts, hours = 1240, 3360
    else:
        num_bts, hours = 10000, 100000


    ''' Population at each administrative level '''

    ''' Distance matrix between all adminstrative level centroids'''

    ''' Compute expected flows, and then compare to observed flows '''

    ''' Compute g-value by aligning the data with the observed flows - i.e. shift the data up/down so as to
    minimise the bias between them '''

    ''' compute the gravity residuals for all under-estimates, i.e. all negative values only '''