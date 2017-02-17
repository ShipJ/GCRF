from __future__ import division
import os, sys
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.neighbors import KDTree
from scipy.spatial.distance import pdist, cdist, squareform

# from spatial_funcs import *
from config import config

def radiation_flux(m, n, s, T=1):
    '''
    m, n : the mass at pair of locations i and j
    s    : the total mass in the circle of radius r_ij
           centred at i, where r_ij is the distance between
           locations i and j
    T    : a scaling factor, i.e., T_i = m_i * (N_c / N),
           where N_c is the total number of commuters/callers/etc
           and N is the total population in the country.
           Leaving T = 1 returns the estimated fraction of i's
           total flow that goes to j.
    returns : the estimated flux between locations i and j
            according to the radiation model
    '''
    return T * m * n / ((m + s) * (m + n + s))




def get_grid(fn):
    grid_points = pd.read_csv(fn)
    return grid_points['population'].values, grid_points[['x','y']].values


def get_point_population(points, grid_fn, metric='euclidean'):
    grid_pop, grid_points = get_grid(grid_fn)
    if metric == 'sqeuclidean':
        D = cdist(points, grid_points, metric=metric)
        D[D>4e8] = 0
    else:
        D = cdist(points, grid_points, metric=metric)
        D[D>2e4] = 0
    W = (1/D)
    del D
    W[np.isinf(W)] = 0
    W = W / W.sum(axis=0)
    W[np.isnan(W)] = 0
    W[np.isinf(W)] = 0
    return (grid_pop * W).sum(axis=1)


def radiation_matrix(points, tree, D, pop):
    n_points = points.shape[0]
    S = np.zeros_like(D)
    F = np.zeros_like(D)
    for i in np.arange(n_points):
        if (i+1) % 100 == 0:
            print "%d of %d" % (i+1, n_points)
        p = points[i]
        m = pop[i]
        for j in np.arange(n_points):
            if i == j:
                F[i,j] = 0.
                continue
            n = pop[j]
            nbrs = tree.query_radius(p, r=D[i, j])[0]
            S[i,j] = pop[nbrs].sum() - m
            if j in nbrs:
                S[i,j] -= n
            F[i,j] = radiation_flux(m, n, S[i,j])
    return F, S


# def save_data(fn, ids, pop, pop2, D, S, S2, F, F2):
#     n = ids.shape[0]
#     f = open(fn, 'wb')
#     f.write('a,b,m1,n1,m2,n2,d,s1,s2,r1,r2\n')
#     for i in np.arange(n):
#         for j in np.arange(n):
#             f.write('%d,%d,%g,%g,%g,%g,%g,%g,%g,%g,%g\n' % (
#                 ids[i], ids[j], pop[i], pop[j], pop2[i], pop2[j],
#                 D[i,j], S[i,j], S2[i,j], F[i,j], F2[i,j])
#             )
#     f.close()

def save_data(fn, ids, pop, D, S, F):
    n = ids.shape[0]
    f = open(fn, 'wb')
    f.write('a,b,m,n,d,s,r\n')
    for i in np.arange(n):
        for j in np.arange(n):
            f.write('%d,%d,%g,%g,%g,%g,%g\n' % (
                ids[i], ids[j], pop[i], pop[j],
                D[i,j], S[i,j], F[i,j])
            )
    f.close()

def run(points_df, out_fn):
    points = points_df[['x','y']].values
    print "computing point population..."
    pop = get_point_population(points, conf['sqr_grid_fn'](500), 'sqeuclidean')

    print "creating tree..."
    tree = KDTree(points, leaf_size=100)
    D = squareform(pdist(points))

    print "estimating flows..."
    F, S = radiation_matrix(points, tree, D, pop)

    # print "computing point pop..."
    # point_pop2 = get_point_population(points, grid_fn, 'sqeuclidean')
    # print "estimating flows..."
    # F2, S2 = radiation_matrix(points, tree, D, point_pop2)

    print "saving data..."
    # save_data(out_fn, points_df['id'].values, pop, point_pop2, D, S, S2, F, F2)
    save_data(out_fn, points_df['id'].values, pop, D, S, F)


def get_total_external_vol(fn):
    vol = np.load(fn)
    return vol.sum() - vol.diagonal().sum()

if __name__ == '__main__':
    loc_code = sys.argv[1]
    clust_type = sys.argv[2]
    size = int(sys.argv[3])
    conf = config(loc_code)

    print "all points..."
    points = pd.read_csv(conf['centroids_fn'](clust_type, size))
    run(points, conf['rad_fn'](clust_type, size))

    print "\nnoncapital points..."
    points = pd.read_csv(conf['noncapital_centroids_fn'](clust_type, size))
    run(points, conf['noncapital_rad_fn'](clust_type, size))

    print "\ncapital points..."
    points = pd.read_csv(conf['capital_centroids_fn'](clust_type, size))
    run(points, conf['capital_rad_fn'](clust_type, size))

    print "fin."









#
