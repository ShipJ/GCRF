from __future__ import division
import numpy as np
import pandas as pd
import os

from scipy.spatial.distance import pdist, cdist, squareform
from scipy.spatial import Delaunay
from sklearn.neighbors import KDTree # slightly faster than scipy.spatial.KDTree
from sklearn.cross_validation import KFold


from spatial_funcs import *
from features import *


# 1. aggregate 12 months of flows
# ------------------------------
# in these matrices the index is original id - 1, so index 0 corresponds to
# bts id 1 in the original data
vol = np.load('../data/sen/cdr_data/flows/total_vol.npy')
dur = np.load('../data/sen/cdr_data/flows/total_dur.npy')
sms = np.load('../data/sen/cdr_data/flows/total_sms.npy')
bts_fn = '../data/sen/cdr_data/bts_xzero.csv'
if os.path.exists(bts_fn):
    bts_points = points_from_csv(bts_fn, x_col=1, y_col=2)
    bts_ids = bts_points.keys()
    bts_coords = [(p.x, p.y) for p in bts_points.values()]
else:
    bts_points = points_from_csv('../data/sen/cdr_data/ContextData/bts_points.csv', x_col=4, y_col=5)
    # remove zero-flow towers from flow matrices
    zero_flow = zero_flows(vol) + 1 # plus 1 to match bts ids
    bts_ids = np.setdiff1d(bts_points.keys(), zero_flow)
    bts_points = {bts_id:bts_points[bts_id] for bts_id in bts_ids}
    bts_coords = [(p.x, p.y) for p in bts_points.values()]
    bts = pd.DataFrame(bts_coords, columns=['x','y'])
    bts['id'] = bts_ids
    bts[['id','x','y']].to_csv(bts_fn, index=False, encoding='utf8')
    del bts

n_bts = len(bts_coords)
ix = bts_ids - 1
vol = vol[ix][:,ix]
dur = dur[ix][:,ix]
sms = sms[ix][:,ix]
del ix

# load population raster data
rast_x = get_raster_data('../data/sen/geo/sen_pop/SEN14adjv1_utm28.tif')

# 2. Compute features at BTS level
# ---------------------------------
bts_features = None
fn = '../data/sen/model_data/bts_12_months.csv'
if os.path.exists(fn):
    bts_features = pd.read_csv(fn)
else:
    # compute gravity flows
    # create array of grid points and population at those points
    grid_points_x = get_grid_points(rast_x)
    grid_pops_x = np.array([get_point_value(rast_x, p) for p in grid_points_x])
    # remove points no-data points (i.e. where population is zero)
    grid_points_x = grid_points_x[grid_pops_x>=0]
    grid_pops_x = grid_pops_x[grid_pops_x>=0]

    tree = KDTree(grid_points_x, leaf_size=10000)
    # for each BTS find number of neighbours (grid points) within 20km
    ks = np.array([tree.query_radius(p, r=20000, count_only=True)[0] for p in bts_coords])
    # find distance to those neighbours
    results = [tree.query(p, k=k) for p,k in zip(bts_coords, ks)]
    nbrs = [ix[0] for ds, ix in results]
    ds = [ds[0] for ds, ix in results]
    # compute weights as inverse distance
    ps = np.array([1 / d for d in ds])
    ps = np.array([p / p.sum() for p in ps])
    # compute BTS population as weighted average of neighbouring grid points
    bts_pop = {}
    for i in range(len(nbrs)):
        bts_pop[bts_ids[i]] = (grid_pops_x[nbrs[i]] * ps[i]).sum()

    # compute gravity flows with population as mass
    G_pop = gravity_flows(bts_points, bts_pop)
    max_g_pop = G_pop[G_pop<np.inf].max()
    # compute gravity flows with total volumn as mass
    vol_sums = vol.sum(axis=1)
    vol_sums = {bts_ids[i]: vol_sums[i] for i in range(len(vol_sums))}
    G_vol = gravity_flows(bts_points, vol_sums)
    max_g_vol = G_vol[G_vol<np.inf].max()
    # write observed and estimated flows in csv file
    D = squareform(pdist(np.array(bts_coords), 'euclidean'))
    vol_undir = undirected_flows(vol, False)
    max_vol = vol_undir.max()
    with open('../data/sen/cdr_data/flows/total_vol.csv', 'w') as f:
        f.write('a,b,dist,vol,g_pop,g_vol\n')
        for i in range(vol.shape[0]):
            for j in range(i+1, vol.shape[1]):
                f.write('%d,%d,%f,%f,%f,%f\n' %
                    (bts_ids[i], bts_ids[j], D[i,j], vol_undir[i,j]/max_vol,
                        G_pop[i,j]/max_g_pop, G_vol[i,j]/max_g_vol))


    bts_features = pd.DataFrame({
        'bts_id': bts_ids,
        'population': [bts_pop[bts_id] for bts_id in bts_ids]})

    vol_features = compute_features(vol)
    vol_features.columns = ['vol_%s'%col for col in vol_features.columns]
    vol_res_features = compute_residual_features(vol, G_pop, G_vol)
    dur_features = compute_features(dur)
    dur_features.columns = ['dur_%s'%col for col in dur_features.columns]
    sms_features = compute_features(sms)
    sms_features.columns = ['sms_%s'%col for col in sms_features.columns]
    bts_features = pd.concat([bts_features, vol_features, vol_res_features, dur_features, sms_features], axis=1)

    bts_features['vol_dur_ratio'] = bts_features.dur_total / bts_features.vol_total
    bts_features['in_vol_dur_ratio'] = bts_features.dur_incoming / bts_features.vol_incoming
    bts_features['out_vol_dur_ratio'] = bts_features.dur_outgoing / bts_features.vol_outgoing
    bts_features['sms_call_ratio'] = bts_features.sms_total / bts_features.vol_total
    bts_features['in_sms_call_ratio'] = bts_features.sms_incoming / bts_features.vol_incoming
    bts_features['out_sms_call_ratio'] = bts_features.sms_outgoing / bts_features.vol_outgoing

    bts_features.to_csv(fn, index=False, encoding='utf8')


# 3. For each grid size...
# ------------------------
# create various size grid arrays
grid_sizes = [500, 1000, 2000, 5000, 10000, 20000, 50000]
n_grids = len(grid_sizes)
pop_rasts = [[]] * n_grids

for i, s in enumerate(grid_sizes):
    rast_fn = '../data/sen/geo/sen_pop/SEN14adjv1_utm28_%d.tif' % s
    if not os.path.exists(rast_fn):
        # transform original raster file to new grid size
        os.system('gdalwarp -tr %d %d -r average ../data/sen/geo/sen_pop/SEN14adjv1_utm28.tif %s' % (s,s,rast_fn))
    rast, gt = get_raster_data(rast_fn)
    # population aggregation is average so scale up accordingly
    scale = rast_x[0][rast_x[0]>0].sum() / rast[rast>0].sum()
    rast[rast>0] *= scale
    pop_rasts[i] = rast, gt

grids = [raster_to_polygons(rast) for rast in pop_rasts]
# create indices for grids
grid_ixs = [get_index(grid[0]) for grid in grids]

grid_points = [[]] * n_grids
grid_pops = [[]] * n_grids
for i in range(n_grids):
    grid_points[i] = get_grid_points(pop_rasts[i])
    grid_pops[i] = np.array([get_point_value(pop_rasts[i], p) for p in grid_points[i]])
    grid_points[i] = grid_points[i][grid_pops[i]>=0]
    grid_pops[i] = grid_pops[i][grid_pops[i]>=0]

# load DHS cluster data
fn = '../data/sen/cluster_wealth_cells.csv'
if os.path.exists(fn):
    dhs = pd.read_csv(fn)
    cluster_coords = dhs[['long','lat']].values
else:
    dhs = pd.read_csv('../data/sen/cluster_wealth.csv')
    # compute zscore of median wealth index
    dhs['z_median'] = (dhs['median'] - dhs['median'].mean()) / dhs['median'].std()
    cluster_coords = dhs[['long','lat']].values

    # assign clusters to grid cell
    for i in range(n_grids):
        tree = KDTree(grid_points[i], leaf_size=10000)
        results = [tree.query(p, k=1) for p in cluster_coords]
        nbrs = [ix[0][0] for ds, ix in results]
        dhs['cell_%d' % grid_sizes[i]] = nbrs
    dhs.to_csv(fn, index=False, encoding='utf8')

# (dis)aggregate BTS features to grid cells
r = 20000.
# feature_grids = [[]] * n_grids
# tree = KDTree(bts_coords, leaf_size=100)
feature_cols = bts_features.columns.tolist()[2:]
for i in range(n_grids-1,-1,-1):
    n_points = len(grids[i][0])
    fn = '../data/sen/model_data/cell_%d_12_months.csv' % grid_sizes[i]
    # if os.path.exists(fn):
    #     feature_grids[i] = pd.read_csv(fn)
    #     continue
    print "aggregating to grid %d x %d" % (grid_sizes[i], grid_sizes[i])
    print "finding neighbours..."
    nbrs, ds = point_within_distance(grids[i][0], bts_points.values(), r, grid_ixs[i])
    ws = np.array([w / w.sum() for w in 1 / (ds+1)])

    # # for each grid point find number of neighbours (BTS) within 20km
    # ks = np.array([tree.query_radius(p, r=20000, count_only=True)[0] for p in grid_points[i]])
    # # find distance to those neighbours
    # results = [tree.query(p, k=k) if k>0 else (np.array([[-1]]),np.array([[-1]]))
    #                     for p,k in zip(grid_points[i], ks)]
    # nbrs = [ix[0] for ds, ix in results]
    # ds = [ds[0] for ds, ix in results]
    # # compute weights as inverse distance
    # ps = np.array([1 / d for d in ds])
    # ps = np.array([p / p.sum() for p in ps])
    # # create weights matrix
    print "computing weights matrix..."
    W = np.zeros((n_points, n_bts), dtype=np.float)
    for j in range(n_points):
        W[j, nbrs[j]] = ws[j]
    # create cell features dataframe
    cell_features = pd.DataFrame({
        'id': range(n_points),
        'x': grid_points[i][:,0],
        'y': grid_points[i][:,1],
        'population': grid_pops[i]})
    print "computing cell features..."
    for col in feature_cols:
        # compute cell features as weighted average of neighbouring BTS towers
        cell_features[col] = np.apply_along_axis(lambda x: (x * bts_features[col]).sum(), 1, W)
    # save data
    print "saving data..."
    cell_features.to_csv(fn, index=False, encoding='utf8')
    # feature_grids[i] = cell_features
print "fin."


# -----------------------------
# Aggregate nearby bts towers
# -----------------------------
members = point_in_polygon(grids[i][0], bts_points.values(), grid_ixs[i])
members.groupby('cell_id').count()


# --------------------------------
# Compute lagged response variable
# --------------------------------
# n_train, n_test = 90, 10
# np.random.seed(1984)
# folds = KFold(dhs.shape[0], 10, shuffle=True)
# with open('../data/civ/model_data/folds_%d-%d_test_ix.txt' %
#     (n_train, n_test), 'wb') as f:
#     for _, test_ix in folds:
#         for i in test_ix:
#             f.write('%d ' % i)
#         f.write('\n')


# # radius based lags 90/10
# radii = np.array([10, 20, 50, 100, 200]) * 1000
# for i in range(n_grids-1,-1,-1):
#     print "computing lagged response for grid size %d" % grid_sizes[i]
#     gpoints = grid_points[i]
#     n_points = gpoints.shape[0]
#     grid_lags = [[]] * len(folds)
#     fold = 0
#     for train_ix, _ in folds:
#         print "\tfold %d" % fold
#         train_coords = dhs.iloc[train_ix][['long','lat']].values
#         train_y = dhs.iloc[train_ix]['z_median'].values
#         lags = pd.DataFrame({'id': range(n_points)})

#         tree = KDTree(train_coords, leaf_size=1000)
#         D = cdist(train_coords, gpoints)
#         for r in radii:
#             print "\t\tradius: %d" % r
#             nbrs = [tree.query_radius(p, r=r)[0] for p in gpoints]
#             ds = np.array([D[nbrs[j],j] for j in range(n_points)])
#             # compute weights as inverse distance
#             ws = np.array([w / w.sum() for w in 1/ds])
#             lags['d_%d'%r] = np.array([sum(ws[j] * train_y[nbrs[j]]) for j in range(n_points)])

    #         # compute weights as inverse squared distance
    #         ws = np.array([w / w.sum() for w in 1/ds**2])
    #         lags['d2_%d'%r] = np.array([sum(ws[j] * train_y[nbrs[j]]) for j in range(n_points)])

    #     lags['fold'] = fold
    #     grid_lags[fold] = lags
    #     fold += 1

    # print "\tsaving data..."
    # grid_lags = pd.concat(grid_lags)
    # grid_lags.to_csv('../data/sen/model_data/cells_%d_radius_lags_%d-%d.csv' %
    #     (grid_sizes[i], n_train, n_test), index=False, encoding='utf8')

# # delaunay triangulation based lags
# for i in range(n_grids-1,-1,-1):
#     print "computing lagged response for grid size %d" % grid_sizes[i]
#     gpoints = grid_points[i]
#     n_points = gpoints.shape[0]
#     grid_lags = [[]] * len(folds)
#     fold = 0
#     for train_ix, _ in folds:
#         print "\tfold %d" % fold
#         train_coords = dhs.iloc[train_ix][['long','lat']].values
#         train_y = dhs.iloc[train_ix]['z_median'].values
#         k = len(train_coords)
#         D = cdist(train_coords, gpoints)
#         lags = pd.DataFrame({
#             'id': range(n_points),
#             'grid_size': [grid_sizes[i]]*n_points,
#             'fold': [fold]*n_points
#         })
#         lag_d = np.zeros(n_points)
#         lag_d2 = np.zeros(n_points)
#         for j in range(n_points):
#             # no way to remove points (or copy original) so have to
#             # recompute each time, otherwise grid points will become
#             # neighbours of each other
#             tri = Delaunay(train_coords, incremental=True)
#             tri.add_points([gpoints[j]])
#             ix, ixptr = tri.vertex_neighbor_vertices
#             nbrs = ixptr[ix[k]:ix[k+1]]
#             if len(nbrs) == 0:
#                 lag_d[j] = lag_d2[j] = np.nan
#                 continue
#             ds = D[nbrs,j]
#             # compute weights as inverse distance
#             ws = 1/ds
#             lag_d[j] = sum(ws * train_y[nbrs]) / sum(ws)
#             # compute weights as inverse squared distance
#             ws = 1/ds**2
#             lag_d2[j] = sum(ws * train_y[nbrs]) / sum(ws)

#         lags['lag_d'] = lag_d
#         lags['lag_d2'] = lag_d2
#         grid_lags[fold] = lags
#         fold += 1

#     print "\tsaving data..."
#     grid_lags = pd.concat(grid_lags)
#     grid_lags.to_csv('../data/sen/model_data/cells_%d_delaunay_%d-%d_lags.csv' %
#         (grid_sizes[i], n_train, n_test), index=False, encoding='utf8')


# 50/50
n_train, n_test = 50, 50
np.random.seed(1984)
folds = KFold(dhs.shape[0], 2, shuffle=True)
with open('../data/civ/model_data/folds_%d-%d_test_ix.txt' %
    (n_train, n_test), 'wb') as f:
    for _, test_ix in folds:
        for i in test_ix:
            f.write('%d ' % i)
        f.write('\n')

i = 0
print "computing lagged response for grid size %d" % grid_sizes[i]
gpoints = grid_points[i]
n_points = gpoints.shape[0]
grid_lags = pd.DataFrame({'cell_id':range(n_points)})
fold = 0
for train_ix, _ in folds:
    print "\tfold %d" % fold
    train_coords = dhs.iloc[train_ix][['long','lat']].values
    train_y = dhs.iloc[train_ix]['z_median'].values

    W = 1/(cdist(train_coords, gpoints)**2)
    W = W/W.sum(axis=0)
    lagged = (W.T * train_y).sum(axis=1)

    grid_lags['fold_%d'%fold] = lagged
    fold += 1

print "\tsaving data..."
grid_lags.to_csv('../data/sen/model_data/cells_%d_d2_lags_%d-%d.csv' %
    (grid_sizes[i], n_train, n_test), index=False, encoding='utf8')


# 10/90
n_train, n_test = 10, 90
np.random.seed(1985)
folds = KFold(dhs.shape[0], 10, shuffle=True)
with open('../data/sen/model_data/folds_%d-%d_test_ix.txt' %
    (n_train, n_test), 'wb') as f:
    for test_ix, _ in folds:
        for i in test_ix:
            f.write('%d ' % i)
        f.write('\n')

i = 0
print "computing lagged response for grid size %d" % grid_sizes[i]
gpoints = grid_points[i]
n_points = gpoints.shape[0]
grid_lags = pd.DataFrame({'cell_id':range(n_points)})
fold = 0
for _, train_ix in folds:
    print "\tfold %d" % fold
    train_coords = dhs.iloc[train_ix][['long','lat']].values
    train_y = dhs.iloc[train_ix]['z_median'].values

    W = 1/(cdist(train_coords, gpoints)**2)
    W = W/W.sum(axis=0)
    lagged = (W.T * train_y).sum(axis=1)

    grid_lags['fold_%d'%fold] = lagged
    fold += 1

print "\tsaving data..."
grid_lags.to_csv('../data/sen/model_data/cells_%d_d2_lags_%d-%d.csv' %
    (grid_sizes[i], n_train, n_test), index=False, encoding='utf8')

# 5/95
n_train, n_test = 5, 95
np.random.seed(1985)
folds = KFold(dhs.shape[0], 20, shuffle=True)
with open('../data/sen/model_data/folds_%d-%d_test_ix.txt' %
    (n_train, n_test), 'wb') as f:
    for test_ix, _ in folds:
        for i in test_ix:
            f.write('%d ' % i)
        f.write('\n')

i = 0
print "computing lagged response for grid size %d" % grid_sizes[i]
gpoints = grid_points[i]
n_points = gpoints.shape[0]
grid_lags = pd.DataFrame({'cell_id':range(n_points)})
fold = 0
for _, train_ix in folds:
    print "\tfold %d" % fold
    train_coords = dhs.iloc[train_ix][['long','lat']].values
    train_y = dhs.iloc[train_ix]['z_median'].values

    W = 1/(cdist(train_coords, gpoints)**2)
    W = W/W.sum(axis=0)
    lagged = (W.T * train_y).sum(axis=1)

    grid_lags['fold_%d'%fold] = lagged
    fold += 1

print "\tsaving data..."
grid_lags.to_csv('../data/sen/model_data/cells_%d_d2_lags_%d-%d.csv' %
    (grid_sizes[i], n_train, n_test), index=False, encoding='utf8')


# 90/10
n_train, n_test = 90, 10
np.random.seed(1985)
folds = KFold(dhs.shape[0], 10, shuffle=True)
with open('../data/sen/model_data/folds_%d-%d_test_ix.txt' %
    (n_train, n_test), 'wb') as f:
    for _, test_ix in folds:
        for i in test_ix:
            f.write('%d ' % i)
        f.write('\n')

i = 0
print "computing lagged response for grid size %d" % grid_sizes[i]
gpoints = grid_points[i]
n_points = gpoints.shape[0]
grid_lags = pd.DataFrame({'cell_id':range(n_points)})
fold = 0
for train_ix, _ in folds:
    print "\tfold %d" % fold
    train_coords = dhs.iloc[train_ix][['long','lat']].values
    train_y = dhs.iloc[train_ix]['z_median'].values

    W = 1/(cdist(train_coords, gpoints)**2)
    W = W/W.sum(axis=0)
    lagged = (W.T * train_y).sum(axis=1)

    grid_lags['fold_%d'%fold] = lagged
    fold += 1

print "\tsaving data..."
grid_lags.to_csv('../data/sen/model_data/cells_%d_d2_lags_%d-%d.csv' %
    (grid_sizes[i], n_train, n_test), index=False, encoding='utf8')


# # randomly offset points
# def randomly_displace(points, max_dist):
#     dist = np.random.uniform(0., max_dist)
#     angle = np.radians(np.random.uniform(0., 360., points.shape[0]))
#     offset = np.array([[d*np.cos(a), d*np.sin(a)] for d, a in zip(dist, angle)])
#     return points + offset


# max_dist = np.array([2000. if u=='Urban' else 5000. for u in df['type']])
# points = df[['x','y']].values
# displaced = randomly_displace(points, max_dist)






#
