from __future__ import division
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from spatial_funcs import *
import config


###############################################################################
# Aggregate flows

def aggregate_vol(bts, vol_fn, offset):
    vol = np.load(vol_fn)
    # merge pairs of points with same label
    for i in bts.index:
        lab_i = bts.ix[i, 'label']
        i += offset
        for j in bts.index:
            lab_j = bts.ix[j, 'label']
            j += offset
            if i < j and lab_i == lab_j:
                # merge rows
                vol[i] += vol[j]
                vol[j] = 0
                # merge columns
                vol[:,i] += vol[:,j]
                vol[:,j] = 0
    # then remove zero flows
    xzero = np.union1d(np.where(vol.sum(axis=0)>0)[0], np.where(vol.sum(axis=1)>0)[0])
    if loc_code == 'civ' and xzero[0] == 0:
        # remove no tower id -1
        xzero = xzero[1:]
    vol = vol[xzero][:,xzero]

    vol_ix = pd.Series(range(vol.shape[0]), index=bts.ix[xzero-offset]['label'].values)
    return vol, vol_ix




###############################################################################
# Compute clustering with MeanShift

# The following bandwidth can be automatically detected using
# bandwidth = estimate_bandwidth(points, quantile=0.05)
# bts = pd.read_csv('../data/%s/cdr_data/bts_xzero.csv' % loc_code)
# points = bts[['x','y']].values

# mean_shift = MeanShift(bandwidth=500, bin_seeding=True)
# mean_shift.fit(points)
# labels = mean_shift.labels_

# cluster_centers = mean_shift.cluster_centers_
# labels_unique = np.unique(labels)
# n_clusters_ = len(labels_unique)

# bts['label'] = labels
# bts.to_csv('../data/%s/cdr_data/bts_meanshift_labels.csv' % loc_code, index=False, encoding='utf8')
# clusters = pd.DataFrame({'id': labels_unique, 'x':cluster_centers[:,0], 'y':cluster_centers[:,1]})
# vol, clusters = aggregate_vol(clusters, bts)
# np.save('../data/%s/cdr_data/flows/meanshift_vol.npy' % loc_code, vol)
# clusters.to_csv('../data/%s/cdr_data/meanshift_clusters.csv' % loc_code, encoding='utf8')

##############################################################################
# Compute clustering with DBSCAN

# dbscan = DBSCAN(eps=500, min_samples=1).fit(points)
# core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
# core_samples_mask[dbscan.core_sample_indices_] = True
# labels = dbscan.labels_

# cluster_centers = dbscan.cluster_centers_ # !! DBSCAN doesn't have centres !!
# labels_unique = np.unique(labels)
# n_clusters_ = len(labels_unique)

# bts['label'] = labels
# bts.to_csv('../data/%s/cdr_data/bts_dbscan_labels.csv' % loc_code, index=False, encoding='utf8')
# clusters = pd.DataFrame({'id': labels_unique, 'x':cluster_centers[:,0], 'y':cluster_centers[:,1]})
# vol, clusters = aggregate_vol(clusters, bts)
# np.save('../data/%s/cdr_data/flows/dbscan_vol.npy' % loc_code, vol)
# clusters.to_csv('../data/%s/cdr_data/dbscan_clusters.csv' % loc_code, encoding='utf8')

###############################################################################
# Aggregate to 1km hex grid


if __name__ == "__main__":
    loc_code = sys.argv[1]
    clust_type = sys.argv[2]
    size = int(sys.argv[3])

    conf = config(loc_code)
    bts = pd.read_csv(conf['bts_xzero_fn'])
    bts.set_index('id', inplace=True)
    bts_points = {i: Point(bts.ix[i]) for i in bts.index}
    grid = points_from_csv(conf['hex_grid_fn'](size), id_col=None, x_col=0, y_col=1)

    print "converting hex grid points to polygons..."
    hexa = voronoi_polygons(grid)
    print "finding points in polygons..."
    pip = point_in_polygon(hexa, bts_points)
    pip_list = []
    for hex_id, members in pip.iteritems():
        for p in members:
            pip_list.append((hex_id, p))
    pip_list = np.array(pip_list, dtype=np.int)
    bts['label'] = pd.Series(pip_list[:,0], index=pip_list[:,1])

    print "finding points in capital region..."
    shp, _ = read_shapefile(conf['capital_geom_fn'])
    pip = point_in_polygon(shp, bts_points)
    bts['capital'] = [int(i in pip[0]) for i in bts.index]

    bts.to_csv(conf['bts_label_fn'](clust_type,size), encoding='utf8')
    centroids = bts[['label','x','y']].groupby('label').mean()
    centroids.index.names = ['id']

    print "computing aggregated volume matrix for all points..."
    vol, vol_ix = aggregate_vol(bts, conf['total_vol_fn'], conf['offset'])
    np.save(conf['vol_fn'](clust_type, size), vol)
    centroids['vol_ix'] = vol_ix
    centroids.to_csv(conf['centroids_fn'](clust_type,size), encoding='utf8')

    print "computing aggregated volume matrix for noncapital points..."
    bts['old_label'] = bts['label'].copy()
    bts.loc[bts['capital']==1,'label'] = -1
    centroids = bts[['label','x','y']].groupby('label').mean()
    centroids.index.names = ['id']

    vol, vol_ix = aggregate_vol(bts, conf['total_vol_fn'], conf['offset'])
    np.save(conf['noncapital_vol_fn'](clust_type,size), vol)
    centroids['vol_ix'] = vol_ix
    centroids.to_csv(conf['noncapital_centroids_fn'](clust_type,size), encoding='utf8')

    print "computing aggregated volume matrix for capital points..."
    bts['label'] = bts['old_label'].copy()
    bts.loc[bts['capital']==0,'label'] = -1
    centroids = bts[['label','x','y']].groupby('label').mean()
    centroids.index.names = ['id']

    vol, vol_ix = aggregate_vol(bts, conf['total_vol_fn'], conf['offset'])
    np.save(conf['capital_vol_fn'](clust_type,size), vol)
    centroids['vol_ix'] = vol_ix
    centroids.to_csv(conf['capital_centroids_fn'](clust_type,size), encoding='utf8')

    print "fin."





#
