from __future__ import division
import sys
import numpy as np
import pandas as pd
import networkx as nx
from shapely.geometry import Point
from spatial_funcs import (voronoi_polygons, geom_distances,
    read_shapefile, point_in_polygon)
import config

def to_undirected(edges):
    out_edges = edges.loc[np.where(edges.a < edges.b)].copy()
    in_edges = edges.loc[np.where(edges.a > edges.b)].copy()
    b = in_edges['b'].copy()
    in_edges['b'] = in_edges['a'].copy()
    in_edges['a'] = b

    out_edges.columns = ['a','b','out']
    in_edges.columns = ['a','b','in']

    out_edges.set_index(['a','b'], inplace=True)
    in_edges.set_index(['a','b'], drop=False, inplace=True)
    undir_edges = out_edges.join(in_edges)
    undir_edges['flow'] = undir_edges['out'] + undir_edges['in']
    return undir_edges[['a','b','flow']]

def to_total(flows):
    population = flows[['a','a_pop']].drop_duplicates().set_index('a')
    population.columns = ['population']
    def entropy(v):
        x = v.values
        p = (x / x.sum())[np.where(x > 0)]
        return -1 * sum(p * np.log(p)) / np.log(p.shape[0])

    def no_zero_median(x):
        return np.median(x[x>0])

    def no_zero_std(x):
        return np.std(x[x>0])
#   seems to get stuck if a index doesn't match b index
    out_grps = flows[['a','volume','gravity','radiation']].groupby('a')
    out_flows = out_grps.aggregate(
        [np.sum, np.median, np.std, entropy, no_zero_median, no_zero_std])
    out_flows.columns = [
        'vol_out_sum','vol_out_med','vol_out_std','vol_out_ent',
        'nz_vol_out_med','nz_vol_out_std',
        'grv_out_sum','grv_out_med','grv_out_std','grv_out_ent',
        'nz_grv_out_med','nz_grv_out_std',
        'rad_out_sum','rad_out_med','rad_out_std','rad_out_ent',
        'nz_rad_out_med','nz_rad_out_std'
    ]
    in_grps = flows[['b','volume','gravity','radiation']].groupby('b')
    in_flows = in_grps.aggregate(
        [np.sum, np.median, np.std, entropy, no_zero_median, no_zero_std])
    in_flows.columns = [
        'vol_in_sum','vol_in_med','vol_in_std','vol_in_ent',
        'nz_vol_in_med','nz_vol_in_std',
        'grv_in_sum','grv_in_med','grv_in_std','grv_in_ent',
        'nz_grv_in_med','nz_grv_in_std',
        'rad_in_sum','rad_in_med','rad_in_std','rad_in_ent',
        'nz_rad_in_med','nz_rad_in_std'
    ]
    df = population.join(out_flows.join(in_flows))
    df.fillna(1.) # na where entropy is computed on single edge
    return df

def graph_features(edges):
    undir_edges = to_undirected(edges)

    G = nx.DiGraph()
    edges = edges[edges.icol(2) > 0]
    G.add_weighted_edges_from(edges.values)
    pagerank = nx.pagerank_numpy(G, weight='weight')

    G = nx.Graph()
    undir_edges = undir_edges[undir_edges.icol(2) > 0]
    G.add_weighted_edges_from(undir_edges.values)
    evc = nx.eigenvector_centrality_numpy(G, weight='weight')

    return pagerank, evc

def compute_features(flows):
    int_flows = flows[flows['a'] == flows['b']][['a','volume']].copy().set_index('a')
    int_flows.columns = ['int_vol']
    ext_flows = flows[flows['a'] != flows['b']].copy()
    # sum, median, st dev, entropy
    df = to_total(ext_flows)
    # volume where a == b
    df = df.join(int_flows)
    # volume per person
    df['vol_norm'] = (df['vol_in_sum'] + df['vol_out_sum'] + df['int_vol']) / df['population']
    # internal volume per person
    df['int_vol_norm'] = df['int_vol'] / df['population']
    # ratio of internal volume to external volume
    df['introv'] = df['int_vol'] / (df['vol_out_sum'] + df['vol_in_sum'])


    # centrality metrics
    pr_vol, evc_vol = graph_features(ext_flows[['a','b','volume']].copy())
    df['vol_pagerank'] = pd.Series(pr_vol)
    df['vol_evc'] = pd.Series(evc_vol)
    pr_grv, evc_grv = graph_features(ext_flows[['a','b','gravity']].copy())
    df['grv_pagerank'] = pd.Series(pr_grv)
    df['grv_evc'] = pd.Series(evc_grv)
    pr_rad, evc_rad = graph_features(ext_flows[['a','b','radiation']].copy())
    df['rad_pagerank'] = pd.Series(pr_rad)
    df['rad_evc'] = pd.Series(evc_rad)

    # scaled difference between observed and expected summary statistics
    df['grv_out_sum_res'] = (df['vol_out_sum'] - df['grv_out_sum']) / df['vol_out_sum']
    df['grv_in_sum_res'] = (df['vol_in_sum'] - df['grv_in_sum']) / df['vol_in_sum']
    df['rad_out_sum_res'] = (df['vol_out_sum'] - df['rad_out_sum']) / df['vol_out_sum']
    df['rad_in_sum_res'] = (df['vol_in_sum'] - df['rad_in_sum']) / df['vol_in_sum']

    df['grv_in_std_res'] = (df['vol_in_std'] - df['grv_in_std']) / df['vol_in_std']
    df['grv_out_std_res'] = (df['vol_out_std'] - df['grv_out_std']) / df['vol_out_std']
    df['rad_in_std_res'] = (df['vol_in_std'] - df['rad_in_std']) / df['vol_in_std']
    df['rad_out_std_res'] = (df['vol_out_std'] - df['rad_out_std']) / df['vol_out_std']
    df['grv_in_ent_res'] = (df['vol_in_ent'] - df['grv_in_ent']) / df['vol_in_ent']
    df['grv_out_ent_res'] = (df['vol_out_ent'] - df['grv_out_ent']) / df['vol_out_ent']
    df['rad_in_ent_res'] = (df['vol_in_ent'] - df['rad_in_ent']) / df['vol_in_ent']
    df['rad_out_ent_res'] = (df['vol_out_ent'] - df['rad_out_ent']) / df['vol_out_ent']
    # difference between observed and expected centrality metrics
    df['grv_pagerank_res'] = df['vol_pagerank'] - df['grv_pagerank']
    df['grv_evc_res'] = df['vol_evc'] - df['grv_evc']
    df['rad_pagerank_res'] = df['vol_pagerank'] - df['rad_pagerank']
    df['rad_evc_res'] = df['vol_evc'] - df['rad_evc']

    ext_flows['grv_res'] = ext_flows['volume'] - ext_flows['gravity']
    ext_flows['rad_res'] = ext_flows['volume'] - ext_flows['radiation']

    def negative_mean(x):
        return x[x<0].mean()

    def positive_mean(x):
        return x[x>0].mean()

    def scaled_negative_mean(x):
        y = x / x.sum()
        return y[y<0].mean()

    def scaled_positive_mean(x):
        y = x / x.sum()
        return y[y>0].mean()

    # average difference between observed and expected pairwise flows
    # split between negative and positive so they don't cancel out,
    # and scaled versions
    out_resid = ext_flows[['a','grv_res','rad_res']].groupby('a').aggregate(
        [negative_mean, positive_mean, scaled_negative_mean, scaled_positive_mean])
    out_resid.columns = [
        'grv_mean_neg_out_res', 'grv_mean_pos_out_res',
        'grv_smean_neg_out_res', 'grv_smean_pos_out_res',
        'rad_mean_neg_out_res', 'rad_mean_pos_out_res',
        'rad_smean_neg_out_res', 'rad_smean_pos_out_res'
    ]

    in_resid = ext_flows[['b','grv_res','rad_res']].groupby('b').aggregate(
        [negative_mean, positive_mean, scaled_negative_mean, scaled_positive_mean])
    in_resid.columns = [
        'grv_mean_neg_in_res', 'grv_mean_pos_in_res',
        'grv_smean_neg_in_res', 'grv_smean_pos_in_res',
        'rad_mean_neg_in_res', 'rad_mean_pos_in_res',
        'rad_smean_neg_in_res', 'rad_smean_pos_in_res'
    ]
    return df.join(out_resid.join(in_resid))

def predict_point_features(feats, cols, feat_polys, dhs, dhs_points): #, radius):
    # nbrs, ds = point_within_distance(feat_polys, {i: Point(x, y) for i,x,y in dhs[['id','x','y']].values}, radius)
    # point_nbrs = {i:[] for i in dhs['id'].values}
    # point_ds = {i:[] for i in dhs['id'].values}
    # for i, nbs in nbrs.iteritems():
    #     for nb,d in zip(nbs, ds[i]):
    #         if not point_nbrs.get(nb):
    #             point_nbrs[nb] = [i]
    #             point_ds[nb] = [d]
    #         else:
    #             point_nbrs[nb].append(i)
    #             point_ds[nb].append(d)

    ds, poly_ix = geom_distances(dhs_points, feat_polys)
    # weights are ~ inverse distance squared
    ws = {i: 1 /(np.array(d)**2 + 1) for i,d in ds.iteritems()}
    ws = {i: w / w.sum() for i,w in ws.iteritems()}

    preds = dhs.copy()
    preds['mean_nbr_dist'] = pd.Series({i: d.mean() for i, d in ds.iteritems()})
    for r in [5, 10, 30, 60, 100]:
        preds['n_nbrs_%dkm'%r] = pd.Series({i: len(d[d<r*1000]) for i, d in ds.iteritems()})

    for col in cols:
        # preds[col] = pd.Series({i:(ws[i] * feats.loc[point_nbrs[i], col].values).sum() for i in point_nbrs.keys()})
        preds[col] = pd.Series({i:(w * feats.loc[poly_ix, col].values).sum() for i, w in ws.iteritems()})

    return preds


def run(flows, centroids, centroid_feats_fn, dhs_feats_fn, dhs, dhs_points, grid_polys):
    flows = flows[['a','b','d','m','n','vol','grav','rad']]
    flows.columns = ['a','b','distance','a_pop','b_pop','volume','gravity','radiation']

    print "computing features..."
    feats = compute_features(flows)
    # NaNs introduced when there are no +ve or no -ve residuals
    feats.fillna(0, inplace=True)
    feats = centroids.join(feats)
    feats.to_csv(centroid_feats_fn, encoding='utf8')

    print "estimating features at prediction points..."
    feat_polys = {i: grid_polys[i] for i in feats.index}
    pred_cols = feats.columns.tolist()[4:]
    pred_point_feats = predict_point_features(feats, pred_cols, feat_polys, dhs, dhs_points)
    pred_point_feats.to_csv(dhs_feats_fn, encoding='utf8')


if __name__ == '__main__':
    loc_tag = sys.argv[1]
    clust_type = sys.argv[2]
    size = int(sys.argv[3])
    conf = config.get(loc_tag)

    print "reading data..."
    dhs = pd.read_csv(conf['dhs_fn'])
    dhs.set_index('clust_id', inplace=True)
    dhs_points = {i: Point(dhs.ix[i, ['x','y']]) for i in dhs.index}
    print "converting grid to polygons..."
    grid_points = pd.read_csv(conf['hex_grid_fn'](size))
    grid_polys = voronoi_polygons({
        i: Point(grid_points.ix[i, ['x','y']]) for i in grid_points.index})

    print "\nall points..."
    run(pd.read_csv(conf['flows_fn'](clust_type,size)),
        pd.read_csv(conf['centroids_fn'](clust_type, size), index_col='id'),
        conf['centroid_feats_fn'](clust_type,size),
        conf['dhs_feats_fn'](clust_type,size),
        dhs,
        dhs_points,
        grid_polys
    )

    shp, _ = read_shapefile(conf['capital_geom_fn'])
    pip = point_in_polygon(shp, dhs_points)[0]
    dhs['capital'] = [i in pip for i in dhs.index]
    dhs.to_csv(conf['dhs_fn'], encoding='utf8')

    print "\nnoncapital points..."
    centroids = pd.read_csv(conf['noncapital_centroids_fn'](clust_type, size), index_col='id')
    centroids.drop(-1, axis=0, inplace=True)
    run(pd.read_csv(conf['noncapital_flows_fn'](clust_type,size)),
        centroids,
        conf['noncapital_centroid_feats_fn'](clust_type,size),
        conf['noncapital_dhs_feats_fn'](clust_type,size),
        dhs[dhs['capital']==False],
        {i: dhs_points[i] for i in dhs.index if i not in pip},
        grid_polys
    )

    print "\ncapital points..."
    centroids = pd.read_csv(conf['capital_centroids_fn'](clust_type, size), index_col='id')
    centroids.drop(-1, axis=0, inplace=True)
    run(pd.read_csv(conf['capital_flows_fn'](clust_type,size)),
        centroids,
        conf['capital_centroid_feats_fn'](clust_type,size),
        conf['capital_dhs_feats_fn'](clust_type,size),
        dhs[dhs['capital']],
        {i: dhs_points[i] for i in dhs.index if i in pip},
        grid_polys
    )

    print "fin."











#
