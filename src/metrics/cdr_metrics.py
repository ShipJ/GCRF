"""
This module computes the CDR-derived metrics as discussed in literature.
It saves the results at the cell tower level, to individual csv files.
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.linear_model import LinearRegression
from src.config import config


def activity(num_towers, adj_matrix_v, adj_matrix_d):
    """
    Sum the total activity (volume and duration) of each cell tower over all time-steps.

    :param num_towers: int - number of cell towers.
    :param adj_matrix_vol:
    :param adj_matrix_dur:
    :return: dataframe - containing total activity of each cell tower
    """
    print 'Computing activity per bts...'
    volume_total, volume_in, volume_out, volume_self = [], [], [], []
    duration_total, duration_in, duration_out, duration_self = [], [], [], []
    for i in range(num_towers):
        vol_self, vol_in, vol_out = adj_matrix_v[i, i], np.sum(adj_matrix_v[:, i]), np.sum(adj_matrix_v[i, :])
        dur_self, dur_in, dur_out = adj_matrix_d[i, i], np.sum(adj_matrix_d[:, i]), np.sum(adj_matrix_d[i, :])
        volume_in.append(vol_in), volume_out.append(vol_out), volume_self.append(vol_self)
        duration_in.append(dur_in), duration_out.append(dur_out), duration_self.append(dur_self)
        volume_total.append(vol_in + vol_out - vol_self), duration_total.append(dur_in + dur_out - dur_self)
    total_activity = pd.DataFrame()
    total_activity['CellTowerID'] = np.array(range(num_towers))
    total_activity['Vol'] = volume_total
    total_activity['Vol_in'] = volume_in
    total_activity['Vol_out'] = volume_out
    total_activity['Vol_self'] = volume_self
    total_activity['Dur'] = duration_total
    total_activity['Dur_in'] = duration_in
    total_activity['Dur_out'] = duration_out
    total_activity['Dur_self'] = duration_self
    return total_activity

def degree_vector(num_towers, adj_matrix):
    """
    Return the degree of each cell tower

    :param num_towers: int - number of cell towers.
    :param adj_matrix:
    :return: dataframe - contains total, in and out degree of each cell tower.
    """
    print 'Computing the degree vector of each bts...'
    total_degree, in_degree, out_degree = [], [], []
    for i in range(num_towers):
        in_deg = np.count_nonzero(adj_matrix[:, i])
        out_deg = np.count_nonzero(adj_matrix[i, :])
        self_deg = 1 if adj_matrix[i, i] > 0 else 0
        total_degree.append(in_deg+out_deg-self_deg), in_degree.append(in_deg), out_degree.append(out_deg)
    deg_vec = pd.DataFrame()
    deg_vec['CellTowerID'] = np.array(range(num_towers))
    deg_vec['Degree'] = total_degree
    deg_vec['In_degree'] = in_degree
    deg_vec['Out_degree'] = out_degree
    return deg_vec

def entropy(activity, adj_matrix, deg_vector):
    """
    Compute the normalised entropy of each cell tower in the data set.
    :param country:
    :param num_towers:
    :param adj_matrix:
    :return:
    """
    print 'Computing the entropy of each bts...'

    deg_vector = deg_vector.as_matrix()
    activity = activity.as_matrix()
    q_matrix = adj_matrix / activity[:, 1, None]
    where_nan = np.where(np.isnan(q_matrix))
    q_matrix[where_nan] = 0
    log_q_matrix = np.log(q_matrix)

    where_inf = np.where(np.isinf(log_q_matrix))
    log_q_matrix[where_inf] = 0
    S = []
    for i in range(num_towers):
        q_sum = 0
        for j in range(num_towers):
            q_sum += q_matrix[i, j] * log_q_matrix[i, j]
        S.append((-1*q_sum)/np.log(deg_vector[i, 1]))
    ent = pd.DataFrame()
    ent['CellTowerID'] = np.array(range(num_towers))
    ent['Entropy'] = S
    return ent

def med_degree(num_towers, adj_matrix):
    """

    :param num_towers:
    :param adj_matrix:
    :return:
    """
    print 'Computing the median degree of each bts...'
    adj = adj_matrix.copy()

    for i in range(num_towers):
        row_col = np.concatenate((adj[i, :], adj[:, i]))
        row_col_self = np.delete(row_col, i)
        median_weight = np.median(row_col_self) - 0.1
        adj[:, i][np.where(adj[i, :] < median_weight)] = 0
        adj[i, :][np.where(adj[:, i] < median_weight)] = 0

    adj[adj > 0] = 1
    total_deg = np.zeros(num_towers)
    for i in range(num_towers):
        total_deg[i] = np.sum(np.delete(np.concatenate((adj[i, :], adj[:, i])), i))

    med_deg = pd.DataFrame()
    med_deg['CellTowerID'] = np.array(range(num_towers))
    med_deg['Med_degree'] = total_deg
    return med_deg

def introversion(num_towers, adj_matrix):
    """
    :param num_towers:
    :param adj_matrix:
    :return:
    """
    print 'Computing the introversion of each bts...'
    introv = np.zeros(num_towers)
    for i in range(num_towers):
        out = np.sum(np.delete(adj_matrix[i, :], i))
        introv[i] = (adj_matrix[i, i] / out) if out > 0 else 0
    introverse = pd.DataFrame()
    introverse['CellTowerID'] = np.array(range(num_towers))
    introverse['Introversion'] = introv
    return introverse


def graph_metrics(adj_matrix):
    """

    :param adj_matrix:
    :return:
    """
    print 'Computing various graph metrics of each bts in the network...'
    G = nx.from_numpy_matrix(adj_matrix)
    pagerank = nx.pagerank_numpy(G, weight='weight')
    evc = nx.eigenvector_centrality_numpy(G, weight='weight')
    degree = nx.degree_centrality(G)
    graph = pd.DataFrame()
    graph['CellTowerID'] = range(num_towers)
    graph['Pagerank'] = np.array(pagerank.values())
    graph['EigenvectorCentrality'] = np.array(evc.values())
    graph['Degree_centrality'] = np.array(degree.values())
    return graph


def g_residuals(adj_matrix, dist_matrix, pop, bts_adm, num_towers):
    """
    Computes the ..
    - takes a slightly different approach in that it does not result in a 'cell-tower' level metrics
    :param adj_matrix:
    :param dist_matrix:
    :param pop:
    :param bts_adm:
    :return:
    """
    print 'Computing the gravity residual value of each bts...'
    dist_matrix = dist_matrix.sort_values(by=['Source', 'Target']).reset_index(drop=True)

    pop_ab = np.array(pop['Pop_2010'])
    source = np.array(dist_matrix['Source'])
    target = np.array(dist_matrix['Target'])
    pop_a = []
    pop_b = []
    vol = []

    for i in range(len(dist_matrix)):
        pop_a.append(pop_ab[source[i]])
        pop_b.append(pop_ab[target[i]])
        vol.append(adj_matrix[source[i], target[i]])

    dist_matrix['source'] = source
    dist_matrix['target'] = target
    dist_matrix['log_pop_source'] = np.log(pop_a)
    dist_matrix['log_pop_target'] = np.log(pop_b)
    dist_matrix['vol'] = np.log(vol)

    dist_matrix = dist_matrix.replace([np.inf, -np.inf], np.nan)
    dist_matrix = dist_matrix.dropna()

    X = dist_matrix[['Distance(km)', 'log_pop_source', 'log_pop_target']]
    y = dist_matrix[['vol']]
    z = np.array(dist_matrix['vol'])

    lm = LinearRegression()
    lm.fit(X, y)

    y_hat = np.array(lm.intercept_[0] * X['log_pop_source'] ** lm.coef_[0][0] * X['log_pop_target'] ** lm.coef_[0][1] * X['Distance(km)'] ** lm.coef_[0][2])

    residuals = np.array(z - y_hat)
    g_residuals = pd.DataFrame()
    g_residuals['source'] = np.array(dist_matrix['source'])
    g_residuals['target'] = np.array(dist_matrix['target'])
    g_residuals['residual'] = residuals

    g_residuals = g_residuals.replace([np.inf, -np.inf], np.nan)
    g_residuals = g_residuals.dropna()

    neg_res = []
    for bts in pd.unique(g_residuals['source']):
        bts_residuals = g_residuals[g_residuals['source'] == bts]
        neg_res.append(sum(bts_residuals[bts_residuals['residual'] < 0]['residual']))
    g_resids = pd.DataFrame()
    g_resids['CellTowerID'] = pd.unique(g_residuals['source'])
    g_resids['G_residuals'] = np.array(neg_res)

    g_resids = g_resids.merge(bts_adm[['CellTowerID', 'Adm_1', 'Adm_2',
                                       'Adm_3', 'Adm_4']], on='CellTowerID', how='outer')
    full = pd.DataFrame()
    full['CellTowerID'] = range(num_towers)
    g_resids = pd.DataFrame(g_resids.merge(full, on='CellTowerID', how='outer'))
    return g_resids


if __name__ == '__main__':
    # path to data store
    PATH = config.get_dir()

    # ask user for country, retrieve country constants
    country = config.get_country()
    constants = config.get_constants(country)
    num_towers = constants['num_towers']

    # retrieve adjacency matrices for each country
    adj_matrix_vol = np.genfromtxt(PATH+'/processed/%s/cdr/adjacency/adj_matrix_vol.csv' % country,
                                   delimiter=',')
    adj_matrix_dur = np.genfromtxt(PATH+'/processed/%s/cdr/adjacency/adj_matrix_dur.csv' % country,
                                   delimiter=',')
    # distance_area matrix between bts's
    dist_matrix = pd.DataFrame(pd.read_csv(PATH + '/processed/%s/distance_area/dist_matrix_bts.csv' % country))

    # population per bts voronoi region (to calculate proportions)
    pop = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/pop/bts_voronoi_pop.csv' % country))

    # reference to in which adm 1,2,3,4 each bts belongs
    bts_adm = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/cdr/bts/bts_adm_1234.csv' % country))

    # Individual metrics
    total_activity = activity(num_towers, adj_matrix_vol, adj_matrix_dur)
    deg_vector = degree_vector(num_towers, adj_matrix_vol)
    entropy = entropy(total_activity[['CellTowerID', 'Vol']], adj_matrix_vol, deg_vector)
    med_deg = med_degree(num_towers, adj_matrix_vol)
    graph = graph_metrics(adj_matrix_vol)
    introv = introversion(num_towers, adj_matrix_vol)
    g_residuals = g_residuals(adj_matrix_vol, dist_matrix, pop, bts_adm, num_towers)

    # Merge individual metrics into single dataframe, preserving missing adm's as NaN's
    cdr_fundamentals_bts = pd.DataFrame(reduce(lambda left, right: pd.merge(left, right,on=['CellTowerID']),
                                               [total_activity, entropy, med_deg, graph, introv, g_residuals]))

    cdr_fundamentals_bts.to_csv(PATH+'/processed/%s/cdr/metrics/cdr_fundamentals_bts.csv' % country, index=None)



