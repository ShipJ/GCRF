"""
This module computes the CDR-derived metrics as discussed in literature.
It saves the results at the cell tower level, to individual csv files.
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.linear_model import LinearRegression
from src.config import config


def activity(num_towers, adj_matrix_vol, adj_matrix_dur):
    """
    Sum the total activity (volume and duration) of each cell tower over all time-steps.

    :param num_towers: int - number of cell towers.
    :param adj_matrix_vol:
    :param adj_matrix_dur:
    :return: dataframe - containing total activity of each cell tower
    """
    volume_total, volume_in, volume_out, volume_self = [], [], [], []
    duration_total, duration_in, duration_out, duration_self = [], [], [], []
    for i in range(num_towers):
        vol_self, vol_in, vol_out = adj_matrix_vol[i, i], np.sum(adj_matrix_vol[:, i]), np.sum(adj_matrix_vol[i, :])
        dur_self, dur_in, dur_out = adj_matrix_dur[i, i], np.sum(adj_matrix_dur[:, i]), np.sum(adj_matrix_dur[i, :])
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

def entropy(country, num_towers, adj_matrix):
    """
    Compute the normalised entropy of each cell tower in the data set.
    :param country:
    :param num_towers:
    :param adj_matrix:
    :return:
    """
    total_activity, deg_vector = config.get_cdr_features(country)
    total_activity = total_activity.as_matrix()
    deg_vector = deg_vector.as_matrix()

    q_matrix = adj_matrix / total_activity[:, 1, None]
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

    for i in range(num_towers):
        row_col = np.concatenate((adj_matrix[i, :], adj_matrix[:, i]))
        row_col_self = np.delete(row_col, i)
        median_weight = np.median(row_col_self) - 0.1
        adj_matrix[:, i][np.where(adj_matrix[i, :] < median_weight)] = 0
        adj_matrix[i, :][np.where(adj_matrix[:, i] < median_weight)] = 0

    adj_matrix[adj_matrix > 0] = 1
    total_deg = np.zeros(num_towers)
    for i in range(num_towers):
        total_deg[i] = np.sum(np.delete(np.concatenate((adj_matrix[i, :], adj_matrix[:, i])), i))

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


def gravity(adj_matrix, dist_matrix, pop, country):
    """
    Computes the ..
    - takes a slightly different approach in that it does not result in a 'cell-tower' level metrics
    :param adj_matrix:
    :param dist_matrix:
    :param pop:
    :param country:
    :return:
    """

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
    g_resids['Residuals'] = np.array(neg_res)

    bts_adm = pd.DataFrame(pd.read_csv('../../../../data/processed/%s/cdr/bts/bts_adm_1234.csv' % country))

    g_resids = g_resids.merge(bts_adm[['CellTowerID', 'Adm_1', 'Adm_2',
                                       'Adm_3', 'Adm_4']], on='CellTowerID', how='outer')

    full = pd.DataFrame()
    full['CellTowerID'] = range(1668)

    g_resids = pd.DataFrame(g_resids.merge(full, on='CellTowerID', how='outer'))
    return g_resids.sort_values('CellTowerID')


if __name__ == '__main__':
    country = config.get_country()
    constants = config.get_constants(country)
    num_towers = constants['num_towers']
    adj_matrix_vol = np.genfromtxt('../../../../data/processed/%s/cdr/metrics/adj_matrix_vol.csv' % country,
                                   delimiter=',')
    adj_matrix_dur = np.genfromtxt('../../../../data/processed/%s/cdr/metrics/adj_matrix_dur.csv' % country,
                                   delimiter=',')

    # total_activity = activity(country, num_towers, adj_matrix_vol, adj_matrix_dur)
    # deg_vector = degree_vector(country, num_towers, adj_matrix_vol)
    # entropy = entropy(country, num_towers, adj_matrix_vol)
    # med_deg = med_degree(country, num_towers, adj_matrix_vol)
    # graph = graph_metrics(adj_matrix_vol)
    # introv = introversion(num_towers, adj_matrix_vol)
    # gravity = gravity(adj_matrix_vol, dist_matrix, pop)
    # radiation = radiation(adj_matrix_vol, dist_matrix, pop)



    # a = pd.DataFrame(pd.read_csv('../../../../data/processed/%s/distance/dist_matrix_bts.csv' % country))
    # a['Distance(km)'] = (a['Distance'] * 132)
    # a = pd.DataFrame(a.drop('Distance', axis=1))
    # a.columns = ['Source', 'Target', 'Distance(km)']
    # a.to_csv('../../../../data/processed/%s/distance/dist_matrix_bts.csv' % country, index=None)


    # pop = pd.DataFrame(pd.read_csv('../../../../data/processed/%s/pop/intersect_pop.csv' % country))
    # bts_pop = pop.groupby('CellTowerID')['Pop_2010'].sum().reset_index()
    # bts_pop = pd.DataFrame(bts_pop.set_index('CellTowerID').reindex(np.array(range(1669))).reset_index())
    # bts_pop.to_csv('../../../../data/processed/%s/pop/bts_voronoi_pop.csv' % country, index=None)


    dist_matrix = pd.DataFrame(pd.read_csv('../../../../data/processed/%s/distance/dist_matrix_bts.csv' % country))
    pop = pd.DataFrame(pd.read_csv('../../../../data/processed/%s/pop/bts_voronoi_pop.csv' % country))

    gravity = gravity(adj_matrix_vol, dist_matrix, pop, country)


    ''' Save to csv '''
    # total_activity.to_csv('../../../../data/processed/%s/cdr/static_metrics/new/total_activity.csv' % country,
    #                       index=None)
    # deg_vector.to_csv('../../../../data/processed/%s/cdr/static_metrics/new/degree_vector.csv' % country,
    #                   index=None)
    # entropy.to_csv('../../../../data/processed/%s/cdr/static_metrics/new/entropy.csv' % country,
    #                index=None)
    # med_deg.to_csv('../../../../data/processed/%s/cdr/static_metrics/new/med_degree.csv' % country,
    #                index=None)
    # introv.to_csv('../../../../data/processed/%s/cdr/staticmetrics/new/introversion.csv' % country,
    #                index=None)
    # graph.to_csv('../../../../data/processed/%s/cdr/static_metrics/new/graph_metrics.csv' % country,
    #              index=None)
    gravity.to_csv('../../../../data/processed/%s/cdr/metrics/gravity.csv' % country,
                   index=None)
    # radiation.to_csv('../../../../data/processed/%s/cdr/static_metrics/new/radiation.csv' % country,
    #                  index=None)


