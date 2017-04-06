"""
This module computes the CDR-derived metrics as discussed in literature.
It saves the results at the cell tower level, to individual csv files.
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.linear_model import LinearRegression
from src.config import config
import sys


def activity(ct, adj_vol, adj_dur):
    """
    Sum the total activity (volume and duration) of each cell tower over all time-steps.

    :param ct: int - number of CTs
    :param adj_vol: numpy matrix - volume adjacency matrix
    :param adj_dur: numpy matrix - duration adjacency matrix
    :return: data frame - containing total activity of each cell tower
    """
    print 'Computing activity per CT...'
    vol_total, vol_in, vol_out, vol_self = [], [], [], []
    dur_total, dur_in, dur_out, dur_self = [], [], [], []
    for i in range(num_ct):
        # Sum of in-activity, out-activity, self-activity
        v_self, v_in, v_out = adj_vol[i, i], np.sum(adj_vol[:, i]), np.sum(adj_vol[i, :])
        d_self, d_in, d_out = adj_dur[i, i], np.sum(adj_dur[:, i]), np.sum(adj_dur[i, :])
        # Append activity of each CT to arrays
        vol_in.append(v_in), vol_out.append(v_out), vol_self.append(v_self)
        dur_in.append(d_in), dur_out.append(d_out), dur_self.append(d_self)
        # Avoids double counting self-activity
        vol_total.append(v_in+v_out-v_self), dur_total.append(d_in+d_out-d_self)
    total, total['CellTowerID'] = pd.DataFrame(), np.array(range(ct))
    total['Vol'], total['Vol_in'], total['Vol_out'], total['Vol_self'] = vol_total, vol_in, vol_out, vol_self
    total['Dur'], total['Dur_in'], total['Dur_out'], total['Dur_self'] = dur_total, dur_in, dur_out, dur_self
    return total

def degree_vector(num_ct, adj_vol):
    """
    Return the degree of each cell tower

    :param num_ct: int - number of cell towers.
    :param adj_vol: numpy matrix - volume adjacency matrix
    :return: data frame - total, in, out-degree of each cell tower.
    """
    print 'Computing the degree vector of each CT...'
    total_degree, in_degree, out_degree = [], [], []
    for i in range(num_ct):
        in_deg = np.count_nonzero(adj_vol[:, i])
        out_deg = np.count_nonzero(adj_vol[i, :])
        self_deg = 1 if adj_vol[i, i] > 0 else 0
        total_degree.append(in_deg+out_deg-self_deg), in_degree.append(in_deg), out_degree.append(out_deg)
    deg_vec = pd.DataFrame()
    deg_vec['CellTowerID'] = np.array(range(num_ct))
    deg_vec['Degree'], deg_vec['In_degree'], deg_vec['Out_degree'] = total_degree, in_degree, out_degree
    return deg_vec

def entropy(activity, adj_vol, deg_vec, ct):
    """
    Compute the normalised entropy of each cell tower in the data set (in literature)
    :param activity: data frame - total volume of each CT
    :param adj_vol: numpy matrix - volume adjacency matrix
    :param deg_vec: data frame - degree vector of each CT
    :param ct: int - number of CTs
    :return:
    """
    print 'Computing the entropy of each CT...'
    deg_vector = deg_vec.as_matrix()
    activity = activity.as_matrix()
    q_matrix = adj_vol / activity[:, 1, None]
    where_nan = np.where(np.isnan(q_matrix))
    q_matrix[where_nan] = 0
    log_q_matrix = np.log(q_matrix)
    where_inf = np.where(np.isinf(log_q_matrix))
    log_q_matrix[where_inf] = 0
    S = []
    for i in range(ct):
        q_sum = 0
        for j in range(ct):
            q_sum += q_matrix[i, j] * log_q_matrix[i, j]
        S.append((-1*q_sum)/np.log(deg_vector[i, 1]))
    ent, ent['CellTowerID'], ent['Entropy'] = pd.DataFrame(), np.array(range(ct)), S
    return ent

def med_degree(adj_vol, ct):
    """
    Computes the median degree of each cell tower (i.e. the degree of CTs after removing links < median strength
    :param adj_matrix: numpy matrix - volume adjacency matrix
    :param ct: int - number of CTs
    :return: data frame - median degree of each CT
    """
    print 'Computing the median degree of each CT...'
    adj = adj_vol.copy()
    for i in range(ct):
        row_col = np.concatenate((adj[i, :], adj[:, i]))
        row_col_self = np.delete(row_col, i)
        median_weight = np.median(row_col_self) - 0.1
        adj[:, i][np.where(adj[i, :] < median_weight)] = 0
        adj[i, :][np.where(adj[:, i] < median_weight)] = 0
    adj[adj > 0] = 1
    total_deg = np.zeros(ct)
    for i in range(ct):
        total_deg[i] = np.sum(np.delete(np.concatenate((adj[i, :], adj[:, i])), i))
    med_deg, med_deg['CellTowerID'], med_deg['Med_degree'] = pd.DataFrame(), np.array(range(ct)), total_deg
    return med_deg

def introversion(adj_vol, ct):
    """
    Compute the introversion of each CT
    :param adj_vol: numpy matrix - volume adjacency matrix
    :param ct: int - number of CTs
    :return:
    """
    print 'Computing the introversion of each CT...'
    introv = np.zeros(ct)
    for i in range(ct):
        out = np.sum(np.delete(adj_vol[i, :], i))
        introv[i] = (adj_vol[i, i] / out) if out > 0 else 0
    ct_introv, ct_introv['CellTowerID'], ct_introv['Introversion'] = pd.DataFrame(), np.array(range(ct)), introv
    return ct_introv


def graph_metrics(adj_vol, ct):
    """
    Compute graph metrics including pagerank, degree centrality and eigenvector centrality (others can be added)
    :param adj_vol: numpy matrix - volume adjacency matrix
    :param ct: int - number of CTs
    :return:
    """
    print 'Computing graph metrics of each CT'
    G = nx.from_numpy_matrix(adj_vol)
    pagerank = nx.pagerank_numpy(G, weight='weight')
    evc, deg = nx.eigenvector_centrality_numpy(G, weight='weight'), nx.degree_centrality(G)
    g, g['CellTowerID'], g['Pagerank'] = pd.DataFrame(), range(ct), np.array(pagerank.values())
    g['Eig_central'], g['Deg_central'] = np.array(evc.values()), np.array(deg.values())
    return g


def g_residuals(adj_vol, ct_dist, pop, ct_adm, ct):
    """
    Computes the gravity residuals of each CT
    :param adj_vol: numpy matrix - volume adjacency matrix
    :param ct_dist: numpy matrix - distance matrix of CTs
    :param pop:
    :param ct_adm: data frame - identifies CTs with ADMs
    :param ct: int - number of CTs
    :return:
    """
    print 'Computing gravity residual of each CT...'
    dist_matrix = ct_dist.sort_values(by=['Source', 'Target']).reset_index(drop=True)

    # Population per voronoi region of each CT (including 0 for missing CTs)
    voronoi_pop = pop.groupby('CellTowerID')['Pop_2010'].sum().reset_index()
    pop, pop['CellTowerID'] = pd.DataFrame(), range(1240)
    pop = np.array(pop.merge(voronoi_pop, on='CellTowerID', how='outer').replace(np.nan, 0)['Pop_2010'])

    source, target = np.array(dist_matrix['Source']), np.array(dist_matrix['Target'])
    pop_a, pop_b, vol = [], [], []

    for i in range(len(dist_matrix)):
        pop_a.append(pop[source[i]]), pop_b.append(pop[target[i]]), vol.append(adj_vol[source[i], target[i]])

    dist_matrix['source'], dist_matrix['target'] = source, target
    dist_matrix['log_pop_source'], dist_matrix['log_pop_target'] = np.log(pop_a), np.log(pop_b)
    dist_matrix['vol'] = np.log(vol)
    dist_matrix = dist_matrix.replace([np.inf, -np.inf], np.nan).dropna()

    X = dist_matrix[['Distance(km)', 'log_pop_source', 'log_pop_target']]
    y = dist_matrix[['vol']]
    z = np.array(dist_matrix['vol'])

    lm = LinearRegression()
    lm.fit(X, y)
    y_hat = np.array(lm.intercept_[0] * X['log_pop_source'] ** lm.coef_[0][0]
                     * X['log_pop_target'] ** lm.coef_[0][1] * X['Distance(km)'] ** lm.coef_[0][2])

    residuals = np.array(z - y_hat)
    g_res = pd.DataFrame()
    g_res['source'], g_res['target'],  = np.array(dist_matrix['source']), np.array(dist_matrix['target'])
    g_res['residual']= residuals
    g_resids = g_res.replace([np.inf, -np.inf], np.nan).dropna()

    neg_res = []
    for ct in pd.unique(g_resids['source']):
        ct_residuals = g_resids[g_resids['source'] == ct]
        neg_res.append(sum(ct_residuals[ct_residuals['residual'] < 0]['residual']))
    g_resids, g_resids['CellTowerID'], g_resids['G_resids'] = pd.DataFrame(), pd.unique(g_resids['source']), np.array(neg_res)
    g_resids = g_resids.merge(ct_adm[['CellTowerID', 'Adm_1', 'Adm_2',
                                      'Adm_3', 'Adm_4']], on='CellTowerID', how='outer')

    full, full['CellTowerID'] = pd.DataFrame(), range(ct)
    g_residuals = pd.DataFrame(g_resids.merge(full, on='CellTowerID', how='outer'))
    return g_residuals

def compute_merge(adj_vol, adj_dur, ct, ct_dist, ct_adm, pop):
    """
    Compute each metric in turn and merge results into a data frame
    :param adj_matrix_vol: numpy matrix
    :param adj_matrix_dur:
    :param num_ct:
    :param ct_dist_matrix:
    :param ct_adm:
    :param pop:
    :return:
    """
    ct_activity = activity(num_ct, adj_vol, adj_dur)
    ct_degree_vector = degree_vector(num_ct, adj_vol)
    ct_entropy = entropy(ct_activity[['CellTowerID', 'Vol']], adj_vol, ct_degree_vector, ct)
    ct_median_degree = med_degree(adj_vol, ct)
    ct_introversion = introversion(adj_vol, ct)
    ct_graph = graph_metrics(adj_vol, ct)
    ct_g_residuals = g_residuals(adj_vol, ct_dist, pop, ct_adm, ct)

    # Merge metrics into single data frame, preserving missing adm's as NaN's
    cdr_fundamentals_ct = pd.DataFrame(reduce(lambda left, right: pd.merge(left, right, on=['CellTowerID']),
                                               [ct_activity, ct_entropy, ct_median_degree,
                                                ct_introversion, ct_graph, ct_g_residuals]))
    return cdr_fundamentals_ct


if __name__ == '__main__':
    # System path to data directory
    PATH = config.get_dir()

    # Ask user for country
    country = config.get_country()
    # Number of CTs per country
    num_ct = config.get_constants(country)['num_towers']

    # distance matrix between CT's
    ct_dist_matrix = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/geo/ct_dist_matrix.csv' % country))
    # population per intersected region (Adms vs CTs)
    pop = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/pop/intersect_pop.csv' % country))
    # reference to in which adm 1,2,3,4 each CT belongs
    ct_adm = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/cdr/ct_locations/ct_adm_1234.csv' % country))

    # Currently only metrics for the 'all-hours' adj matrix. Change 'all' to 'working' (or both) for different metrics.
    for i in ['all']:
        print 'Retrieving adjacency matrices'
        adj_matrix_vol = np.genfromtxt(PATH +
                                       '/processed/%s/cdr/adjacency/adj_matrix_vol_%s.csv' % (country, i),
                                       delimiter=',')
        adj_matrix_dur = np.genfromtxt(PATH +
                                       '/processed/%s/cdr/adjacency/adj_matrix_dur_%s.csv' % (country, i),
                                       delimiter=',')

        # Compute metrics
        cdr_fundamentals_ct = compute_merge(adj_matrix_vol, adj_matrix_dur, num_ct, ct_dist_matrix, ct_adm, pop)
        # Save to csv
        cdr_fundamentals_ct.to_csv(PATH +
                                   '/processed/%s/cdr/metrics/cdr_fundamentals_bts_%s.csv' % (country, i),
                                   index=None)
