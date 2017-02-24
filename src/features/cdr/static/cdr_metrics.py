import pandas as pd
import numpy as np
import networkx as nx

from sklearn.linear_model import LinearRegression
from src.config import config


def activity(country, num_towers, adj_matrix_vol, adj_matrix_dur):
    """
    Sum the total activity (volume and duration) of each cell tower over all time-steps.
    :param country: str - country code.
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

def degree_vector(country, num_towers, adj_matrix):
    """
    Return the degree of each cell tower
    :param country: str - country code.
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

def med_degree(country, num_towers, adj_matrix):
    """

    :param country:
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

def gravity(adj_matrix, dist_matrix, pop):
    """

    :param adj_matrix:
    :param pop:
    :param dist:
    :return:
    """

    ### X = log (Pop(a), Pop(b), dist(a, b))
    ### y = log(flows[flows > 0))
    X = 0
    y = 0

    lm = LinearRegression()
    lm.fit(X, y)

    beta = np.concatenate(([np.exp(lm.intercept_)], lm.coef_))

    y_hat = beta[0] * X['Pop(A)'] ** beta[1] * X['Pop(B)'] ** beta[2] * X['dist(A,B)'] ** beta[3]

    return lm.score(X, y)



if __name__ == '__main__':
    country = config.get_country()
    constants = config.get_constants(country)
    num_towers = constants['num_towers']
    adj_matrix_vol = np.genfromtxt('../../../../data/processed/%s/cdr/staticmetrics/adj_matrix_vol.csv' % country,
                                   delimiter=',')
    adj_matrix_dur = np.genfromtxt('../../../../data/processed/%s/cdr/staticmetrics/adj_matrix_dur.csv' % country,
                                   delimiter=',')

    # total_activity = activity(country, num_towers, adj_matrix_vol, adj_matrix_dur)
    # deg_vector = degree_vector(country, num_towers, adj_matrix_vol)
    # entropy = entropy(country, num_towers, adj_matrix_vol)
    # med_deg = med_degree(country, num_towers, adj_matrix_vol)
    # graph = graph_metrics(adj_matrix_vol)
    introv = introversion(num_towers, adj_matrix_vol)
    # gravity = gravity(adj_matrix_vol, dist_matrix, pop)
    # radiation = radiation(adj_matrix_vol, dist_matrix, pop)

    ''' Save to csv '''
    # total_activity.to_csv('../../../../data/processed/%s/cdr/static_metrics/new/total_activity.csv' % country,
    #                       index=None)
    # deg_vector.to_csv('../../../../data/processed/%s/cdr/static_metrics/new/degree_vector.csv' % country,
    #                   index=None)
    # entropy.to_csv('../../../../data/processed/%s/cdr/static_metrics/new/entropy.csv' % country,
    #                index=None)
    # med_deg.to_csv('../../../../data/processed/%s/cdr/static_metrics/new/med_degree.csv' % country,
    #                index=None)
    introv.to_csv('../../../../data/processed/%s/cdr/staticmetrics/new/introversion.csv' % country,
                   index=None)
    # graph.to_csv('../../../../data/processed/%s/cdr/static_metrics/new/graph_metrics.csv' % country,
    #              index=None)
    # gravity.to_csv('../../../../data/processed/%s/cdr/static_metrics/new/gravity.csv' % country,
    #                index=None)
    # radiation.to_csv('../../../../data/processed/%s/cdr/static_metrics/new/radiation.csv' % country,
    #                  index=None)


