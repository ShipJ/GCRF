import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import sklearn.metrics as sk
import networkx as nx
import os
from src.config import config


def activity(country):
    adj_matrix = config.get_adj_matrix(country)
    print adj_matrix

    sys.exit()

    volume_total, volume_in, volume_out = [], [], []
    for i in range(1240):
        vol_self = adj_matrix[i, i]
        vol_in = np.sum(adj_matrix[:, i])
        vol_out = np.sum(adj_matrix[i, :])
        volume_in.append(vol_in)
        volume_out.append(vol_out)
        volume_total.append(vol_in + vol_out - vol_self)

    total_activity = pd.DataFrame()
    total_activity['ID'] = np.array(range(num_bts))
    total_activity['Vol'] = volume_total
    total_activity['Vol_in'] = volume_in
    total_activity['Vol_out'] = volume_out

    total_activity.to_csv('activity.csv', delimiter=',', index=None)






if __name__ == '__main__':
    country = config.get_country()
    constants = config.get_constants(country)

    activity(country)












''' Activity '''
# adj_matrix = np.genfromtxt(path+'/CDR/staticmetrics/Other/adj_matrix_civ.csv', delimiter=',')
# volume_total, volume_in, volume_out = [], [], []
# for i in range(1240):
#     vol_self = adj_matrix[i, i]
#     vol_in = np.sum(adj_matrix[:, i])
#     vol_out = np.sum(adj_matrix[i, :])
#     volume_in.append(vol_in)
#     volume_out.append(vol_out)
#     volume_total.append(vol_in + vol_out - vol_self)
#
# total_activity = pd.DataFrame()
# total_activity['ID'] = np.array(range(num_bts))
# total_activity['Vol'] = volume_total
# total_activity['Vol_in'] = volume_in
# total_activity['Vol_out'] = volume_out
#
# total_activity.to_csv('activity.csv', delimiter=',', index=None)

# adj_matrix = np.genfromtxt(path+'/CDR/staticmetrics/Other/adj_matrix_civ.csv', delimiter=',')
# deg_vector = []
# for i in range(1240):
#     in_deg = np.count_nonzero(adj_matrix[:, i])
#     out_deg = np.count_nonzero(adj_matrix[i, :])
#     self_deg = 1 if adj_matrix[i, i] > 0 else 0
#     deg_vector.append(in_deg+out_deg-self_deg)
# np.savetxt('deg_vector.csv', deg_vector, delimiter=',')

''' Entropy '''
# adj_matrix = np.genfromtxt(path+'/CDR/staticmetrics/Other/adj_matrix_civ.csv', delimiter=',')
# total_activity = np.genfromtxt('activity.csv', delimiter=',', skiprows=1)
# deg_vector = np.genfromtxt('deg_vector.csv', delimiter=',')
#
# q_matrix = adj_matrix / np.array(total_activity[:, 1, None])
# where_nan = np.where(np.isnan(q_matrix))
# q_matrix[where_nan] = 0
# log_q_matrix = np.log(q_matrix)
# where_inf = np.where(np.isinf(log_q_matrix))
# log_q_matrix[where_inf] = 0
# S = []
# for i in range(1240):
#     q_sum = 0
#     for j in range(1240):
#         q_sum += q_matrix[i, j] * log_q_matrix[i, j]
#     S.append((-1*q_sum)/np.log(deg_vector[i]))
# np.savetxt('entropy.csv', S, delimiter=',')


''' Median Degree '''
# adj_matrix = np.genfromtxt(path+'/CDR/staticmetrics/Other/adj_matrix_civ.csv', delimiter=',')
# for i in range(num_bts):
#     row_col = np.concatenate((adj_matrix[i, :], adj_matrix[:, i]))
#     row_col_self = np.delete(row_col, i)
#     median_weight = np.median(row_col_self) - 0.1
#     adj_matrix[:, i][np.where(adj_matrix[i, :] < median_weight)] = 0
#     adj_matrix[i, :][np.where(adj_matrix[:, i] < median_weight)] = 0
# adj_matrix[adj_matrix > 0] = 1
# total_deg = np.zeros(num_bts)
# for i in range(num_bts):
#     total_deg[i] = np.sum(np.delete(np.concatenate((adj_matrix[i, :], adj_matrix[:, i])), i))
# np.savetxt('med_degree.csv', total_deg, delimiter=',')


''' Introversion '''
# adj_matrix = np.genfromtxt(path+'/CDR/staticmetrics/Other/adj_matrix_civ.csv', delimiter=',')
# introversion = np.zeros(num_bts)
# for i in range(num_bts):
#     out = np.sum(np.delete(adj_matrix[i, :], i))
#     introversion[i] = (adj_matrix[i, i] / out) if out > 0 else 0
# np.savetxt('introversion.csv', introversion, delimiter=',')


''' Eigen/degree/pagerank centrality '''
# compute eigenvector centrality and pagerank of each different graph
# adj_matrix = np.genfromtxt(path+'/CDR/staticmetrics/Other/adj_matrix_civ.csv', delimiter=',')
#
# G = nx.from_numpy_matrix(adj_matrix)
# # pagerank = nx.pagerank_numpy(G, weight='weight')
# # evc = nx.eigenvector_centrality_numpy(G, weight='weight')
# degree = nx.degree_centrality(G)
# closeness = nx.closeness_centrality(G)
#
# # np.savetxt('pagerank.csv', pagerank.values(), delimiter=',')
# # np.savetxt('evc.csv', evc.values(), delimiter=',')
# np.savetxt('degree.csv', degree.values(), delimiter=',')
# np.savetxt('closeness.csv', closeness.values(), delimiter=',')


# evc = np.genfromtxt('evc.csv', delimiter=',')
# pagerank = np.genfromtxt('pagerank.csv', delimiter=',')
# degree = np.genfromtxt('degree.csv', delimiter=',')
# closeness = np.genfromtxt('closeness.csv', delimiter=',')
#
# cdr = pd.DataFrame(pd.read_csv(path+'/CDR/celltowers/CDR_Adm_1234.csv'))
#
# evc = evc[np.array(cdr['CellTowerID'])]
# pagerank = pagerank[np.array(cdr['CellTowerID'])]
# degree = degree[np.array(cdr['CellTowerID'])]
# closeness = closeness[np.array(cdr['CellTowerID'])]
#
# cdr['evc'] = evc
# cdr['pagerank'] = pagerank
# cdr['degree'] = degree
# cdr['closeness'] = closeness
#
# metrics_4 = pd.DataFrame(cdr.groupby('Adm_4')['evc', 'pagerank', 'degree', 'closeness'].median().reset_index())
# metrics_4.to_csv('graph_metrics.csv', index=None)







# sys.exit()
# ''' Gravity Residual '''
# adj_matrix = np.genfromtxt(path+'/CDR/staticmetrics/Other/adj_matrix_civ.csv', delimiter=',')
# intersect_pop = pd.DataFrame(pd.read_csv(path+'/Population/IntersectPop.csv'))
# pop_1 = np.array(intersect_pop.groupby('Adm_1')['Pop_2010'].sum())
# pop_2 = np.array(intersect_pop.groupby('Adm_2')['Pop_2010'].sum())
# pop_3 = np.array(intersect_pop.groupby('Adm_3')['Pop_2010'].sum())
# pop_4 = np.array(intersect_pop.groupby('Adm_4')['Pop_2010'].sum())
#
# dist_matrix_1 = pd.DataFrame(pd.read_csv(path+'/CDR/staticmetrics/GravityResidual/dist_matrix_1.csv'))
# # dist_matrix_2 = pd.DataFrame(pd.read_csv(path+'/CDR/staticmetrics/GravityResidual/dist_matrix_2.csv'))
# # dist_matrix_3 = pd.DataFrame(pd.read_csv(path+'/CDR/staticmetrics/GravityResidual/dist_matrix_3.csv'))
# # dist_matrix_4 = pd.DataFrame(pd.read_csv(path+'/CDR/staticmetrics/GravityResidual/dist_matrix_4.csv'))
#
# # Adm_1
# g_resid = []
# for i in range(14):
#     F_est = []
#     for j in range(1, 15):
#         dist = dist_matrix_1[dist_matrix_1['InputID'] == i+1]
#         dist = np.array(dist[dist['TargetID'] == j]['Distance'])
#         F_est.append(((pop_1[i] * pop_1[j-1]) / np.power(dist, 2))[0])
#     g_resid.append(F_est)
#
# observed = np.zeros((14, 14))
# for i in range(1, 15):
#     u = np.array(intersect_pop[intersect_pop['Adm_1'] == i]['CellTowerID'])
#     for j in np.setdiff1d(range(1, 15), [i]):
#         v = np.array(intersect_pop[intersect_pop['Adm_1'] == j]['CellTowerID'])
#         flows = 0
#         for source in u:
#             for target in v:
#                 flows += adj_matrix[source, target]
#         observed[i-1, j-1] = flows
#
# where_inf = np.where(np.array(g_resid) > 10000000000)
# observed[where_inf] = 1.1
# g_resid = np.array(g_resid)
# g_resid[where_inf] = 1.1
#
# observed = np.log(observed)
# expected = np.multiply(np.log(g_resid), 4.241)
#
# g_residual_1 = np.zeros(14)
# for i in range(14):
#     neg_res = np.zeros(14)
#     for j in range(14):
#         neg_res[j] = (observed[i, j] - expected[i, j])
#     num_neg = len(neg_res[neg_res < 0])
#     sum_neg = sum(neg_res[neg_res < 0])
#     g_residual_1[i] = sum_neg / float(num_neg)
#
# np.savetxt('g_residual_1.csv', g_residual_1, delimiter=',')
# #
# #
# # # Adm_2
# # g_resid = []
# # for i in range(33):
# #     F_est = []
# #     for j in range(1, 34):
# #         dist = dist_matrix_1[dist_matrix_1['InputID'] == i+1]
# #         dist = np.array(dist[dist['TargetID'] == j]['Distance'])
# #         F_est.append(((pop_2[i] * pop_2[j-1]) / np.power(dist, 2))[0])
# #     g_resid.append(F_est)
# #
# # observed = np.zeros((33, 33))
# # for i in range(1, 34):
# #     u = np.array(intersect_pop[intersect_pop['Adm_2'] == i]['CellTowerID'])
# #     for j in np.setdiff1d(range(1, 34), [i]):
# #         v = np.array(intersect_pop[intersect_pop['Adm_2'] == j]['CellTowerID'])
# #         flows = 0
# #         for source in u:
# #             for target in v:
# #                 flows += adj_matrix[source, target]
# #         observed[i-1, j-1] = flows
# #
# # where_inf = np.where(np.array(g_resid) > 10000000000)
# # observed[where_inf] = 1.1
# # g_resid = np.array(g_resid)
# # g_resid[where_inf] = 1.1
# #
# # observed = np.log(observed)
# # expected = np.multiply(np.log(g_resid), 4.241)
# #
# # g_residual_2 = np.zeros(33)
# # for i in range(33):
# #     neg_res = np.zeros(33)
# #     for j in range(33):
# #         neg_res[j] = (observed[i, j] - expected[i, j])
# #     num_neg = len(neg_res[neg_res < 0])
# #     sum_neg = sum(neg_res[neg_res < 0])
# #     g_residual_2[i] = sum_neg / float(num_neg)
# #
# #
# #
# # np.savetxt('g_residual_1', g_residual_1, delimiter=',')
# # np.savetxt('g_residual_2', g_residual_2, delimiter=',')
# # np.savetxt('g_residual_3', g_residual_3, delimiter=',')
# # np.savetxt('g_residual_4', g_residual_4, delimiter=',')
#
#
#











''' Normalise each metric by Adm region '''
# activity = pd.DataFrame(pd.read_csv('activity.csv', delimiter=','))
# entropy = np.genfromtxt('entropy.csv', delimiter=',')
# med_degree = np.genfromtxt('med_degree.csv', delimiter=',')
# introversion = np.genfromtxt('introversion.csv', delimiter=',')
#
# cell_tower_adm = pd.DataFrame(pd.read_csv(path+'/CDR/celltowers/CDR_Adm_1234.csv'))
# intersect_pop = pd.DataFrame(pd.read_csv(path+'/Population/IntersectPop.csv'))
#
# activity_1, activity_2, activity_3, activity_4 = [], [], [], []
# entropy_1, entropy_2, entropy_3, entropy_4 = [], [], [], []
# med_degree_1, med_degree_2, med_degree_3, med_degree_4 = [], [], [], []
# introversion_1, introversion_2, introversion_3, introversion_4 = [], [], [], []
#
# pop_1 = intersect_pop.groupby('Adm_1')['Pop_2010'].sum()
# pop_2 = intersect_pop.groupby('Adm_2')['Pop_2010'].sum()
# pop_3 = intersect_pop.groupby('Adm_3')['Pop_2010'].sum()
# pop_4 = intersect_pop.groupby('Adm_4')['Pop_2010'].sum()
#
# cell_towers_adm_1 = np.array(cell_tower_adm.groupby('ID_1')['ID_1'].count())
# cell_towers_adm_2 = np.array(cell_tower_adm.groupby('ID_2')['ID_2'].count())
# cell_towers_adm_3 = np.array(cell_tower_adm.groupby('ID_3')['ID_3'].count())
# cell_towers_adm_4 = np.array(cell_tower_adm.groupby('ID_4')['ID_4'].count())
#
# for i in range(1, 15):
#     data = intersect_pop[intersect_pop['Adm_1'] == i]
#     act_sum, ent_sum, med_sum, int_sum = 0, 0, 0, 0
#     for j in range(len(data)):
#         cell_tower = np.array(data['CellTowerID'])[j]
#         act_sum += (np.array(data['Pop_2010'])[j] / pop_1[i]) * np.array(activity['Vol'])[cell_tower]
#         ent_sum += (np.array(data['Pop_2010'])[j] / pop_1[i]) * entropy[cell_tower]
#         med_sum += (np.array(data['Pop_2010'])[j] / pop_1[i]) * med_degree[cell_tower]
#         int_sum += (np.array(data['Pop_2010'])[j] / pop_1[i]) * introversion[cell_tower]
#     activity_1.append(act_sum)
#     entropy_1.append(ent_sum/float(cell_towers_adm_1[i-1]))
#     med_degree_1.append(med_sum/float(cell_towers_adm_1[i-1]))
#     introversion_1.append(int_sum/float(cell_towers_adm_1[i-1]))
# np.savetxt('activity_1.csv', activity_1, delimiter=',')
# np.savetxt('entropy_1.csv', entropy_1, delimiter=',')
# np.savetxt('med_degree_1.csv', med_degree_1, delimiter=',')
# np.savetxt('introversion_1.csv', introversion_1, delimiter=',')
#
# for i in range(1, 34):
#     data = intersect_pop[intersect_pop['Adm_2'] == i]
#     act_sum, ent_sum, med_sum, int_sum = 0, 0, 0, 0
#     for j in range(len(data)):
#         cell_tower = np.array(data['CellTowerID'])[j]
#         act_sum += (np.array(data['Pop_2010'])[j] / pop_2[i]) * np.array(activity['Vol'])[cell_tower]
#         ent_sum += (np.array(data['Pop_2010'])[j] / pop_2[i]) * entropy[cell_tower]
#         med_sum += (np.array(data['Pop_2010'])[j] / pop_2[i]) * med_degree[cell_tower]
#         int_sum += (np.array(data['Pop_2010'])[j] / pop_2[i]) * introversion[cell_tower]
#     activity_2.append(act_sum)
#     entropy_2.append(ent_sum/float(cell_towers_adm_2[i-1]))
#     med_degree_2.append(med_sum/float(cell_towers_adm_2[i-1]))
#     introversion_2.append(int_sum/float(cell_towers_adm_2[i-1]))
# np.savetxt('activity_2.csv', activity_2, delimiter=',')
# np.savetxt('entropy_2.csv', entropy_2, delimiter=',')
# np.savetxt('med_degree_2.csv', med_degree_2, delimiter=',')
# np.savetxt('introversion_2.csv', introversion_2, delimiter=',')
#
# for i in range(1, 112):
#     data = intersect_pop[intersect_pop['Adm_3'] == i]
#     act_sum, ent_sum, med_sum, int_sum = 0, 0, 0, 0
#     for j in range(len(data)):
#         cell_tower = np.array(data['CellTowerID'])[j]
#         act_sum += (np.array(data['Pop_2010'])[j] / pop_3[i]) * np.array(activity['Vol'])[cell_tower]
#         ent_sum += (np.array(data['Pop_2010'])[j] / pop_3[i]) * entropy[cell_tower]
#         med_sum += (np.array(data['Pop_2010'])[j] / pop_3[i]) * med_degree[cell_tower]
#         int_sum += (np.array(data['Pop_2010'])[j] / pop_3[i]) * introversion[cell_tower]
#     activity_3.append(act_sum)
#     entropy_3.append(ent_sum/float(cell_towers_adm_3[i-1]))
#     med_degree_3.append(med_sum/float(cell_towers_adm_3[i-1]))
#     introversion_3.append(int_sum/float(cell_towers_adm_3[i-1]))
# np.savetxt('activity_3.csv', activity_3, delimiter=',')
# np.savetxt('entropy_3.csv', entropy_3, delimiter=',')
# np.savetxt('med_degree_3.csv', med_degree_3, delimiter=',')
# np.savetxt('introversion_3.csv', introversion_3, delimiter=',')
#
# for i in range(1, 179):
#     data = intersect_pop[intersect_pop['Adm_4'] == i]
#     act_sum, ent_sum, med_sum, int_sum = 0, 0, 0, 0
#     for j in range(len(data)):
#         cell_tower = np.array(data['CellTowerID'])[j]
#         act_sum += (np.array(data['Pop_2010'])[j] / pop_4[i]) * np.array(activity['Vol'])[cell_tower]
#         ent_sum += (np.array(data['Pop_2010'])[j] / pop_4[i]) * entropy[cell_tower]
#         med_sum += (np.array(data['Pop_2010'])[j] / pop_4[i]) * med_degree[cell_tower]
#         int_sum += (np.array(data['Pop_2010'])[j] / pop_4[i]) * introversion[cell_tower]
#     activity_4.append(act_sum)
#     entropy_4.append(ent_sum/float(cell_towers_adm_4[i-1]))
#     med_degree_4.append(med_sum/float(cell_towers_adm_4[i-1]))
#     introversion_4.append(int_sum/float(cell_towers_adm_4[i-1]))
# np.savetxt('activity_4.csv', activity_4, delimiter=',')
# np.savetxt('entropy_4.csv', entropy_4, delimiter=',')
# np.savetxt('med_degree_4.csv', med_degree_4, delimiter=',')
# np.savetxt('introversion_4.csv', introversion_4, delimiter=',')
