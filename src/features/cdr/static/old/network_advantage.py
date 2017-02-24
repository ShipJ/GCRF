''' Two Network Advantage metrics: 1. Median degree, 2. Normalised Entropy '''
# Input:
# Output:

import pandas as pd
import numpy as np
import sys

if __name__ == "__main__":

    # country = sys.argv[1]
    country = 'IvoryCoast'
    path = '/Users/JackShipway/Desktop/UCLProject/Data/%s' % country

    # Set known data set values (number of towers, and time length of data)
    if country == 'Senegal':
        num_bts, hours = 1668, 8760
    elif country == 'IvoryCoast':
        num_bts, hours = 1240, 3360
    else:
        num_bts, hours = 10000, 100000

    # Load activity matrices
    adj_matrix = np.genfromtxt(path+'/CDR/staticmetrics/Other/adj_matrix.csv', delimiter=',')
    log_q_matrix = np.genfromtxt(path+'/CDR/staticmetrics/Other/log_q_matrix.csv', delimiter=',')
    where_nan = np.isnan(log_q_matrix)
    log_q_matrix[where_nan] = 0
    deg_vector = np.genfromtxt(path+'/CDR/staticmetrics/Other/degree_vector.csv', delimiter=',')

    ''' Median Degree '''
    # Remove links < median(weight)
    for i in range(num_bts):
        row_col = np.concatenate((adj_matrix[i, :], adj_matrix[:, i]))
        row_col_self = np.delete(row_col, i)
        median_weight = np.median(row_col_self) - 0.1
        adj_matrix[:, i][np.where(adj_matrix[i, :] < median_weight)] = 0
        adj_matrix[i, :][np.where(adj_matrix[:, i] < median_weight)] = 0
    adj_matrix[adj_matrix > 0] = 1
    # Degree matrix after weights < median(weight) removed
    total_deg = np.zeros(num_bts)
    in_deg = np.zeros(num_bts)
    out_deg = np.zeros(num_bts)
    for i in range(num_bts):
        total_deg[i] = np.sum(np.delete(np.concatenate((adj_matrix[i, :], adj_matrix[:, i])), i))
        in_deg[i] = np.sum(adj_matrix[i, :])
        out_deg[i] = np.sum(adj_matrix[:, i])

    ''' Entropy '''
    entropy = np.zeros(num_bts)
    for i in range(num_bts):
        sum_row = np.sum(log_q_matrix[i, :], axis=0)
        entropy[i] = (-1*sum_row) / np.log10(deg_vector[i])
    # np.savetxt(path+'/CDR/staticmetrics/Entropy/total_entropy.csv', entropy, delimiter=',')

    ''' Normalise by Population '''
    # Proportion of each cell tower associated with each administrative region
    intersect_pop = pd.DataFrame(pd.read_csv(path+'/Population/IntersectPop.csv'))
    # Population per adm region
    adm_1_pop = intersect_pop.groupby('Adm_1')['Pop_2010'].sum().reset_index()
    adm_2_pop = intersect_pop.groupby('Adm_2')['Pop_2010'].sum().reset_index()
    adm_3_pop = intersect_pop.groupby('Adm_3')['Pop_2010'].sum().reset_index()
    adm_4_pop = intersect_pop.groupby('Adm_4')['Pop_2010'].sum().reset_index()

    # Only use metric values of cell towers that exist
    cell_tower_adm = pd.DataFrame(pd.read_csv(path+"/CDR/cell_towers/bts_adm_1234.csv"))
    list = np.array(cell_tower_adm['CellTowerID'])
    entropy, total_deg, in_deg, out_deg = entropy[list], total_deg[list], in_deg[list], out_deg[list]
    cell_tower_adm['entropy'] = entropy
    cell_tower_adm['total_deg'], cell_tower_adm['in_deg'], cell_tower_adm['out_deg'] = total_deg, in_deg, out_deg

    # Number of cell towers per adm region
    cell_towers_adm_1 = cell_tower_adm.groupby('ID_1')['ID_1'].count().reset_index(drop=True)
    cell_towers_adm_2 = cell_tower_adm.groupby('ID_2')['ID_2'].count().reset_index(drop=True)
    cell_towers_adm_3 = cell_tower_adm.groupby('ID_3')['ID_3'].count().reset_index(drop=True)
    cell_towers_adm_4 = cell_tower_adm.groupby('ID_4')['ID_4'].count().reset_index(drop=True)

    # Normalise by average population per cell tower in each adm region
    entropy_adm_1 = cell_tower_adm.groupby('ID_1')['entropy'].sum().reset_index()['entropy'] * (adm_1_pop['Pop_2010'] / cell_towers_adm_1)
    entropy_adm_2 = cell_tower_adm.groupby('ID_2')['entropy'].sum().reset_index()['entropy'] * (adm_2_pop['Pop_2010'] / cell_towers_adm_2)
    entropy_adm_3 = cell_tower_adm.groupby('ID_3')['entropy'].sum().reset_index()['entropy'] * (adm_3_pop['Pop_2010'] / cell_towers_adm_3)
    entropy_adm_4 = cell_tower_adm.groupby('ID_4')['entropy'].sum().reset_index()['entropy'] * (adm_4_pop['Pop_2010'] / cell_towers_adm_4)

    # Overall degree (assuming an undirected network)
    total_deg_adm_1 = cell_tower_adm.groupby('ID_1')['total_deg'].sum().reset_index()['total_deg'] * (
    adm_1_pop['Pop_2010'] / cell_towers_adm_1)
    total_deg_adm_2 = cell_tower_adm.groupby('ID_2')['total_deg'].sum().reset_index()['total_deg'] * (
    adm_2_pop['Pop_2010'] / cell_towers_adm_2)
    total_deg_adm_3 = cell_tower_adm.groupby('ID_3')['total_deg'].sum().reset_index()['total_deg'] * (
    adm_3_pop['Pop_2010'] / cell_towers_adm_3)
    total_deg_adm_4 = cell_tower_adm.groupby('ID_4')['total_deg'].sum().reset_index()['total_deg'] * (
    adm_4_pop['Pop_2010'] / cell_towers_adm_4)

    # In-degree (assuming a directed network)
    in_deg_adm_1 = cell_tower_adm.groupby('ID_1')['in_deg'].sum().reset_index()['in_deg'] * (
        adm_1_pop['Pop_2010'] / cell_towers_adm_1)
    in_deg_adm_2 = cell_tower_adm.groupby('ID_2')['in_deg'].sum().reset_index()['in_deg'] * (
        adm_2_pop['Pop_2010'] / cell_towers_adm_2)
    in_deg_adm_3 = cell_tower_adm.groupby('ID_3')['in_deg'].sum().reset_index()['in_deg'] * (
        adm_3_pop['Pop_2010'] / cell_towers_adm_3)
    in_deg_adm_4 = cell_tower_adm.groupby('ID_4')['in_deg'].sum().reset_index()['in_deg'] * (
        adm_4_pop['Pop_2010'] / cell_towers_adm_4)

    # Out-degree (assuming a directed network)
    out_deg_adm_1 = cell_tower_adm.groupby('ID_1')['out_deg'].sum().reset_index()['out_deg'] * (
        adm_1_pop['Pop_2010'] / cell_towers_adm_1)
    out_deg_adm_2 = cell_tower_adm.groupby('ID_2')['out_deg'].sum().reset_index()['out_deg'] * (
        adm_2_pop['Pop_2010'] / cell_towers_adm_2)
    out_deg_adm_3 = cell_tower_adm.groupby('ID_3')['out_deg'].sum().reset_index()['out_deg'] * (
        adm_3_pop['Pop_2010'] / cell_towers_adm_3)
    out_deg_adm_4 = cell_tower_adm.groupby('ID_4')['out_deg'].sum().reset_index()['out_deg'] * (
        adm_4_pop['Pop_2010'] / cell_towers_adm_4)
    
    # np.savetxt('/CDR/staticmetrics/Entropy/entropy_adm_1.csv', entropy_adm_1, delimiter=',')
    # np.savetxt('/CDR/staticmetrics/Entropy/entropy_adm_2.csv', entropy_adm_2, delimiter=',')
    # np.savetxt('/CDR/staticmetrics/Entropy/entropy_adm_3.csv', entropy_adm_3, delimiter=',')
    # np.savetxt('/CDR/staticmetrics/Entropy/entropy_adm_4.csv', entropy_adm_4, delimiter=',')
    # np.savetxt('/CDR/staticmetrics/MedianDegree/total_deg_adm_1.csv', total_deg_adm_1, delimiter=',')
    # np.savetxt('/CDR/staticmetrics/MedianDegree/total_deg_adm_2.csv', total_deg_adm_2, delimiter=',')
    # np.savetxt('/CDR/staticmetrics/MedianDegree/total_deg_adm_3.csv', total_deg_adm_3, delimiter=',')
    # np.savetxt('/CDR/staticmetrics/MedianDegree/total_deg_adm_4.csv', total_deg_adm_4, delimiter=',')
    # np.savetxt('/CDR/staticmetrics/MedianDegree/in_deg_adm_1.csv', in_deg_adm_1, delimiter=',')
    # np.savetxt('/CDR/staticmetrics/MedianDegree/in_deg_adm_2.csv', in_deg_adm_2, delimiter=',')
    # np.savetxt('/CDR/staticmetrics/MedianDegree/in_deg_adm_3.csv', in_deg_adm_3, delimiter=',')
    # np.savetxt('/CDR/staticmetrics/MedianDegree/in_deg_adm_4.csv', in_deg_adm_4, delimiter=',')
    # np.savetxt('/CDR/staticmetrics/MedianDegree/out_deg_adm_1.csv', out_deg_adm_1, delimiter=',')
    # np.savetxt('/CDR/staticmetrics/MedianDegree/out_deg_adm_2.csv', out_deg_adm_2, delimiter=',')
    # np.savetxt('/CDR/staticmetrics/MedianDegree/out_deg_adm_3.csv', out_deg_adm_3, delimiter=',')
    # np.savetxt('/CDR/staticmetrics/MedianDegree/out_deg_adm_4.csv', out_deg_adm_4, delimiter=',')












