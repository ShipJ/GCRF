import pandas as pd
import numpy as np
import sys

dhs = pd.DataFrame(pd.read_csv('../../../data/final/civ/dhs.csv'))

dist = pd.DataFrame(pd.read_csv('../../../data/processed/civ/distance/dist_matrix_adm4.csv'))


spatial_lag = []
for i in dhs['Adm_4']:
    response_i = dhs[dhs['Adm_4'] == i]

    a = dist[dist['Source'] == i]
    a_dist = a[a['Distance(km)']>0]
    sum_weights = np.sum(np.reciprocal(a_dist['Distance(km)'] ** 2))

    sum_lag = 0
    for j in a['Target']:
        b = a[a['Target'] == j]
        sum_lag += ((np.array(b['Distance(km)'])[0] / float(sum_weights)) * np.array(response_i['HealthAccessDifficulty'])[0])

    spatial_lag.append(sum_lag)

a = pd.DataFrame(spatial_lag)
print a.to_csv('../../../data/final/civ/spatial_lag.csv', index=None)








