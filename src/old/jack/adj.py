import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# adj_matrix = np.genfromtxt('/Users/JackShipway/Desktop/UCLProject/Data/IvoryCoast/'
#                                       'CDR/StaticMetrics/Other/adj_matrix_civ.csv', delimiter=',')

# activity = []
# for i in range(1240):
#     vin = np.sum(adj_matrix[:, i])
#     vout = np.sum(adj_matrix[i, :])
#     self = adj_matrix[i, i]
#     activity.append(vin + vout - self)

act = pd.DataFrame(pd.read_csv('activity.csv'))
act_1 = np.array(act.groupby('Adm_1')['Activity'].sum())
act_2 = np.array(act.groupby('Adm_2')['Activity'].sum())
act_3 = np.array(act.groupby('Adm_3')['Activity'].sum())
pop = pd.DataFrame(pd.read_csv('/Users/JackShipway/Desktop/UCLProject/Data/IvoryCoast/Population/IntersectPop.csv'))
pop_1 = np.array(pop.groupby('Adm_1')['Pop_2010'].sum())
pop_2 = np.array(pop.groupby('Adm_2')['Pop_2010'].sum())
pop_3 = np.array(pop.groupby('Adm_3')['Pop_2010'].sum())

malaria = pd.DataFrame(pd.read_csv('/Users/JackShipway/Desktop/UCLProject/Data/IvoryCoast/DHS/Metrics/Malaria/'
                                   'bloodtest.csv'))
dhs = pd.DataFrame(pd.read_csv('/Users/JackShipway/Desktop/UCLProject/Data/IvoryCoast/DHS/SPSS/DHS_Adm_1234.csv'))

for i in np.setdiff1d(malaria['DHSClust'], dhs['DHSClust']):
    malaria = malaria[malaria['DHSClust'] != i]


# dhs['malaria'] = np.array(malaria['Malaria']*100000)
# mal_mean = np.array(dhs.groupby('Adm_1')['malaria'].mean())
# mal_med = np.array(dhs.groupby('Adm_1')['malaria'].median())

malaria = pd.DataFrame(pd.read_csv('/Users/JackShipway/Desktop/UCLProject/Data/IvoryCoast/DHS/Metrics/DHSData.csv'))
mal_mean = np.array(malaria.groupby('Adm_3')['MalariaPerPop'].mean())
mal_med = np.array(malaria.groupby('Adm_3')['MalariaPerPop'].median())
act_3 = np.divide(act_3[1:], pop_3)

print act_3, len(act_3), pop_3, len(pop_3)

plt.scatter(act_3, mal_mean)
plt.show()

# outlier = np.intersect1d(np.where(mal_med < 1), np.where(act_2 < 140))
# act_2 = act_2[outlier]
#
# mal_med = mal_med[outlier]
# mal_mean = mal_mean[outlier]
# a = np.log(act_2)
# b = np.log(mal_med+1)
#
# print pearsonr((a-np.median(a))/np.std(a), (b-np.median(b))/np.std(b))
# plt.scatter(a, b)
# plt.show()
# plt.scatter((a-np.median(a))/np.std(a), (b-np.median(b))/np.std(b))
# plt.show()

# a = np.divide(act_1[1:], pop_1)
# # b = mal_mean
# c = mal_med
#
# # print pearsonr(a/max(a), b/max(b))
# print pearsonr(a/max(a), c/max(c))
#
# # plt.scatter(a/max(a), b/max(b))
# plt.scatter(a/max(a), c/max(c), c='r')
# plt.show()






