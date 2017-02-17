import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import inf
from scipy.stats import pearsonr
import sys

path = '/Users/JackShipway/Desktop/UCLProject/Data/IvoryCoast'

# malaria = pd.DataFrame(pd.read_csv(path+'/DHS/Metrics/Malaria/malaria_feb.csv'))
#
# dhs = pd.DataFrame(pd.read_csv(path+'/DHS/Metrics/DHSData.csv'))
# malaria_4 = dhs.groupby('Adm_4')['MalariaPerPop'].mean().reset_index()
#
# malaria = malaria.applymap(lambda x: -1 if isinstance(x, basestring) and x.isspace() else int(x))
#
# # Number of houses within the cluster
# num_houses = np.array(malaria.groupby('DHSClust')['HouseID'].count())
# # Number of members per cluster
# num_members = np.array(malaria.groupby('DHSClust')['NumMembers'].sum())
#
# bloodtest = malaria.iloc[:, 3:39]
# bloodtest.loc[:, 'DHS'] = malaria['DHSClust']
# bloodtest.loc[:, 'NumMembers'] = malaria['NumMembers']
#
# rapidtest = malaria.iloc[:, 39:]
# rapidtest.loc[:, 'DHS'] = malaria['DHSClust']
# rapidtest.loc[:, 'NumMembers'] = malaria['NumMembers']
#
# blood_pos, blood_neg, blood_tot, rapid_pos, rapid_neg, rapid_tot = [], [], [], [], [], []
#
# for i in np.setdiff1d(range(1, 353), [225]):
#     # Get data corresponding to blood/rapid test results (rows: households, columns: individuals)
#     blood = bloodtest[bloodtest['DHS'] == i].iloc[:, 0:36]
#     rapid = rapidtest[rapidtest['DHS'] == i].iloc[:, 0:36]
#     # Store key-value pairs in dict: -1: not sampled, 0: negative, 1: positive, {6,7,8,9}: test inconclusive
#     blood_freq = pd.DataFrame(blood.stack().value_counts(), columns=['count']).reset_index().set_index('index').to_dict()['count']
#     rapid_freq = pd.DataFrame(rapid.stack().value_counts(), columns=['count']).reset_index().set_index('index').to_dict()['count']
#     # Sum positive, negative, and total sampled for blood/rapid tests
#     blood_pos.append(sum(blood_freq[k] for k in blood_freq.keys() if k == 1))
#     blood_neg.append(sum(blood_freq[k] for k in blood_freq.keys() if k == 0))
#     blood_tot.append(sum(blood_freq[k] for k in blood_freq.keys() if k >= 0))
#     rapid_pos.append(sum(rapid_freq[k] for k in rapid_freq.keys() if k == 1))
#     rapid_neg.append(sum(rapid_freq[k] for k in rapid_freq.keys() if k == 0))
#     rapid_tot.append(sum(rapid_freq[k] for k in rapid_freq.keys() if k >= 0))
#
# malaria = pd.DataFrame()
# malaria['DHSClust'] = np.setdiff1d(range(1, 353), [225])
# malaria['Blood_pos'], malaria['Blood_neg'], malaria['Blood_tot'] = blood_pos, blood_neg, blood_tot
# malaria['Rapid_pos'], malaria['Rapid_neg'], malaria['Rapid_tot'] = rapid_pos, rapid_neg, rapid_tot
# malaria['Num_members'], malaria['Num_houses'] = num_members, num_houses
#
# DHS_Adm = pd.DataFrame(pd.read_csv(path+'/DHS/SPSS/DHS_Adm_1234.csv'))
#
# malaria['Adm_1'] = DHS_Adm['Adm_1']
# malaria['Adm_2'] = DHS_Adm['Adm_2']
# malaria['Adm_3'] = DHS_Adm['Adm_3']
# malaria['Adm_4'] = DHS_Adm['Adm_4']
#
# malaria = malaria.groupby('Adm_4')['Blood_pos', 'Blood_neg', 'Blood_tot', 'Rapid_pos',
#                                'Rapid_neg', 'Rapid_tot', 'Num_members', 'Num_houses'].sum().reset_index()
#
#
# malaria.to_csv(path+'/DHS/Metrics/Malaria/master_malaria.csv', index=None)


master = pd.DataFrame(pd.read_csv(path+'/Master.csv'))
malaria = pd.DataFrame(pd.read_csv(path+'/DHS/Metrics/Malaria/master_malaria.csv'))

master =  pd.DataFrame(master.merge(malaria, on='Adm_4', how='outer'))
master.to_csv(path+'/DHS/Metrics/Malaria/now3.csv', index=None)


#
# # # master = pd.DataFrame(pd.read_csv(path+'/DHS/Metrics/Malaria/now.csv'))
# # #
# # # master.to_csv(path+'/Master.csv', index=None)




