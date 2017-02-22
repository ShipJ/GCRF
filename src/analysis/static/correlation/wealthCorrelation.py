import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
#
# wealth_index = pd.DataFrame(pd.read_csv('Data/IvoryCoast/DHS/Extracted/dhs_cluster_wealth.csv'))
# clust_adm_1234 = pd.DataFrame(pd.read_csv('Data/IvoryCoast/CDR/CellTowerInfo/ICSubpref_Adm_1234.csv'))
#
# clust_adm_1234['Poverty'] = np.array(wealth_index['poverty_rate'])
#
#
# wealth_adm_1 = clust_adm_1234.groupby(by='ID_1')['Poverty'].mean().reset_index()
# wealth_adm_2 = clust_adm_1234.groupby(by='ID_2')['Poverty'].mean().reset_index()
# wealth_adm_3 = clust_adm_1234.groupby(by='ID_3')['Poverty'].mean().reset_index()
# wealth_adm_4 = clust_adm_1234.groupby(by='ID_4')['Poverty'].mean().reset_index()
#
activity_adm_1234 = pd.DataFrame(pd.read_csv('Data/IvoryCoast/CDR/metrics/activity_bts_level.csv'))
print activity_adm_1234
print np.sum(activity_adm_1234['Activity'])


#
# activity_adm_1 = activity_adm_1234.groupby(by='ID_1')['Activity'].sum().reset_index()
# activity_adm_2 = activity_adm_1234.groupby(by='ID_2')['Activity'].sum().reset_index()
# activity_adm_3 = activity_adm_1234.groupby(by='ID_3')['Activity'].sum().reset_index()
# activity_adm_4 = activity_adm_1234.groupby(by='ID_4')['Activity'].sum().reset_index()
#
#
# clust_pop = pd.DataFrame(pd.read_csv('Data/IvoryCoast/DHS/Extracted/DHSClusterPop.csv'))
# clust_pop['ID_4'] = np.array(clust_adm_1234['ID_4'])
# clust_pop = clust_pop.groupby(by='ID_4')['Pop'].sum().reset_index()
#
# missing1 = np.setdiff1d(clust_pop['ID_4'], activity_adm_4['ID_4'])
# missing2 = np.setdiff1d(activity_adm_4['ID_4'], clust_pop['ID_4'])
#
#
# for i in missing2:
#     activity_adm_4 = activity_adm_4[activity_adm_4['ID_4'] != i]
#
# for j in missing1:
#     clust_pop = clust_pop[clust_pop['ID_4'] != j]
#     wealth_adm_4 = wealth_adm_4[wealth_adm_4['ID_4'] != j]
#
# activity_adm_4['Pop'] = np.array(clust_pop['Pop'])
# activity_adm_4['Activity_pp'] = activity_adm_4['Activity'] / activity_adm_4['Pop']
#
# activity_adm_4['Wealth'] = np.array(wealth_adm_4['Poverty'])
#
# print pearsonr(activity_adm_4['Activity_pp'], activity_adm_4['Wealth'])
#



