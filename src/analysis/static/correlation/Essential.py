import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

# Read all CDR data - activity (volume and duration, maybe a ratio of the two?, entropy.csv, introversion, median..
cdr_activity = pd.DataFrame(pd.read_csv('Data/IvoryCoast/CDR/Metrics/activity_bts_level.csv'))

# # Get rid of all inactive towers (this leaves me with exactly 1200 towers)
# inactive = cdr_activity[cdr_activity['Activity'] < 12400]['CellTowerID']
# for i in inactive:
#     cdr_activity = cdr_activity[cdr_activity['CellTowerID'] != i]
#
# # Aggregate to Adm levels
# voronoi_proportion = pd.DataFrame(pd.read_csv('Data/IvoryCoast/Essential/VoronoiPopPerSubpref.csv'))
# # Adm level 4
# activity_adm_4 = np.zeros(191)
# for i in range(1, 192):
#     vor_prop = voronoi_proportion[voronoi_proportion['clustID'] == i]
#     for j in vor_prop['cellID']:
#         if cdr_activity['Activity'][cdr_activity['CellTowerID'] == j].empty:
#             pass
#         else:
#             activity_adm_4[i-1] += int(cdr_activity['Activity'][cdr_activity['CellTowerID'] == j])
#
# adm_4 = pd.DataFrame()
# adm_4['Activity'] = np.array(activity_adm_4)
# adm_4['Adm_4'] = range(1, 192)
#
# adm_4.to_csv('Data/IvoryCoast/Essential/proportional_activity.csv', index=None)

# Activity proportional to appropriate adm level
proportional_activity = pd.DataFrame(pd.read_csv('Data/IvoryCoast/Essential/proportional_activity.csv'))
prop_act_adm_1 = proportional_activity.groupby('Adm_1')['Activity'].sum().reset_index()
prop_act_adm_2 = proportional_activity.groupby('Adm_2')['Activity'].sum().reset_index()
prop_act_adm_3 = proportional_activity.groupby('Adm_3')['Activity'].sum().reset_index()
prop_act_adm_4 = proportional_activity.groupby('Adm_4')['Activity'].sum().reset_index()

# Normalise CDR data by the population of the adm level (again, 4 different populations for 4 different adm's)
dhs_population = pd.DataFrame(pd.read_csv('Data/IvoryCoast/Essential/dhs_clusters.csv',
                                          usecols=['DHSClust', 'Pop_1km', 'Adm_1',
                                                   'Adm_2', 'Adm_3', 'Adm_4']))
dhs_pop_adm_1 = dhs_population.groupby('Adm_1')['Pop_1km'].sum().reset_index()
dhs_pop_adm_2 = dhs_population.groupby('Adm_2')['Pop_1km'].sum().reset_index()
dhs_pop_adm_3 = dhs_population.groupby('Adm_3')['Pop_1km'].sum().reset_index()
dhs_pop_adm_4 = dhs_population.groupby('Adm_4')['Pop_1km'].sum().reset_index()

# Some locations are missing - remove data from those locations (as it would be inaccurate to use them)
for i in np.setdiff1d(prop_act_adm_3['Adm_3'], dhs_pop_adm_3['Adm_3']):
    prop_act_adm_3 = prop_act_adm_3[prop_act_adm_3['Adm_3'] != i]
for i in np.setdiff1d(prop_act_adm_4['Adm_4'], dhs_pop_adm_4['Adm_4']):
    prop_act_adm_4 = prop_act_adm_4[prop_act_adm_4['Adm_4'] != i]

# Estimated population per adm region
prop_act_adm_1['Pop'] = np.array(dhs_pop_adm_1['Pop_1km'])
prop_act_adm_2['Pop'] = np.array(dhs_pop_adm_2['Pop_1km'])
prop_act_adm_3['Pop'] = np.array(dhs_pop_adm_3['Pop_1km'])
prop_act_adm_4['Pop'] = np.array(dhs_pop_adm_4['Pop_1km'])

# Activity per person
prop_act_adm_1['Activity_PP'] = np.array(prop_act_adm_1['Activity']) / np.array(dhs_pop_adm_1['Pop_1km'])
prop_act_adm_2['Activity_PP'] = np.array(prop_act_adm_2['Activity']) / np.array(dhs_pop_adm_2['Pop_1km'])
prop_act_adm_3['Activity_PP'] = np.array(prop_act_adm_3['Activity']) / np.array(dhs_pop_adm_3['Pop_1km'])
prop_act_adm_4['Activity_PP'] = np.array(prop_act_adm_4['Activity']) / np.array(dhs_pop_adm_4['Pop_1km'])

# DHS Wealth
dhs_wealth = pd.DataFrame(pd.read_csv('Data/IvoryCoast/Essential/dhs_clusters.csv',
                                      usecols=['DHSClust', 'Z_Med', 'Adm_1',
                                               'Adm_2', 'Adm_3', 'Adm_4']))
wealth_adm_1 = dhs_wealth.groupby(by='Adm_1')['Z_Med'].mean().reset_index()
wealth_adm_2 = dhs_wealth.groupby(by='Adm_2')['Z_Med'].mean().reset_index()
wealth_adm_3 = dhs_wealth.groupby(by='Adm_3')['Z_Med'].mean().reset_index()
wealth_adm_4 = dhs_wealth.groupby(by='Adm_4')['Z_Med'].mean().reset_index()

dhs_poverty = pd.DataFrame(pd.read_csv('Data/IvoryCoast/Essential/dhs_clusters.csv',
                                      usecols=['DHSClust', 'Poverty', 'Adm_1',
                                               'Adm_2', 'Adm_3', 'Adm_4']))
poverty_adm_1 = dhs_poverty.groupby(by='Adm_1')['Poverty'].mean().reset_index()
poverty_adm_2 = dhs_poverty.groupby(by='Adm_2')['Poverty'].mean().reset_index()
poverty_adm_3 = dhs_poverty.groupby(by='Adm_3')['Poverty'].mean().reset_index()
poverty_adm_4 = dhs_poverty.groupby(by='Adm_4')['Poverty'].mean().reset_index()

# Pearson correlation of all measures

print pearsonr(prop_act_adm_1['Activity_PP'].iloc[1:], wealth_adm_1['Z_Med'].iloc[1:])
print pearsonr(prop_act_adm_2['Activity_PP'].iloc[1:], wealth_adm_2['Z_Med'].iloc[1:])
print pearsonr(prop_act_adm_3['Activity_PP'].iloc[1:], wealth_adm_3['Z_Med'].iloc[1:])
print pearsonr(prop_act_adm_4['Activity_PP'].iloc[1:], wealth_adm_4['Z_Med'].iloc[1:])
plt.scatter(prop_act_adm_1['Activity_PP'].iloc[1:], wealth_adm_1['Z_Med'].iloc[1:])
plt.show()
plt.scatter(prop_act_adm_2['Activity_PP'].iloc[1:], wealth_adm_2['Z_Med'].iloc[1:])
plt.show()
plt.scatter(prop_act_adm_3['Activity_PP'].iloc[1:], wealth_adm_3['Z_Med'].iloc[1:])
plt.show()
plt.scatter(prop_act_adm_4['Activity_PP'].iloc[1:], wealth_adm_4['Z_Med'].iloc[1:])
plt.show()

print pearsonr(prop_act_adm_1['Activity_PP'], poverty_adm_1['Poverty'])
print pearsonr(prop_act_adm_2['Activity_PP'], poverty_adm_2['Poverty'])
print pearsonr(prop_act_adm_3['Activity_PP'], poverty_adm_3['Poverty'])
print pearsonr(prop_act_adm_4['Activity_PP'], poverty_adm_4['Poverty'])
plt.scatter(prop_act_adm_1['Activity_PP'].iloc[1:], poverty_adm_1['Poverty'].iloc[1:])
plt.show()
plt.scatter(prop_act_adm_2['Activity_PP'].iloc[1:], poverty_adm_2['Poverty'].iloc[1:])
plt.show()
plt.scatter(prop_act_adm_3['Activity_PP'].iloc[1:], poverty_adm_3['Poverty'].iloc[1:])
plt.show()
plt.scatter(prop_act_adm_4['Activity_PP'].iloc[1:], poverty_adm_4['Poverty'].iloc[1:])
plt.show()



# # HIV Data
# hiv_data = pd.DataFrame(pd.read_csv('Data/IvoryCoast/DHS/Extracted/hiv_per_dhs_clust.csv'))
# hiv_data = pd.DataFrame(hiv_data.groupby(by='DHSClustID')['Result'].sum().reset_index())
# hiv_data.to_csv('Data/IvoryCoast/Essential/HIV.csv', index=None)
#
# print hiv_data














