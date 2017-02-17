import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import pearsonr

# CDR metrics
activity_1 = pd.DataFrame(pd.read_csv('activity_1.csv'))
entropy_1 = pd.DataFrame(pd.read_csv('entropy_1.csv'))
med_degree_1 = pd.DataFrame(pd.read_csv('med_degree_1.csv'))

dhs = pd.DataFrame(pd.read_csv('/Users/JackShipway/Desktop/UCLProject/Data/IvoryCoast/DHS/Metrics/DHSData.csv'))
malaria_1 = pd.DataFrame(pd.read_csv('/Users/JackShipway/Desktop/UCLProject/Data/'
                                        'IvoryCoast/DHS/Metrics/Malaria/bloodtest.csv'))
for i in np.setdiff1d(malaria_1['DHSClust'], dhs['DHSClust']):
    malaria_1 = malaria_1[malaria_1['DHSClust'] != i]
dhs['malaria_1'] = np.array(malaria_1['Malaria'])

malaria_1 = dhs.groupby('Adm_1')['malaria_1'].mean().reset_index()


# # DHS metrics
# malaria_1 = pd.DataFrame(pd.read_csv('/Users/JackShipway/Desktop/UCLProject/Data/'
#                                         'IvoryCoast/DHS/Metrics/Malaria/bloodtest.csv'))
# dhs = pd.DataFrame(pd.read_csv('/Users/JackShipway/Desktop/UCLProject/Data/IvoryCoast/DHS/Metrics/DHSData.csv'))

# for i in np.setdiff1d(malaria_3['DHSClust'], dhs['DHSClust']):
#     malaria_3 = malaria_3[malaria_3['DHSClust'] != i]
# dhs['malaria_1'] = np.array(malaria_1['Malaria'])
#
# malaria_3 = dhs.groupby('Adm_3')['malaria_3'].mean().reset_index()
#
# for j in np.setdiff1d(activity_3['Adm_3'], malaria_3['Adm_3']):
#     activity_3 = activity_3[activity_3['Adm_3'] != j]
#     entropy_3 = entropy_3[entropy_3['Adm_3'] != j]
#
# for k in np.setdiff1d(malaria_3['Adm_3'], activity_3['Adm_3']):
#     malaria_3 = malaria_3[malaria_3['Adm_3'] != k]

# poverty_3 = dhs.groupby('Adm_3')['Poverty'].mean().reset_index()
# for j in np.setdiff1d(activity_3['Adm_3'], poverty_3['Adm_3']):
#     activity_3 = activity_3[activity_3['Adm_3'] != j]
#     entropy_3 = entropy_3[entropy_3['Adm_3'] != j]
#     med_degree_3 = med_degree_3[med_degree_3['Adm_3'] != j]
#
# for k in np.setdiff1d(poverty_3['Adm_3'], activity_3['Adm_3']):
#     poverty_3 = poverty_3[poverty_3['Adm_3'] != k]

pop = pd.DataFrame(pd.read_csv('/Users/JackShipway/Desktop/UCLProject/Data/IvoryCoast/Population/IntersectPop.csv'))
pop = pop.groupby('Adm_1')['Pop_2010'].sum()
print pop
print activity_1

a = np.divide(activity_1['Activity'], pop)
b = np.array(malaria_1['malaria_1'])*100000
print pearsonr(a, b)
plt.scatter(a, b)
plt.show()

c = np.log(entropy_1['Entropy'][1:])
d = np.array(malaria_1['malaria_1'][1:])*100000
print pearsonr(c, d)
plt.scatter(c, d)
plt.show()

e = np.log(med_degree_1['Med_degree'][1:])
f = np.array(malaria_1['malaria_1'][1:])*100000
print pearsonr(e, f)
plt.scatter(e, f)
plt.show()
