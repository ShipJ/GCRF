import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

path = '/Users/JackShipway/Desktop/UCLProject/Data/IvoryCoast'

dhs = pd.DataFrame(pd.read_csv(path+'/DHS/SPSS/DHS_Adm_1234.csv'))

wealth = pd.DataFrame(pd.read_csv(path+'/DHS/Metrics/wealth.csv'))

for index, row in wealth.iterrows():
    if row.Index == 1:
        row.Index = 1
    else:
        row.Index = 0


dhs_clusters = np.unique(np.array(wealth['DHSClust']))
houses = np.array(wealth.groupby('DHSClust')['DHSClust'].count())
poor = np.array(wealth.groupby('DHSClust')['Index'].sum())
mean = np.array(wealth.groupby('DHSClust')['Score'].mean())
median = np.array(wealth.groupby('DHSClust')['Score'].median())
std = np.array(wealth.groupby('DHSClust')['Score'].std())
z_score = np.divide((median - np.mean(median)), std)

clust_poor = np.divide(poor, np.array(houses, dtype=float))


wealth = pd.DataFrame()
wealth['DHSClust'] = np.array(dhs_clusters)
wealth['poor'] = np.array(clust_poor)
wealth['mean'] = np.array(mean)
wealth['median'] = np.array(median)
wealth['std'] = np.array(std)
wealth['z_score'] = np.array(z_score)

for i in np.setdiff1d(wealth['DHSClust'], dhs['DHSClust']):
    wealth = wealth[wealth['DHSClust'] != i]
for j in np.setdiff1d(dhs['DHSClust'], wealth['DHSClust']):
    dhs = dhs[dhs['DHSClust'] != j]

dhs['poverty_rate'] = np.array(wealth['poor'])
dhs['mean_wealth'] = np.array(wealth['mean'])
dhs['z_score'] = np.array(wealth['z_score'])

poverty_1 = dhs.groupby('ID_1')['poverty_rate'].mean().reset_index()
poverty_2 = dhs.groupby('ID_2')['poverty_rate'].mean().reset_index()
poverty_3 = dhs.groupby('ID_3')['poverty_rate'].mean().reset_index()
poverty_4 = dhs.groupby('ID_4')['poverty_rate'].mean().reset_index()

z_score_1 = dhs.groupby('ID_1')['z_score'].mean().reset_index()
z_score_2 = dhs.groupby('ID_2')['z_score'].mean().reset_index()
z_score_3 = dhs.groupby('ID_3')['z_score'].mean().reset_index()
z_score_4 = dhs.groupby('ID_4')['z_score'].mean().reset_index()

activity_1 = pd.DataFrame(pd.read_csv(path+'/CDR/staticmetrics/Activity/activity_adm1.csv'))
activity_2 = pd.DataFrame(pd.read_csv(path+'/CDR/staticmetrics/Activity/activity_adm2.csv'))
activity_3 = pd.DataFrame(pd.read_csv(path+'/CDR/staticmetrics/Activity/activity_adm3.csv'))
activity_4 = pd.DataFrame(pd.read_csv(path+'/CDR/staticmetrics/Activity/activity_adm4.csv'))

print pearsonr(z_score_1['z_score'], activity_1['Vol'])
print pearsonr(z_score_2['z_score'], activity_2['Vol'])

plt.scatter(z_score_1['z_score'], activity_1['Vol'])
plt.show()
plt.scatter(z_score_2['z_score'], activity_2['Vol'])
plt.show()
