import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import linregress
from scipy.stats import pearsonr
import math
import statsmodels.api as sm
import sys

path = '/Users/JackShipway/Desktop/UCLProject/Data/IvoryCoast'
activity_1 = pd.DataFrame(pd.read_csv(path+'/CDR/StaticMetrics/Activity/activity_adm1.csv'))
activity_2 = pd.DataFrame(pd.read_csv(path+'/CDR/StaticMetrics/Activity/activity_adm2.csv'))
activity_3 = pd.DataFrame(pd.read_csv(path+'/CDR/StaticMetrics/Activity/activity_adm3.csv'))
activity_4 = pd.DataFrame(pd.read_csv(path+'/CDR/StaticMetrics/Activity/activity_adm4.csv'))
dhs = pd.DataFrame(pd.read_csv(path+'/DHS/Metrics/DHSData.csv'))
malaria_1 = dhs.groupby('Adm_1')['MalariaPerPop'].mean().reset_index()
malaria_2 = dhs.groupby('Adm_2')['MalariaPerPop'].mean().reset_index()
malaria_3 = dhs.groupby('Adm_3')['MalariaPerPop'].mean().reset_index()
malaria_4 = dhs.groupby('Adm_4')['MalariaPerPop'].mean().reset_index()

for i in np.setdiff1d(activity_3['Adm_3'], malaria_3['Adm_3']):
    activity_3 = activity_3[activity_3['Adm_3'] != i]

for i in np.setdiff1d(activity_4['Adm_4'], malaria_4['Adm_4']):
    activity_4 = activity_4[activity_4['Adm_4'] != i]

for i in np.setdiff1d(malaria_4['Adm_4'], activity_4['Adm_4']):
    malaria_4 = malaria_4[malaria_4['Adm_4'] != i]


a = np.array(activity_4['Vol'])
b = np.array(malaria_4['MalariaPerPop']*100)

plt.scatter(a, b)
plt.show()

sys.exit()

outliers = np.intersect1d(np.intersect1d(np.where(a > 0), np.where(b > 0)),
                          np.intersect1d(np.where(a < 100), np.where(b < 0.2)))
a = a[outliers]
b = b[outliers]

c = np.log(a)
d = np.log(b)

e = (c - np.mean(c)) / np.std(c)
f = (d - np.mean(d)) / np.std(d)

plt.scatter(e, f)
plt.show()

# plt.scatter(e, f)
# plt.grid(b=True, which='both', color='0.65', linestyle='-')
# plt.show()
#
# plt.plot(sorted(e))
# plt.plot(sorted(f))
# plt.show()

X = sm.add_constant(e, prepend=False)
y = f

model = sm.OLS(y, X)

results = model.fit()

print results.summary()










