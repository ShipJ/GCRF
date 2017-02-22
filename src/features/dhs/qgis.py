import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

path = '/Users/JackShipway/Desktop/UCLProject/Data/IvoryCoast'

dhs = pd.DataFrame(pd.read_csv(path+'/DHS/SPSS/DHS_Adm_1234.csv'))
bloodtest = pd.DataFrame(pd.read_csv(path+'/DHS/metrics/Malaria/bloodtest.csv')).dropna()
rapidtest = pd.DataFrame(pd.read_csv(path+'/DHS/metrics/Malaria/rapidtest.csv')).dropna()
for i in np.setdiff1d(range(1, 352), dhs['DHSClust']):
    bloodtest = bloodtest[bloodtest['DHSClust'] != i]
    rapidtest = rapidtest[rapidtest['DHSClust'] != i]
dhs['bloodtest'] = np.array(bloodtest['Malaria'])
dhs['rapidtest'] = np.array(rapidtest['Malaria'])

blood_1 = dhs.groupby('ID_1')['bloodtest'].mean().reset_index()
blood_2 = dhs.groupby('ID_2')['bloodtest'].mean().reset_index()
blood_3 = dhs.groupby('ID_3')['bloodtest'].mean().reset_index()
blood_4 = dhs.groupby('ID_4')['bloodtest'].mean().reset_index()
blood_4.to_csv('/Users/JackShipway/Desktop/blood_4.csv', index=None)

rapid_1 = dhs.groupby('ID_1')['rapidtest'].mean().reset_index()
rapid_2 = dhs.groupby('ID_2')['rapidtest'].mean().reset_index()
rapid_3 = dhs.groupby('ID_3')['rapidtest'].mean().reset_index()
rapid_4 = dhs.groupby('ID_4')['rapidtest'].mean().reset_index()

introversion_1 = pd.DataFrame(pd.read_csv(path+'/CDR/staticmetrics/Introversion/introversion_adm1.csv'))
introversion_2 = pd.DataFrame(pd.read_csv(path+'/CDR/staticmetrics/Introversion/introversion_adm2.csv'))
introversion_3 = pd.DataFrame(pd.read_csv(path+'/CDR/staticmetrics/Introversion/introversion_adm3.csv'))
introversion_4 = pd.DataFrame(pd.read_csv(path+'/CDR/staticmetrics/Introversion/introversion_adm4.csv'))

for i in np.setdiff1d(introversion_3['Adm_3'], blood_3['ID_3']):
    introversion_3 = introversion_3[introversion_3['Adm_3'] != i]
for j in np.setdiff1d(blood_3['ID_3'], introversion_3['Adm_3']):
    blood_3 = blood_3[blood_3['ID_3'] != j]
    rapid_3 = rapid_3[rapid_3['ID_3'] != j]
    
for i in np.setdiff1d(introversion_4['Adm_4'], blood_4['ID_4']):
    introversion_4 = introversion_4[introversion_4['Adm_4'] != i]
for j in np.setdiff1d(blood_4['ID_4'], introversion_4['Adm_4']):
    blood_4 = blood_4[blood_4['ID_4'] != j]
    rapid_4 = rapid_4[rapid_4['ID_4'] != j]

a = introversion_1['Introversion']
b = rapid_1['rapidtest']

c = introversion_2['Introversion']
d = rapid_2['rapidtest']

e = introversion_3['Introversion']
f = rapid_3['rapidtest'] 

g = introversion_4['Introversion']
h = rapid_4['rapidtest']

print pearsonr(a, b)
print pearsonr(c, d)
print pearsonr(e, f)
print pearsonr(g, h)

plt.scatter((a-np.mean(a))/np.std(a), (b-np.mean(b))/np.std(b))
plt.show()
plt.scatter((c-np.mean(c))/np.std(c), (d-np.mean(d))/np.std(d))
plt.show()
plt.scatter((e-np.mean(e))/np.std(e), (f-np.mean(f))/np.std(f))
plt.show()
plt.scatter((g-np.mean(g))/np.std(g), (h-np.mean(h))/np.std(h))
plt.show()






