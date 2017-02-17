import pandas as pd
import numpy as np
import sys
import sklearn.metrics as sk

path = '/Users/JackShipway/Desktop/UCLProject'

master = pd.DataFrame(pd.read_csv(path+'/Data/IvoryCoast/Master.csv'))

entropy = pd.DataFrame(pd.read_csv(path+'/GCRF-Project/IvoryCoast/entropy.csv'))
med_degree = pd.DataFrame(pd.read_csv(path+'/GCRF-Project/IvoryCoast/med_degree.csv'))
introversion = pd.DataFrame(pd.read_csv(path+'/GCRF-Project/IvoryCoast/introversion.csv'))

all = pd.DataFrame()
all['CellTowerID'] = range(1, 1240)
all['Entropy'] = entropy
all['Introversion'] = introversion
all['Med_degree'] = med_degree

cdr_adm = pd.DataFrame(pd.read_csv(path+'/Data/IvoryCoast/CDR/CellTowers/CDR_Adm_1234.csv',
                                   usecols=['CellTowerID', 'Adm_1', 'Adm_2', 'Adm_3', 'Adm_4']))


a = all.merge(cdr_adm, on='CellTowerID', how='outer')
cdr = a.groupby('Adm_4')['Entropy', 'Introversion', 'Med_degree'].mean().reset_index()

master = pd.DataFrame(master.merge(cdr, on='Adm_4', how='outer'))
master.to_csv(path+'/Data/IvoryCoast/CDR/new.csv', index=None)
