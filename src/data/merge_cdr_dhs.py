import pandas as pd
import numpy as np
import sys

''' combine cdr metrics into one dataframe'''
activity = pd.DataFrame(pd.read_csv('../../data/processed/%s/cdr/staticmetrics/new/total_activity.csv'))
entropy = pd.DataFrame(pd.read_csv('../../data/processed/%s/cdr/staticmetrics/new/total_activity.csv'))
med_degree = pd.DataFrame(pd.read_csv('../../data/processed/%s/cdr/staticmetrics/new/total_activity.csv'))
graph_metrics = pd.DataFrame(pd.read_csv('../../data/processed/%s/cdr/staticmetrics/new/total_activity.csv'))
introversion = pd.DataFrame(pd.read_csv('../../data/processed/%s/cdr/staticmetrics/new/total_activity.csv'))
# introversion = pd.DataFrame(pd.read_csv('../../data/processed/%s/cdr/staticmetrics/new/total_activity.csv'))




master = pd.DataFrame()
master['CellTowerID'] = range(1, 1240)
all['Entropy'] = entropy
all['Introversion'] = introversion
all['Med_degree'] = med_degree

cdr_adm = pd.DataFrame(pd.read_csv(path+'/Data/IvoryCoast/CDR/celltowers/CDR_Adm_1234.csv',
                                   usecols=['CellTowerID', 'Adm_1', 'Adm_2', 'Adm_3', 'Adm_4']))
''' merge adm regions'''
a = all.merge(cdr_adm, on='CellTowerID', how='outer')
cdr = a.groupby('Adm_4')['Entropy', 'Introversion', 'Med_degree'].mean().reset_index()

master = pd.DataFrame(master.merge(cdr, on='Adm_4', how='outer'))
master.to_csv(path+'/Data/IvoryCoast/CDR/new.csv', index=None)

''' merge populations and distance data '''

''' merge dhs metrics '''

''' merge climate and gegraphic data '''