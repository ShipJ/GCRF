import pandas as pd
from src.config import config
import numpy as np
import sys


country = config.get_country()

activity = pd.DataFrame(pd.read_csv('../../data/processed/%s/cdr/staticmetrics/new/total_activity.csv' % country))
entropy = pd.DataFrame(pd.read_csv('../../data/processed/%s/cdr/staticmetrics/new/entropy.csv' % country))
med_degree = pd.DataFrame(pd.read_csv('../../data/processed/%s/cdr/staticmetrics/new/med_degree.csv' % country))
graph_metrics = pd.DataFrame(pd.read_csv('../../data/processed/%s/cdr/staticmetrics/new/graph_metrics.csv' % country))
introversion = pd.DataFrame(pd.read_csv('../../data/processed/%s/cdr/staticmetrics/new/introversion.csv' % country))

master = activity.merge(entropy,
                        on='CellTowerID').merge(med_degree,
                                                on='CellTowerID').merge(graph_metrics,
                                                                        on='CellTowerID').merge(introversion,
                                                                                                on='CellTowerID')

# Merge Adm regions
cdr_adm = pd.DataFrame(pd.read_csv('../../data/processed/%s/cdr/CellTowers/CDR_Adm_1234.csv' % country,
                                   usecols=['CellTowerID', 'Adm_1', 'Adm_2', 'Adm_3', 'Adm_4']))

master = master.merge(cdr_adm, on='CellTowerID')

''' merge populations and distance data '''
pop = pd.DataFrame(pd.read_csv('../../data/processed/%s/pop/IntersectPop.csv' % country))
pop_10 = pop.groupby('CellTowerID')['Pop_2010'].sum().reset_index()
pop_14 = pop.groupby('CellTowerID')['Pop_2014'].sum().reset_index()

master = master.merge(pop_10, on='CellTowerID').merge(pop_14, on='CellTowerID')

constants = config.get_constants(country)
num_towers = constants['num_towers']








