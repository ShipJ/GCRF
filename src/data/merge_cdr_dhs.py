import pandas as pd
from src.config import config
import numpy as np
import sys


if __name__ == '__main__':
    country = config.get_country()
    cdr_metrics = config.get_cdr_metrics(country)

    print cdr_metrics




    sys.exit()





    # Merge Adm region
    cdr_adm = pd.DataFrame(pd.read_csv('../../data/processed/%s/cdr/CellTowers/CDR_Adm_1234.csv' % country,
                                       usecols=['CellTowerID', 'Adm_1', 'Adm_2', 'Adm_3', 'Adm_4']))

master = master.merge(cdr_adm, on='CellTowerID')

''' merge populations and distance data '''
intersect_pop = pd.DataFrame(pd.read_csv('../../data/processed/%s/pop/IntersectPop.csv' % country))
pop_10 = intersect_pop.groupby('CellTowerID')['Pop_2010'].sum().reset_index()
pop_14 = intersect_pop.groupby('CellTowerID')['Pop_2014'].sum().reset_index()

master = master.merge(pop_10, on='CellTowerID').merge(pop_14, on='CellTowerID')

constants = config.get_constants(country)
num_towers = constants['num_towers']

pop_4 = config.get_pop(country, 4)

vor_prop = np.zeros(len(intersect_pop))
for i in range(len(intersect_pop)):
    adm_4 = int(intersect_pop.iloc[i].Adm_4)
    pop_adm_4 = pop_4[pop_4['Adm_4'] == adm_4]['Pop_2010']

    vor_prop[i] = intersect_pop.iloc[i].Pop_2010 / pop_adm_4

intersect_pop['vorProp'] = vor_prop

master_adm = pd.DataFrame(pd.read_csv('../../data/processed/%s/cdr/celltowers/Adm_1234.csv' % country))



