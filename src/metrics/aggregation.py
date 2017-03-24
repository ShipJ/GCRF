"""
This module aggregated
"""

import pandas as pd
import numpy as np
from src.config import config
import sys


def model_1(PATH, cdr_bts, country):

    intersect_pop = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/pop/intersect_pop.csv' % country))
    totals = intersect_pop.groupby('Adm_4')['Pop_2010'].sum().reset_index()

    vol1 = []
    for adm in sorted(pd.unique(cdr_bts['Adm_4'])):
        adm_i = intersect_pop[intersect_pop['Adm_4'] == adm]
        total = np.array(totals[totals['Adm_4'] == adm]['Pop_2010'])[0]

        vol = 0
        for index, row in adm_i.iterrows():
            existing_tower = cdr_bts[cdr_bts['CellTowerID'] == row.CellTowerID]
            if len(existing_tower) > 0:
                prop = row.Pop_2010/float(total)
                vol += prop * np.array(existing_tower['Vol'])[0]
        vol1.append(vol)

    print vol1
    print sum(vol1)



    sys.exit()


    adm = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/cdr/bts/adm_1234.csv' % country))

    cdr_fundamentals_adm = pd.DataFrame(prop_vol,
                                      columns=[metric for metric in cdr_bts.columns if
                                               metric not in ['CellTowerID', 'Adm_1', 'Adm_2', 'Adm_3', 'Adm_4']])
    return pd.DataFrame(pd.concat([adm, cdr_fundamentals_adm], axis=1))

if __name__ == '__main__':
    PATH = config.get_dir()
    country = config.get_country()
    for i in ['all', 'working']:
        cdr_bts = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/cdr/metrics/cdr_derived_bts_%s.csv' % (country, i)))
        # Using Model 1 (reference literature), aggregate CT level data to administrative levels
        cdr_adm = model_1(PATH, cdr_bts, country)



        # cdr_adm.to_csv(PATH+'/processed/%s/cdr/metrics/cdr_aggregate_adm_%s.csv' % (country, i), index=None)