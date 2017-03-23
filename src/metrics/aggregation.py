"""
This module aggregated
"""

import pandas as pd
import numpy as np
from src.config import config

def model_1(cdr_bts, country, PATH):
    # Population within the bts-voronoi and adm-region intersections
    pop_intersect = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/pop/intersect_pop.csv' % country))
    # Do not include areas for which there are no people
    low_pop = pop_intersect[pop_intersect['Pop_2010'] < 1]['CellTowerID']
    for i in low_pop:
        pop_intersect = pop_intersect[pop_intersect['CellTowerID'] != i]
    prop_vol = []
    for i in pd.unique(pop_intersect['Adm_4']):
        adm_i = pop_intersect[pop_intersect['Adm_4'] == i]
        vol = np.zeros(15)
        count = 0
        for j in adm_i['CellTowerID']:
            cell_tower_pop = np.sum(pop_intersect[pop_intersect['CellTowerID'] == j]['Pop_2010'])
            prop_pop = np.sum(adm_i[adm_i['CellTowerID'] == j]['Pop_2010'])
            cell_tower_all = np.sum(cdr_bts[cdr_bts['CellTowerID'] == j])
            vol[:14] += prop_pop/float(cell_tower_pop)*np.array(cell_tower_all[1:15])
            grav = cell_tower_all['G_residuals']
            if not np.isnan(grav):
                vol[14] += prop_pop/float(cell_tower_pop)*np.array(grav)
                count +=1

        vol[8:14] = vol[8:14] / np.array(len(adm_i))
        vol[14] = vol[14] / np.array(count)
        prop_vol.append(vol)
    adm = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/cdr/bts/adm_1234.csv' % country))

    cdr_fundamentals_adm = pd.DataFrame(prop_vol,
                                      columns=[metric for metric in cdr_bts.columns if
                                               metric not in ['CellTowerID', 'Adm_1', 'Adm_2', 'Adm_3', 'Adm_4']])
    return pd.DataFrame(pd.concat([adm, cdr_fundamentals_adm], axis=1))

if __name__ == '__main__':
    # path to data store
    PATH = config.get_dir()
    # ask user for country, retrieve country constants
    country = config.get_country()
    # Retrieve CDR data
    cdr_fundamentals_bts = pd.DataFrame(pd.read_csv(PATH +
                                                    '/processed/%s/cdr/metrics/cdr_fundamentals_bts.csv' % country))

    # Using Model 1 (reference literature), aggregate CT level data to administrative levels
    cdr_fundamentals_adm = model_1(cdr_fundamentals_bts, country, PATH)
    cdr_fundamentals_adm.to_csv(PATH+'/processed/%s/cdr/metrics/cdr_fundamentals_adm.csv' % country, index=None)