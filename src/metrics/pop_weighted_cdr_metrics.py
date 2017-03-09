"""
This module takes the CDR features derived in CDR_metrics.py, and aggregates
them to the administrative levels, and proportional to the population within
"""


import pandas as pd
import numpy as np
from src.config import config


def merge_cdr(df_list, country):
    """

    :param df_list:
    :return:
    """
    master = df_list[0]
    for i in range(1, len(df_list)):
        master = master.merge(df_list[i], on='CellTowerID')
    # Merge with adm regions
    cdr_adm = pd.DataFrame(pd.read_csv('../../data/processed/%s/cdr/CellTowers/bts_adm_1234.csv' % country,
                                       usecols=['CellTowerID', 'Adm_1', 'Adm_2', 'Adm_3', 'Adm_4']))
    master = master.merge(cdr_adm, on='CellTowerID')
    return master

def feature_per_adm(master, pop_intersect):
    prop_vol = []
    for i in pd.unique(pop_intersect['Adm_4']):
        adm_i = pop_intersect[pop_intersect['Adm_4'] == i]

        vol = 0
        for j in adm_i['CellTowerID']:
            cell_tower_pop = np.sum(pop_intersect[pop_intersect['CellTowerID'] == j]['Pop_2010'])
            prop_pop = np.sum(adm_i[adm_i['CellTowerID'] == j]['Pop_2010'])
            cell_tower_vol = np.sum(master[master['CellTowerID'] == j]['Vol'])
            vol += prop_pop/float(cell_tower_pop)*cell_tower_vol
        prop_vol.append(vol)
    return prop_vol


def all_feature_per_adm(master, pop_intersect):
    prop_vol = []
    for i in pd.unique(pop_intersect['Adm_4']):
        adm_i = pop_intersect[pop_intersect['Adm_4'] == i]

        vol = np.zeros(14)
        for j in adm_i['CellTowerID']:
            cell_tower_pop = np.sum(pop_intersect[pop_intersect['CellTowerID'] == j]['Pop_2010'])
            prop_pop = np.sum(adm_i[adm_i['CellTowerID'] == j]['Pop_2010'])
            cell_tower_all = np.sum(master[master['CellTowerID'] == j])
            vol += prop_pop/float(cell_tower_pop)*np.array(cell_tower_all[1:15])
        vol[8:] = vol[8:] / np.array(len(adm_i))
        prop_vol.append(vol)
    return prop_vol


if __name__ == '__main__':
    country = config.get_country()
    cdr_metrics = config.get_cdr_metrics(country)
    master = merge_cdr(cdr_metrics, country)

    pop_adm_4 = config.get_pop(country, 4)
    pop_intersect = pd.DataFrame(pd.read_csv('../../data/processed/%s/pop/intersect_pop.csv' % country))
    low_pop = pop_intersect[pop_intersect['Pop_2010'] < 1]['CellTowerID']

    for i in low_pop:
        pop_intersect = pop_intersect[pop_intersect['CellTowerID'] != i]

    adm = pd.DataFrame(pd.read_csv('../../data/processed/%s/cdr/cell_towers/adm_1234.csv' % country))
    all_features_adm_4 = pd.DataFrame(all_feature_per_adm(master, pop_intersect), columns=master.columns[1:15])
    master = pd.DataFrame()
    master = pd.DataFrame(pd.concat([master, adm, all_features_adm_4], axis=1))

    # master.to_csv('../../data/processed/%s/cdr/master_cdr.csv' % country, index=None)


