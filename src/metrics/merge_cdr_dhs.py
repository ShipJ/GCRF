"""
Run once to obtain a merged CDR/DHS file ready for modelling
"""


import pandas as pd
from src.config import config


def merge_cdr_dhs(a, b):
    """

    :param a:
    :param b:
    :return:
    """
    return pd.DataFrame(a.merge(b, on=['Adm_1', 'Adm_2', 'Adm_3', 'Adm_4'])).dropna()


def merge_other(cdr_dhs, other):
    """

    :param cdr_dhs:
    :param other:
    :return:
    """
    mode = lambda x: x.value_counts().index[0]

    urbrurcapital = other.groupby('Adm_4')['UrbRur', 'Capital'].agg(mode).reset_index()
    pov_z_med = other.groupby('Adm_4')['Poverty', 'Z_Med', 'Pop_1km'].mean().reset_index()

    master_cdr_dhs_other = pd.DataFrame(cdr_dhs.merge(urbrurcapital,
                                                      on='Adm_4',
                                                      how='outer').merge(pov_z_med,
                                                                         on='Adm_4',
                                                                         how='outer'), index=None)
    return master_cdr_dhs_other

def merge_popdensity(cdr_dhs, popdense):
    return pd.DataFrame(cdr_dhs.merge(popdense, on=['Adm_1', 'Adm_2', 'Adm_3', 'Adm_4'])).dropna()

if __name__ == '__main__':

    country = config.get_country()
    cdr = config.get_master_cdr(country)
    dhs = config.get_master_dhs(country)
    other = config.get_master_other(country)
    popdense = pd.DataFrame(pd.read_csv('../../data/processed/%s/pop/popdensity.csv' % country))

    cdr_dhs = merge_cdr_dhs(cdr, dhs)
    cdr_dhs_other = merge_other(cdr_dhs, other)

    cdr_dhs_other_pop = merge_popdensity(cdr_dhs_other, popdense)

    cdr_dhs_other_pop.to_csv('../../data/processed/%s/master_pop.csv' % country, index=None)










