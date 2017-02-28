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

mode = lambda x: x.value_counts().index[0]





    urbrurcapital = other.groupby('Adm_4')['UrbRur', 'Capital'].agg(mode).reset_index()
    pov_z_med = other.groupby('Adm_4')['Poverty', 'Z_Med', 'Pop_1km'].mean().reset_index()

    master_cdr_dhs_other = master.merge(urbrurcapital,
                                        on='Adm_4', how='outer').merge(pov_z_med,
                                                                       on='Adm_4', how='outer')
    pop = pd.DataFrame(pd.read_csv('../../data/processed/%s/pop/IntersectPop.csv' % country))
    pop = pop.groupby('Adm_4')['Pop_2010', 'Pop_2014'].sum().reset_index()

    master = pd.DataFrame(master_cdr_dhs_other.merge(pop, on='Adm_4', how='outer'))



    cdr = pd.DataFrame(pd.read_csv('../../data/processed/%s/cdr/staticmetrics/master_cdr.csv' % country))
    master = master.merge(cdr, on='Adm_4', how='outer')


if __name__ == '__main__':

    country = config.get_country()
    cdr = config.get_master_cdr(country)
    dhs = config.get_master_dhs(country)
    cdr_dhs = merge_cdr_dhs(cdr, dhs)
    cdr_dhs.to_csv('')










