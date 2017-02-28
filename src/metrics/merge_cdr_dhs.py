"""
Run once to obtain a merged CDR/DHS file ready for modelling
"""


import pandas as pd
from src.config import config


def combine_cdr_dhs(a, b):
    """

    :param a:
    :param b:
    :return:
    """
    return pd.DataFrame(a.merge(b, on=['Adm_1', 'Adm_2', 'Adm_3', 'Adm_4'])).dropna()


if __name__ == '__main__':

    country = config.get_country()
    cdr = config.get_master_cdr(country)
    dhs = config.get_master_dhs(country)
    cdr_dhs = combine_cdr_dhs(cdr, dhs)










