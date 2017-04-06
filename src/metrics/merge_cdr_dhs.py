"""
Run once to obtain a merged CDR/DHS file ready for modelling
"""


import pandas as pd
from src.config import config

if __name__ == '__main__':
    PATH = config.get_dir()
    country = config.get_country()

    cdr = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/cdr/metrics/cdr_derived_adm_all.csv' % country))
    dhs = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/dhs/metrics/dhs_fundamentals_adm.csv' % country))

    cdr_dhs = pd.DataFrame(cdr.merge(dhs, on=['Adm_1', 'Adm_2', 'Adm_3', 'Adm_4']))

    # Save to csv
    cdr_dhs.to_csv(PATH+'/final/%s/master.csv' % country, index=None)


