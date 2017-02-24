import pandas as pd
import numpy as np
import sys

from src.data import  process_dhs_funcs
from src.config import config

if __name__ == '__main__':
    country = config.get_country()
    constants = config.get_constants(country)

    dhs = config.get_dhs(country)
    for i in range(len(dhs)):
        print "Reading DHS Data set %s: " % i
        dhs[i] = dhs[i].applymap(lambda x: -1 if isinstance(x, basestring) and x.isspace() else int(x))

    if country == 'civ':
        malaria, child_mort, women_health_access, hiv, preventable_disease = dhs
    elif country == 'sen':
        malaria, child_mort, women_health_access, preventable_disease = dhs
        hiv = []
    else:
        malaria, child_mort, women_health_access, hiv, preventable_disease = [], [], [], [], []


    mal = process_dhs_funcs.malaria_rate(malaria, country)
    hiv = process_dhs_funcs.hiv_rate(hiv, country)
    child = process_dhs_funcs.child_mort_rate(child_mort, country)
    # prevent_disease = process_dhs_funcs.prevent_disease(preventable_disease, country)
    women_health_access = process_dhs_funcs.health_access(women_health_access, country)

    all_dhs = mal.merge(hiv,
                        on='Adm_4').merge(child,
                                          on='Adm_4').merge(women_health_access,
                                                            on='Adm_4').set_index('Adm_4')
    all_dhs = all_dhs.reindex(range(constants['Adm_4']+1)).reset_index()
    adm = pd.DataFrame(pd.read_csv('../../data/processed/%s/cdr/cell_towers/Adm_1234.csv' % country))
    all_dhs = pd.concat([all_dhs, adm], axis=1)
    print all_dhs

