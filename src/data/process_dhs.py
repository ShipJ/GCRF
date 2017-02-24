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

    if country == 'civ':
        all_dhs = mal.merge(hiv,
                            on='Adm_4').merge(child,
                                              on='Adm_4').merge(women_health_access,
                                                                on='Adm_4').set_index('Adm_4')
    elif country == 'sen':
        all_dhs = mal.merge(child, on='Adm_4').set_index('Adm_4')
    else:
        print "Please Select an actual country"
        all_dhs = []

    all_dhs = all_dhs.reindex(range(constants['Adm_4']+1)).reset_index()
    adm = pd.DataFrame(pd.read_csv('../../data/processed/%s/cdr/celltowers/Adm_1234.csv' % country))
    all_dhs = pd.DataFrame(all_dhs.merge(adm, on='Adm_4', how='outer'))

    all_dhs.to_csv('../../data/processed/%s/dhs/master_dhs.csv' % country, index=None)


