"""
This module takes csv files containing DHS data extracted from SPSS datasets (in turn downloaded
from the DHSProgram website), computes a range of health-related metrics, and aggregates the
data at the highest possible administrative level (i.e. level 4).

the 'save to csv' function is commented out to prevent accidental overriding
"""

import pandas as pd
import dhs_metrics
from src.config import config

if __name__ == '__main__':
    country = config.get_country()
    constants = config.get_constants(country)
    dhs = config.get_dhs(country)

    for i in range(len(dhs)):
        print "Reading DHS Data set %s: " % i
        # Convert all data to integers, and non-existent data to '-1' - more readable
        dhs[i] = dhs[i].applymap(lambda x: -1 if isinstance(x, basestring) and x.isspace() else int(x))

    if country == 'civ':
        malaria, child_mort, women_health_access, hiv, preventable_disease = dhs
    elif country == 'sen':
        malaria, child_mort, women_health_access, preventable_disease = dhs
        hiv = []
    else:
        malaria, child_mort, women_health_access, hiv, preventable_disease = [], [], [], [], []

    mal = dhs_metrics.malaria_rate(malaria, country)
    hiv = dhs_metrics.hiv_rate(hiv, country)
    child = dhs_metrics.child_mort_rate(child_mort, country)
    # prevent_disease = process_dhs_funcs.prevent_disease(preventable_disease, country)
    women_health_access = dhs_metrics.health_access(women_health_access, country)

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
    adm = pd.DataFrame(pd.read_csv('../../data/processed/%s/cdr/bts/Adm_1234.csv' % country))
    all_dhs = pd.DataFrame(all_dhs.merge(adm, on='Adm_4', how='outer'))

    # all_dhs.to_csv('../../data/processed/%s/dhs/master_dhs.csv' % country, index=None)


