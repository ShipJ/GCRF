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


    # mal = process_dhs_funcs.malaria_rate(malaria, country)

    hiv = process_dhs_funcs.hiv_rate(hiv, country)




