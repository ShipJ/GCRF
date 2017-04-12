"""
This module takes the fundamental data metrics extracted from the raw data, and computes various transformation,
normalisations and derived metrics (i.e. metrics that involve multiple variables)
"""

import pandas as pd
import numpy as np
from src.config import config
import sys

if __name__ == '__main__':
    # System path to data directory
    PATH = config.get_dir()
    # Ask user for country
    country = config.get_country()

    # Currently for all hours -> future, add 'working hours' etc.
    for i in ['all']:
        cdr_aggregated = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/cdr/metrics/cdr_aggregate_adm_%s.csv'
                                                  % (country, i)))

        # Population of adm regions
        ct_pop = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/pop/intersect_pop.csv' % country))
        if country == 'civ':
            ct_pop = ct_pop.groupby('Adm_4')['Pop_2010', 'Pop_2014'].sum().reset_index()
        else:
            ct_pop = ct_pop.groupby('Adm_4')['Pop_2010'].sum().reset_index()

        # Area in m^2 and km^2 of each adm region
        area = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/geo/area.csv' % country,
                                        usecols=['Adm_4', 'Area_m2', 'Area_km2']))

        # Merge area and population data
        cdr_derived = cdr_aggregated.merge(area, on='Adm_4').merge(ct_pop, on='Adm_4', how='outer')
        # Volume per person
        cdr_derived['Vol_pp'] = cdr_derived['Vol']/cdr_derived['Pop_2010']
        # Population density
        cdr_derived['Pop_density'] = cdr_derived['Pop_2010']/cdr_derived['Area_km2']
        # Log(Pop density)
        cdr_derived['Log_pop_density'] = np.log(cdr_derived['Pop_density'])
        # Log(Volume)
        cdr_derived['Log_vol'] = np.log(cdr_derived['Vol'])

        # Save derived metrics to csv
        cdr_derived.to_csv(PATH+'/processed/%s/cdr/metrics/cdr_derived_adm_%s.csv' % (country, i), index=None)
