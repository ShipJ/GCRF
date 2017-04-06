"""
This module takes the fundamental data metrics extracted from the raw data, and computes various transformation,
normalisations and derived metrics (i.e. metrics that involve multiple variables)
"""

import pandas as pd
import numpy as np
from src.config import config
import sys

if __name__ == '__main__':
    PATH = config.get_dir()
    country = config.get_country()

    for i in ['all']:
        cdr_fundamentals = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/cdr/metrics/cdr_fundamentals_ct_%s.csv'
                                                    % (country, i)))

        # Population of each adm region
        ct_pop = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/pop/intersect_pop.csv' % country))
        ct_pop = ct_pop.groupby('CellTowerID')['Pop_2010'].sum().reset_index()

        # Area in m^2 and km^2 of each adm region
        area = pd.DataFrame(pd.read_csv(PATH + '/processed/%s/geo/area.csv' % country,
                                        usecols=['Adm_4', 'Area_m2', 'Area_km2']))

        print area

        sys.exit()
        cdr_fundamentals = cdr_fundamentals.merge(ct_pop, on='CellTowerID', how='outer').merge(area, on='Adm_4')
        # Volume per person
        cdr_fundamentals['Vol_pp'] = cdr_fundamentals['Vol']/cdr_fundamentals['Pop_2010']
        # Population density
        cdr_fundamentals['Pop_density'] = cdr_fundamentals['Pop_2010']/cdr_fundamentals['Area_km2']
        # Log(Pop density)
        cdr_fundamentals['Log_pop_density'] = np.log(cdr_fundamentals['Pop_density'])

        cdr_fundamentals.to_csv(PATH+'/processed/%s/cdr/metrics/cdr_derived_bts_%s.csv' % (country, i), index=None)



