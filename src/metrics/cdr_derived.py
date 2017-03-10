"""
This module takes the fundamental data metrics extracted from the raw data, and computes various transformation,
normalisations and derived metrics (i.e. metrics that involve multiple variables)
"""

import pandas as pd
import numpy as np
from src.config import config


if __name__ == '__main__':
    PATH = config.get_dir()
    country = config.get_country()

    cdr_fundamentals = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/cdr/metrics/cdr_fundamentals_adm.csv' % country))

    # Population of each adm region
    cdr_fundamentals = cdr_fundamentals.merge(config.get_pop(country, 4), on='Adm_4')
    # Area in m^2 and km^2 of each adm region
    area = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/distance_area/area.csv' % country, usecols=['Adm_4',
                                                                                                    'Area_m2',
                                                                                                    'Area_km2']))
    cdr_fundamentals = cdr_fundamentals.merge(area, on='Adm_4', how='outer')

    # Log volume
    cdr_fundamentals['Log_vol'] = np.log(cdr_fundamentals['Vol'])
    # Volume per person
    cdr_fundamentals['Vol_pp'] = cdr_fundamentals['Vol']/cdr_fundamentals['Pop_2010']
    # Population density
    cdr_fundamentals['Pop_density'] = cdr_fundamentals['Pop_2010']/cdr_fundamentals['Area_km2']
    # Log(Pop density)
    cdr_fundamentals['Log_pop_density'] = np.log(cdr_fundamentals['Pop_density'])

    cdr_derived = pd.DataFrame(cdr_fundamentals)

    cdr_derived.to_csv(PATH+'/processed/%s/cdr/metrics/cdr_derived_adm.csv' % country, index=None)


