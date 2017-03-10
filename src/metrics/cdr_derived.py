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

    # Add the population within each adm region
    cdr_fundamentals = cdr_fundamentals.merge(config.get_pop(country, 4), on='Adm_4')

    # Log volume
    cdr_fundamentals['Log_vol'] = np.log(cdr_fundamentals['Vol'])

    # Volume per person
    cdr_fundamentals['Vol_pp'] = cdr_fundamentals['Vol']/cdr_fundamentals['Pop_2010']


