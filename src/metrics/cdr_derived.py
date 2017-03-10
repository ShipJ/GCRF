"""
This module takes the fundamental data metrics extracted from the raw data, and computes various transformation,
normalisations and derived metrics (i.e. metrics that involve multiple variables)
"""

import pandas as pd
from src.config import config


if __name__ == '__main__':
    PATH = config.get_dir()
    country = config.get_country()

    cdr_fundamentals = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/cdr/metrics/cdr_fundamentals_adm.csv' % country))

    print cdr_fundamentals