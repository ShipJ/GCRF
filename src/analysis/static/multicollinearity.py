"""
TODO: Model 1: No data transformation, minimal outlier detection applied, no distance weighted function,
no sparsification applied either - i.e. using all data, all links, at all times.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from src.config import config


if __name__ == '__main__':
    PATH = config.get_dir()
    country = config.get_country()


    data = pd.DataFrame(pd.read_csv(PATH+'/final/%s/master_2.0.csv' % country,
                                    usecols=['Vol', 'Entropy', 'Pagerank', 'EigenvectorCentrality',
                                             'Introversion', 'G_residuals', 'Vol_pp',
                                             'Log_pop_density']))
    a = data.dropna().corr()

    print a[a>0.7]