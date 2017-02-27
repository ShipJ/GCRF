import pandas as pd
import numpy as np
from src.config import config





if __name__ == '__main__':
    country = config.get_country()
    constants = config.get_constants(country)
    num_towers = constants['num_towers']
    num_hours = constants['hours']
    adj_matrix_vol = np.genfromtxt('../../../../data/processed/%s/cdr/staticmetrics/adj_matrix_vol.csv' % country,
                                   delimiter=',')

