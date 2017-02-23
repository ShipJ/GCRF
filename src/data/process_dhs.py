import pandas as pd
import numpy as np
import sys

from src.config import config


if __name__ == '__main__':
    country = config.get_country()
    constants = config.get_constants(country)

    malaria = np.genfromtxt('../../../data/interim/%s/dhs/.csv' % country, delimiter=',')

    print malaria



