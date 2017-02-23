import pandas as pd
import numpy as np
import sys

from src.config import config


if __name__ == '__main__':
    country = config.get_country()
    constants = config.get_constants(country)

    if country == 'civ':
        malaria, child_mort, women_health_access, hiv, preventable_disease = config.get_dhs(country)
    else:
        malaria, child_mort, women_health_access, preventable_disease = config.get_dhs(country)






