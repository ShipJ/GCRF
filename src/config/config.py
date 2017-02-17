import numpy as np

"""
This file contains various functions that used by various scripts, as well as providing constant values for
different countries - i.e. data that will not change.
"""

def get_country():
    """
    Ask user for country code.

    :return: str - country for which there is data.
    """
    print "Process data for which country? ['sen': Senegal, 'civ': Ivory Coast]: "
    input_country = raw_input()
    if input_country == 'sen':
        country = 'sen'
    elif input_country == 'civ':
        country = 'civ'
    else:
        print "Please type the country abbreviation (lower case): "
        return get_country()
    return country


def get_constants(country):
    """
    Return constant values for the cdr data set for each country.

    :param country: str - country code.
    :return:
    """
    if country == 'civ':
        constants = {'country': 'civ', 'num_towers': 1240, 'hours': 3278}
        return constants
    elif country == 'sen':
        constants = {'country': 'sen', 'num_towers': 1668, 'hours': 8733}
        return constants
    else:
        print "Please type the country abbreviation (lower case): "
        return get_constants(country)


def get_adj_matrix(country):
    """
        Return volume and duration adjacency matrices for requested country.

        :param country: str - country code.
        :return:
        """
    if country in ['civ', 'sen']:
        return np.genfromtxt('../../../../data/processed/%s/cdr/staticmetrics/'
                             'adj_matrix_vol.csv' % country, delimiter=','),\
               np.genfromtxt('../../../../data/processed/%s/cdr/staticmetrics/'
                             'adj_matrix_dur.csv' % country, delimiter=',')
    else:
        print "Please type the country abbreviation (lower case): "
        return get_constants(country)