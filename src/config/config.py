import numpy as np
import pandas as pd

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
        constants = {'country': 'civ', 'num_towers': 1240, 'hours': 3278,
                     'Adm_1': 11, 'Adm_2': 33, 'Adm_3': 111, 'Adm_4': 191}
        return constants
    elif country == 'sen':
        constants = {'country': 'sen', 'num_towers': 1668, 'hours': 8733,
                     'Adm_1': 14, 'Adm_2': 45, 'Adm_3': 122, 'Adm_4': 431}
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


def get_cdr_features(country):
    """
    Return the cdr features of the given country.
    :param country: country code.
    :return: dataframe containing the respective data.
    """
    if country in ['civ', 'sen']:
        return pd.DataFrame(pd.read_csv('../../../../data/processed/%s/cdr/staticmetrics/'
                                        'new/total_activity.csv' % country)),\
               pd.DataFrame(pd.read_csv('../../../../data/processed/%s/cdr/staticmetrics/'
                                        'new/degree_vector.csv' % country)),\
               # pd.DataFrame(pd.read_csv('../../../../data/processed/%s/cdr/staticmetrics/new/total_activity.csv'))

def get_pop(country, adm):
    """

    :param country:
    :param adm:
    :return:
    """
    if country in ['civ', 'sen']:
        return pd.DataFrame(pd.read_csv('../../data/processed/%s/pop/pop_adm_%d_2010.csv' % (country, adm)))


def get_cdr_metrics(country):
    activity = pd.DataFrame(pd.read_csv('../../data/processed/%s/cdr/staticmetrics/new/total_activity.csv' % country))
    entropy = pd.DataFrame(pd.read_csv('../../data/processed/%s/cdr/staticmetrics/new/entropy.csv' % country))
    med_degree = pd.DataFrame(pd.read_csv('../../data/processed/%s/cdr/staticmetrics/new/med_degree.csv' % country))
    graph = pd.DataFrame(pd.read_csv('../../data/processed/%s/cdr/staticmetrics/new/graph_metrics.csv' % country))
    introversion = pd.DataFrame(pd.read_csv('../../data/processed/%s/cdr/staticmetrics/new/introversion.csv' % country))
    return [activity, entropy, med_degree, graph, introversion]


def get_dhs(country):
    malaria = pd.DataFrame(pd.read_csv('../../data/interim/%s/dhs/malaria.csv' % country))
    child_mort = pd.DataFrame(pd.read_csv('../../data/interim/%s/dhs/child_mort.csv' % country))
    women_health_access = pd.DataFrame(pd.read_csv('../../data/interim/%s/dhs/women_health_access.csv' % country))
    preventable_disease = pd.DataFrame(pd.read_csv('../../data/interim/%s/dhs/preventable_disease.csv' % country))
    if country == 'civ':
        hiv = pd.DataFrame(pd.read_csv('../../data/interim/%s/dhs/hiv.csv' % country))
        return [malaria, child_mort, women_health_access, hiv, preventable_disease]
    else:
        return [malaria, child_mort, women_health_access, preventable_disease]

def get_master_cdr(country):
    return pd.DataFrame(pd.read_csv('../../data/processed/%s/cdr/staticmetrics/master_cdr.csv' % country))

def get_master_dhs(country):
    return pd.DataFrame(pd.read_csv('../../data/processed/%s/dhs/master_dhs.csv' % country))


