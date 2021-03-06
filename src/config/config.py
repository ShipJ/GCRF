"""
This file contains various functions that are used by different scripts, as well
as providing constant values for different countries - i.e. data that will not change.
"""

import numpy as np
import pandas as pd
import os


def get_dir():
    """
    This simple function returns the path to the data directory, which sits
    at the same level as the src directory. This allows access to data files
    using the same commands within different subdirectories.
    :return:
    """
    root_path = os.path.dirname(os.path.abspath(__file__+'../../../'))
    PATH = root_path+'/data'
    if PATH.endswith('/data'):
        return PATH
    else:
        print 'Path to data source not found. Check documentation for correct structure. \n'


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
        print "Please type the country abbreviation (lower case): \n"
        return get_country()
    return country


def get_constants(country):
    """
    Return constant values for the cdr data set for each country.

    :param country: str - country code.
    :return:
    """
    if country == 'civ':
        constants = {'country': 'civ', 'num_towers': 1240, 'num_hours': 3278,
                     'Adm_1': 11, 'Adm_2': 33, 'Adm_3': 111, 'Adm_4': 191}
        return constants
    elif country == 'sen':
        constants = {'country': 'sen', 'num_towers': 1668, 'num_hours': 8733,
                     'Adm_1': 14, 'Adm_2': 45, 'Adm_3': 122, 'Adm_4': 431}
        return constants
    else:
        print "Please type the country abbreviation (lower case): \n"
        return get_constants(country)


def get_adj_matrix(country):
    """
        Return volume and duration adjacency matrices for requested country.

        :param country: str - country code.
        :return:
        """
    PATH = get_dir()
    if country in ['civ', 'sen']:
        adj_vol = np.genfromtxt(PATH+'/processed/%s/cdr/adjacency/adj_matrix_vol_unchanged.csv' % country, delimiter=',')
        adj_dur = np.genfromtxt(PATH+'/processed/%s/cdr/adjacency/adj_matrix_dur_unchanged.csv' % country, delimiter=',')
        return adj_vol, adj_dur
    else:
        print "Please type a correct country abbreviation (lower case): \n"
        return get_constants(country)


def get_pop(country, *args):
    """
    Returns the population of the intersections between adminstrative regions and voronoi regions of cell towers. They
    are labelled and so it is possible to calculate the proportion of cell tower features within an area that belong
    to a particular administrative region, using data extracted from raster files using QGIS.  The user manual describes
    this in more detail, but essentially the user can provide an argument asking to aggregate the population at a
    particular administrative level, or not.
    :param country:
    :param adm:
    :return:
    """
    PATH = get_dir()
    if country in ['civ', 'sen']:
        data = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/pop/intersect_pop.csv' % country))
        if len(args) > 0:
            if args[0] in [1, 2, 3, 4]:
                pop = data.groupby('Adm_%s' % args[0])['Pop_2010'].sum().reset_index()
                return pop
            else:
                print 'Sorry, it is not possible to aggregate population at that level.\n'
        else:
            return data
    else:
        print 'Did not recognise country code, please try again:\n'
        return get_country()


def get_raw_dhs(country):
    """

    :param country:
    :return:
    """
    PATH = get_dir()
    malaria = pd.DataFrame(pd.read_csv(PATH+'/interim/%s/dhs/malaria.csv' % country))
    child_mort = pd.DataFrame(pd.read_csv(PATH+'/interim/%s/dhs/child_mort.csv' % country))
    if country == 'civ':
        hiv = pd.DataFrame(pd.read_csv(PATH+'/interim/%s/dhs/hiv.csv' % country))
        women_health_access = pd.DataFrame(pd.read_csv(PATH+'/interim/%s/dhs/women_health_access.csv' % country))
        return [malaria, hiv, child_mort, women_health_access]
    else:
        return [malaria, child_mort]


def get_master_cdr(country, *args):
    PATH=get_dir()
    return pd.DataFrame(pd.read_csv(PATH+'/processed/%s/cdr/metrics/cdr_fundamentals_%s.csv' % (country, args[0])))


def get_master_dhs(country):
    PATH=get_dir()
    return pd.DataFrame(pd.read_csv(PATH+'/processed/%s/dhs/metrics/dhs_fundamentals_adm.csv' % country))


def get_master_other(country):
    PATH=get_dir()
    features = ['UrbRur', 'Poverty', 'Z_Med', 'Capital', 'Pop_1km', 'Adm_1', 'Adm_2', 'Adm_3', 'Adm_4']

    return pd.DataFrame(pd.read_csv(PATH+'processed/%s/dhs/wealth/dhs_wealth_poverty.csv' % country,
                                    usecols=features))


def get_headers(country, *args):
    if args[0] == 'adm':
        return ['Adm_1', 'Adm_2', 'Adm_3', 'Adm_4']
    elif args[0] == 'cdr':
        return ['Vol', 'Entropy', 'Introversion', 'Med_deg',
                'Deg_central', 'Eig_central', 'G_resids', 'Log_vol', 'Vol_pp', 'Log_pop_density',
                'Pagerank']
    elif args[0] == 'dhs':
        if country == 'civ':
            return ['BloodPosRate', 'RapidPosRate', 'PosRate', 'DeathRate', 'DifficultyScore']
        elif country == 'sen':
            return ['BloodPosRate', 'RapidPosRate', 'DeathRate']
    elif args[0] == 'models':
        return ['Baseline', 'CDR', 'Baseline+CDR', 'Lag', 'Baseline+Lag', 'All']
    elif args[0] == 'group':
        return ['Poverty', 'Z_Med', 'UrbRur', 'Capital']
