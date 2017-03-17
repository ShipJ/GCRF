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
    data_path = root_path+'/data'
    if data_path.endswith('/data'):
        return data_path
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
    source = get_dir()
    if country in ['civ', 'sen']:
        adj_vol = np.genfromtxt(source + '/processed/%s/cdr/metrics/adj_matrix_vol.csv' % country, delimiter=',')
        adj_dur = np.genfromtxt(source + '/processed/%s/cdr/metrics/adj_matrix_dur.csv' % country, delimiter=',')
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
    source = get_dir()
    if country in ['civ', 'sen']:
        data = pd.DataFrame(pd.read_csv(source+'/processed/%s/pop/intersect_pop.csv' % country))
        if len(args) > 0:
            if args[0] in [1, 2, 3, 4]:
                pop = data.groupby('Adm_%s'%args[0])['Pop_2010'].sum().reset_index()
                return pop
            else:
                print 'Sorry, it is not possible to aggregate population at that level.\n'
        else:
            return data
    else:
        print 'Did not recognise country code, please try again: \n'
        return get_country()


def get_cdr_metrics(country):
    source = get_dir()
    activity = pd.DataFrame(pd.read_csv(source+'/processed/%s/cdr/metrics/total_activity.csv' % country))
    entropy = pd.DataFrame(pd.read_csv(source+'/processed/%s/cdr/metrics/entropy.csv' % country))
    med_degree = pd.DataFrame(pd.read_csv(source+'/processed/%s/cdr/metrics/med_degree.csv' % country))
    graph = pd.DataFrame(pd.read_csv(source+'/processed/%s/cdr/metrics/graph_metrics.csv' % country))
    introversion = pd.DataFrame(pd.read_csv(source+'/processed/%s/cdr/metrics/introversion.csv' % country))
    residuals = pd.DataFrame(pd.read_csv(source+'/processed/%s/cdr/metrics/residuals.csv' % country))
    # Insert some kind of merge function to return one dataframe, push into the func below
    return activity, entropy, med_degree, graph, introversion, residuals


def get_raw_dhs(country):
    """

    :param country:
    :return:
    """
    source = get_dir()
    malaria = pd.DataFrame(pd.read_csv(source+'/interim/%s/dhs/malaria.csv' % country))
    child_mort = pd.DataFrame(pd.read_csv(source+'/interim/%s/dhs/child_mort.csv' % country))

    # preventable_disease = pd.DataFrame(pd.read_csv(source+'/interim/%s/dhs/preventable_disease.csv' % country))
    if country == 'civ':
        hiv = pd.DataFrame(pd.read_csv(source+'/interim/%s/dhs/hiv.csv' % country))
        women_health_access = pd.DataFrame(pd.read_csv(source+'/interim/%s/dhs/women_health_access.csv' % country))
        return [malaria, hiv, child_mort, women_health_access]
    else:
        return [malaria, child_mort]


def get_master_cdr(country):
    source=get_dir()
    return pd.DataFrame(pd.read_csv(source+'/processed/%s/cdr/cdr_fundamentals.csv' % country))


def get_master_dhs(country):
    source=get_dir()
    return pd.DataFrame(pd.read_csv(source+'/processed/%s/dhs/dhs_fundamentals.csv' % country))


def get_master_other(country):
    source=get_dir()
    return pd.DataFrame(pd.read_csv(source+'processed/%s/dhs/wealth/dhs_wealth_poverty.csv' % country,
                                    usecols=['UrbRur', 'Poverty', 'Z_Med', 'Capital', 'Pop_1km',
                                             'Adm_1', 'Adm_2', 'Adm_3', 'Adm_4']))


def get_headers(country, *args):
    if args[0] == 'adm':
        return ['Adm_1', 'Adm_2', 'Adm_3', 'Adm_4']
    elif args[0] == 'cdr':
        return ['Vol', 'Entropy', 'Introversion', 'Med_degree',
                'Degree_centrality', 'EigenvectorCentrality', 'G_residuals', 'Log_vol', 'Vol_pp', 'Log_pop_density',
                'Pagerank']
    elif args[0] == 'dhs':
        if country == 'civ':
            return ['BloodPosRate'] # , 'RapidPosRate', 'PosRate', 'DeathRate', 'DifficultyScore'
        elif country == 'sen':
            return ['BloodPosRate'] # , 'RapidPosRate', 'DeathRate'
    elif args[0] == 'models':
        return ['Baseline', 'CDR', 'Baseline+CDR', 'Lag', 'Baseline+Lag', 'All']
    elif args[0] == 'group':
        return ['Poverty', 'Z_Med', 'UrbRur', 'Capital']


