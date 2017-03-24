"""
This module takes csv files containing DHS data extracted from SPSS datasets (in turn downloaded
from the DHSProgram website), computes a range of health-related metrics, and aggregates the
data at the highest possible administrative level (i.e. level 4).

"""

import pandas as pd
import numpy as np
from src.config import config
from collections import Counter


def malaria_rate(PATH, malaria, country):
    """

    :param malaria - dataframe containing blood/rapid test results from processed dhs data:
    :param country - ...
    :return:
    """
    max_household = (malaria.shape[1] - 2) / 2
    # Number of houses within the cluster
    bloodtest = malaria.iloc[:, 2:max_household + 2]
    bloodtest.loc[:, 'DHS'] = malaria.DHSClust
    rapidtest = malaria.iloc[:, max_household + 2:]
    rapidtest.loc[:, 'DHS'] = malaria.DHSClust
    blood_pos, blood_neg, blood_tot, rapid_pos, rapid_neg, rapid_tot = [], [], [], [], [], []
    print 'Extracting positive/negative malaria cases for: %s\n' % country
    for i in pd.unique(malaria.DHSClust):
        # Get data corresponding to blood/rapid test results (rows: households, columns: individuals)
        blood = bloodtest[bloodtest['DHS'] == i].iloc[:, 0:max_household]
        rapid = rapidtest[rapidtest['DHS'] == i].iloc[:, 0:max_household]
        # Store key-value pairs in dict: -1: not sampled, 0: negative, 1: positive, {6,7,8,9}: test inconclusive
        blood_frequency = pd.DataFrame(blood.stack().value_counts(),
                                       columns=['count']).reset_index().set_index('index').to_dict()['count']
        rapid_frequency = pd.DataFrame(rapid.stack().value_counts(),
                                       columns=['count']).reset_index().set_index('index').to_dict()['count']
        # Sum positive, negative, and total sampled for blood/rapid tests
        blood_pos.append(sum(blood_frequency[k] for k in blood_frequency.keys() if k == 1))
        blood_neg.append(sum(blood_frequency[k] for k in blood_frequency.keys() if k == 0))
        blood_tot.append(sum(blood_frequency[k] for k in blood_frequency.keys() if k >= 0))
        rapid_pos.append(sum(rapid_frequency[k] for k in rapid_frequency.keys() if k == 1))
        rapid_neg.append(sum(rapid_frequency[k] for k in rapid_frequency.keys() if k == 0))
        rapid_tot.append(sum(rapid_frequency[k] for k in rapid_frequency.keys() if k >= 0))
    # Cluster level
    raw, raw['DHSClust'] = pd.DataFrame(), pd.unique(malaria.DHSClust)
    raw['Blood_pos'], raw['Blood_neg'], raw['Blood_tot'] = blood_pos, blood_neg, blood_tot
    raw['Rapid_pos'], raw['Rapid_neg'], raw['Rapid_tot'] = rapid_pos, rapid_neg, rapid_tot
    raw['BloodPosRate'] = np.true_divide(np.array(raw['Blood_pos']), np.array(raw['Blood_tot']))
    raw['RapidPosRate'] = np.true_divide(np.array(raw['Rapid_pos']), np.array(raw['Rapid_tot']))
    # Aggregated to adm level
    adm = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/dhs/cluster_locations/dhs_adm1234.csv' % country))
    aggregated = pd.DataFrame(raw.drop(['BloodPosRate', 'RapidPosRate'], axis=1))
    aggregated['Adm_4'] = adm['Adm_4']
    aggregated = aggregated.groupby('Adm_4')['Blood_pos', 'Blood_neg', 'Blood_tot', 'Rapid_pos',
                                             'Rapid_neg', 'Rapid_tot'].sum().reset_index()
    aggregated['BloodPosRate'] = np.true_divide(np.array(aggregated['Blood_pos']), np.array(aggregated['Blood_tot']))
    aggregated['RapidPosRate'] = np.true_divide(np.array(aggregated['Rapid_pos']), np.array(aggregated['Rapid_tot']))
    return raw, aggregated


def child_mort_rate(PATH, child_mort, country):
    """

    :param child_mort:
    :param country:
    :return:
    """
    total_born = pd.DataFrame(child_mort, columns=['DHSClust', 'HouseID', 'TotalBorn'])
    born_cluster, died_cluster = [], []
    print 'Extracting child mortality cases for: %s\n' % country
    for i in pd.unique(child_mort.DHSClust):
        born_i = np.sum(total_born[total_born['DHSClust'] == i].drop_duplicates()['TotalBorn'])
        died_i = pd.DataFrame(child_mort[child_mort['DHSClust'] == i]['AgeAtDeath'].value_counts())
        died_i_dict = died_i.reset_index().set_index('index').to_dict()['AgeAtDeath']
        born_cluster.append(born_i)
        died_cluster.append(sum(died_i_dict[k] for k in died_i_dict.keys() if k > 0))
    # Cluster level
    raw = pd.DataFrame()
    raw['DHSClust'], raw['TotalBorn'], raw['TotalDied'] = pd.unique(child_mort.DHSClust), born_cluster, died_cluster
    raw['DeathRate'] = np.true_divide(np.array(raw['TotalDied']), np.array(raw['TotalBorn']))
    # Aggregated to adm level
    adm = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/dhs/cluster_locations/dhs_adm1234.csv' % country))
    aggregated = pd.DataFrame(raw.drop(['DeathRate'], axis=1))
    aggregated['Adm_4'] = adm['Adm_4']
    aggregated = aggregated.groupby('Adm_4')['TotalBorn', 'TotalDied'].sum().reset_index()
    aggregated['DeathRate'] = np.true_divide(np.array(aggregated['TotalDied']), np.array(aggregated['TotalBorn']))
    return raw, aggregated

def hiv_rate(PATH, hiv, country):
    """

    :param hiv:
    :param country:
    :return:
    """
    blood_pos, blood_neg, blood_tot = [], [], []
    print 'Extracting positive/negative HIV rates for: %s\n' % country
    for i in pd.unique(hiv['DHSClust']):
        count = pd.DataFrame(hiv[hiv['DHSClust'] == i]['BloodResult'].value_counts())
        count_dict = count.reset_index().set_index('index').to_dict()['BloodResult']
        blood_neg.append(sum(count_dict[k] for k in count_dict.keys() if k == 0))
        blood_pos.append(sum(count_dict[k] for k in count_dict.keys() if k == 1))
        blood_tot.append(sum(count_dict[k] for k in count_dict.keys() if k >= 0))
        
    # Cluster level
    raw = pd.DataFrame()
    raw['DHSClust'], raw['Neg'], raw['Pos'], raw['Tot'] = pd.unique(hiv.DHSClust), blood_neg, blood_pos, blood_tot
    raw['PosRate'] = np.true_divide(np.array(raw['Pos']), np.array(raw['Tot']))
    # Aggregated to adm level
    adm = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/dhs/cluster_locations/dhs_adm1234.csv' % country))
    aggregated = pd.DataFrame(raw.drop(['PosRate'], axis=1))
    aggregated['Adm_4'] = adm['Adm_4']
    aggregated = aggregated.groupby('Adm_4')['Neg', 'Pos', 'Tot'].sum().reset_index()
    aggregated['PosRate'] = np.true_divide(np.array(aggregated['Pos']), np.array(aggregated['Tot']))
    return raw, aggregated
        

def health_access(PATH, data, country):
    """

    :param data:
    :param country:
    :return:
    """
    big_probs, no_probs, other_probs, all_probs = [], [], [], []

    print 'Extracting health access difficulty scores for: %s' % country
    for i in pd.unique(data['DHSClust']):
        data_i = data[data['DHSClust'] == i].drop_duplicates()

        permission = pd.DataFrame(data_i['Permission'].value_counts())
        permission_dict = permission.reset_index().set_index('index').to_dict()['Permission']
        finance = pd.DataFrame(data_i['Finance'].value_counts())
        finance_dict = finance.reset_index().set_index('index').to_dict()['Finance']
        distance = pd.DataFrame(data_i['Distance'].value_counts())
        distance_dict = distance.reset_index().set_index('index').to_dict()['Distance']
        noCompany = pd.DataFrame(data_i['NoCompany'].value_counts())
        noCompany_dict = noCompany.reset_index().set_index('index').to_dict()['NoCompany']
        femaleProvider = pd.DataFrame(data_i['FemaleProvider'].value_counts())
        femaleProvider_dict = femaleProvider.reset_index().set_index('index').to_dict()['FemaleProvider']

        problems = sum((Counter(dict(x)) for x in [permission_dict, finance_dict, distance_dict,
                                                   noCompany_dict, femaleProvider_dict]), Counter())
        big_probs.append(problems[1]), no_probs.append(problems[2]), other_probs.append(problems[9])
        all_probs.append(problems[1]+problems[2]+problems[9])
        
    # Cluster level
    raw, raw['DHSClust'] = pd.DataFrame(), pd.unique(data.DHSClust)
    raw['BigProbs'], raw['NoProbs'], raw['OtherProbs'], raw['AllProbs'] = big_probs, no_probs, other_probs, all_probs
    raw['DifficultyScore'] = np.true_divide(np.array(raw['BigProbs']), np.array(raw['AllProbs']))
    # Aggregated to adm level
    adm = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/dhs/cluster_locations/dhs_adm1234.csv' % country))
    aggregated = pd.DataFrame(raw.drop(['DifficultyScore'], axis=1))
    aggregated['Adm_4'] = adm['Adm_4']
    aggregated = aggregated.groupby('Adm_4')['BigProbs', 'NoProbs', 'OtherProbs', 'AllProbs'].sum().reset_index()
    aggregated['DifficultyScore'] = np.true_divide(np.array(aggregated['BigProbs']), np.array(aggregated['AllProbs']))
    return raw, aggregated


if __name__ == '__main__':

    PATH = config.get_dir()
    country = config.get_country()
    constants = config.get_constants(country)
    dhs = config.get_raw_dhs(country)

    for i in range(len(dhs)):
        print "Reading DHS Data set %s: " % i
        # Convert all data to integers, and non-existent data to '-1': more readable
        dhs[i] = dhs[i].applymap(lambda x: -1 if isinstance(x, basestring) and x.isspace() else int(x))

    if country == 'civ':
        malaria, hiv, child_mort, women_health_access = dhs
        mal, agg_mal = malaria_rate(PATH, malaria, country)
        hiv, agg_hiv = hiv_rate(PATH, hiv, country)
        child, agg_child = child_mort_rate(PATH, child_mort, country)
        wha, agg_wha = health_access(PATH, women_health_access, country)

        master_raw = agg_mal.merge(agg_hiv,
                                   on='Adm_4').merge(agg_child,
                                                     on='Adm_4').merge(agg_wha,
                                                                       on='Adm_4').set_index('Adm_4')

    else:
        malaria, child_mort = dhs
        mal, agg_mal = malaria_rate(PATH, malaria, country)
        child, agg_child = child_mort_rate(PATH, child_mort, country)
        master_raw = mal.merge(child, on='Adm_4').set_index('Adm_4')

    dhs_fundamentals = master_raw.reindex(range(constants['Adm_4']+1)).reset_index()
    adm = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/cdr/bts/adm_1234.csv' % country))
    dhs_fundamentals = pd.DataFrame(dhs_fundamentals.merge(adm, on='Adm_4', how='outer'))

    dhs_fundamentals.to_csv(PATH+'/processed/%s/dhs/dhs_fundamentals_adm.csv' % country, index=None)


