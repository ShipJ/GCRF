import pandas as pd
import numpy as np
from collections import Counter


def malaria_rate(malaria, country):
    """

    :param malaria - dataframe containing blood/rapid test results from processed dhs data:
    :param country - ...
    :return:
    """
    max_household = (malaria.shape[1] - 2) / 2
    # Number of houses within the cluster
    num_houses = np.array(malaria.groupby('DHSClust')['HouseID'].count())
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
    malaria_2 = pd.DataFrame()
    malaria_2['DHSClust'] = pd.unique(malaria.DHSClust)
    malaria_2['Blood_pos'], malaria_2['Blood_neg'], malaria_2['Blood_tot'] = blood_pos, blood_neg, blood_tot
    malaria_2['Rapid_pos'], malaria_2['Rapid_neg'], malaria_2['Rapid_tot'] = rapid_pos, rapid_neg, rapid_tot
    DHS_Adm = pd.DataFrame(pd.read_csv('../../data/processed/%s/dhs/dhs_locations/dhs_adm1234.csv' % country))
    malaria_2['Adm_4'] = DHS_Adm['Adm_4']
    malaria_2 = malaria_2.groupby('Adm_4')['Blood_pos', 'Blood_neg', 'Blood_tot', 'Rapid_pos',
                                           'Rapid_neg', 'Rapid_tot'].sum().reset_index()
    malaria_2['BloodPosRate'] = np.true_divide(np.array(malaria_2['Blood_pos']), np.array(malaria_2['Blood_tot']))
    malaria_2['RapidPosRate'] = np.true_divide(np.array(malaria_2['Rapid_pos']), np.array(malaria_2['Rapid_tot']))
    return malaria_2


def child_mort_rate(child_mort, country):
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
    born_died = pd.DataFrame()
    born_died['DHSClust'] = pd.unique(child_mort.DHSClust)
    born_died['TotalBorn'] = born_cluster
    born_died['TotalDied'] = died_cluster
    DHS_Adm = pd.DataFrame(pd.read_csv('../../data/processed/%s/dhs/dhs_locations/dhs_adm1234.csv' % country))
    born_died['Adm_4'] = DHS_Adm['Adm_4']
    born_died_clust = born_died.groupby('Adm_4')['TotalBorn', 'TotalDied'].sum().reset_index()
    born_died_clust['DeathRate'] = np.true_divide(np.array(born_died_clust['TotalDied']),
                                                  np.array(born_died_clust['TotalBorn']))
    return born_died_clust

def hiv_rate(hiv, country):
    """

    :param hiv:
    :param country:
    :return:
    """
    if country == 'civ':
        blood_pos, blood_neg, blood_tot = [], [], []
        print 'Extracting positive/negative HIV rates for: %s\n' % country
        for i in pd.unique(hiv['DHSClust']):
            count = pd.DataFrame(hiv[hiv['DHSClust'] == i]['BloodResult'].value_counts())
            count_dict = count.reset_index().set_index('index').to_dict()['BloodResult']
            blood_neg.append(sum(count_dict[k] for k in count_dict.keys() if k == 0))
            blood_pos.append(sum(count_dict[k] for k in count_dict.keys() if k == 1))
            blood_tot.append(sum(count_dict[k] for k in count_dict.keys() if k >= 0))
        hiv_rate = pd.DataFrame()
        hiv_rate['DHSClust'] = pd.unique(hiv.DHSClust)
        hiv_rate['Neg'] = blood_neg
        hiv_rate['Pos'] = blood_pos
        hiv_rate['Tot'] = blood_tot
        DHS_Adm = pd.DataFrame(pd.read_csv('../../data/processed/%s/dhs/dhs_locations/dhs_adm1234.csv' % country))
        hiv_rate['Adm_4'] = DHS_Adm['Adm_4']
        hiv_rate = hiv_rate.groupby('Adm_4')['Neg', 'Pos', 'Tot'].sum().reset_index()
        hiv_rate['PosRate'] = np.true_divide(np.array(hiv_rate['Pos']), np.array(hiv_rate['Tot']))
        return hiv_rate
    else:
        print 'HIV Data does not exist for: %s\n' % country


def prevent_disease(preventable_disease, country):
    """

    :param preventable_disease:
    :param country:
    :return:
    """
    print preventable_disease




    # blood_pos, blood_neg, blood_tot = [], [], []
    # print 'Extracting positive/negative HIV rates for: %s\n' % country
    # for i in pd.unique(hiv['DHSClust']):
    #     count = pd.DataFrame(hiv[hiv['DHSClust'] == i]['BloodResult'].value_counts())
    #     count_dict = count.reset_index().set_index('index').to_dict()['BloodResult']
    #     blood_neg.append(sum(count_dict[k] for k in count_dict.keys() if k == 0))
    #     blood_pos.append(sum(count_dict[k] for k in count_dict.keys() if k == 1))
    #     blood_tot.append(sum(count_dict[k] for k in count_dict.keys() if k >= 0))
    # hiv_rate = pd.DataFrame()
    # hiv_rate['DHSClust'] = pd.unique(hiv.DHSClust)
    # hiv_rate['Neg'] = blood_neg
    # hiv_rate['Pos'] = blood_pos
    # hiv_rate['Tot'] = blood_tot
    # DHS_Adm = pd.DataFrame(pd.read_csv('../../data/processed/%s/dhs/dhs_locations/dhs_adm1234.csv' % country))
    # hiv_rate['Adm_4'] = DHS_Adm['Adm_4']
    # hiv_rate = hiv_rate.groupby('Adm_4')['Neg', 'Pos', 'Tot'].sum().reset_index()
    # hiv_rate['PosRate'] = np.true_divide(np.array(hiv_rate['Pos']), np.array(hiv_rate['Tot']))
    # return hiv_rate

def health_access(data, country):
    """

    :param data:
    :param country:
    :return:
    """
    if country == 'civ':
        big_problem, no_problem, other_problem, all_problems = [], [], [], []

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
            big_problem.append(problems[1])
            no_problem.append(problems[2])
            other_problem.append(problems[9])
            all_problems.append(problems[1]+problems[2]+problems[9])
        health_access = pd.DataFrame()
        health_access['DHSClust'] = pd.unique(data.DHSClust)
        health_access['BigProblem'] = big_problem
        health_access['NoProblem'] = no_problem
        health_access['OtherProblem'] = other_problem
        health_access['AllProblems'] = all_problems
        DHS_Adm = pd.DataFrame(pd.read_csv('../../data/processed/%s/dhs/dhs_locations/dhs_adm1234.csv' % country))
        health_access['Adm_4'] = DHS_Adm['Adm_4']
        health_access = health_access.groupby('Adm_4')['BigProblem', 'NoProblem',
                                                       'OtherProblem', 'AllProblems'].sum().reset_index()
        health_access['DifficultyScore'] = np.true_divide(np.array(health_access['BigProblem']),
                                                          np.array(health_access['AllProblems']))
        return health_access

    elif country == 'sen':
        print "No data on women's health access for Senegal"

