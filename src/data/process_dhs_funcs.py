import pandas as pd
import numpy as np
import sys


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
    print "Extracting positive/negative malaria cases...\n"
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
    malaria_2['Num_houses'] = num_houses
    DHS_Adm = pd.DataFrame(pd.read_csv('../../data/processed/%s/dhs/dhs_clusters/DHS_Adm_1234.csv' % country))
    malaria_2['Adm_1'] = DHS_Adm['Adm_1']
    malaria_2['Adm_2'] = DHS_Adm['Adm_2']
    malaria_2['Adm_3'] = DHS_Adm['Adm_3']
    malaria_2['Adm_4'] = DHS_Adm['Adm_4']
    malaria_2 = malaria_2.groupby('Adm_4')['Blood_pos', 'Blood_neg', 'Blood_tot', 'Rapid_pos',
                                           'Rapid_neg', 'Rapid_tot', 'Num_houses'].sum().reset_index()
    return malaria_2

def hiv_rate(hiv, country):
    """

    :param hiv:
    :param country:
    :return:
    """
    blood_pos, blood_neg, blood_tot = [], [], []
    for i in pd.unique(hiv['DHSClust']):
        count = pd.DataFrame(hiv[hiv['DHSClust'] == i]['BloodResult'].value_counts())
        count_dict = count.reset_index().set_index('index').to_dict()['BloodResult']
        blood_neg.append(sum(count_dict[k] for k in count_dict.keys() if k == 0))
        blood_pos.append(sum(count_dict[k] for k in count_dict.keys() if k == 1))
        blood_tot.append(sum(count_dict[k] for k in count_dict.keys() if k >= 0))
    return [blood_pos, blood_neg, blood_tot]


