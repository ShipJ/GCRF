import pandas as pd
import numpy as np
import sys
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # DHS data
    hiv_rate = pd.DataFrame(pd.read_csv('Data/IvoryCoast/Essential/HIV.csv'))
    # Population data
    pop = pd.DataFrame(pd.read_csv('Data/IvoryCoast/Essential/Pop_Per_Adm_1234.csv', usecols=['DHSClust', 'UrbRur', 'Pop_1km', 'Adm_1', 'Adm_2', 'Adm_3', 'Adm_4']))

    for i in np.setdiff1d(hiv_rate['DHSClustID'], pop['DHSClust']):
        hiv_rate = hiv_rate[hiv_rate['DHSClustID'] != i]

    for i in np.setdiff1d(pop['DHSClust'], hiv_rate['DHSClustID']):
        pop = pop[pop['DHSClust'] != i]

    rate = np.array(hiv_rate['Result']) / np.array(pop['Pop_1km'])
    hiv_rate['rate'] = rate
    hiv_rate['Adm_1'] = np.array(pop['Adm_1'])
    hiv_rate['Adm_2'] = np.array(pop['Adm_2'])
    hiv_rate['Adm_3'] = np.array(pop['Adm_3'])
    hiv_rate['Adm_4'] = np.array(pop['Adm_4'])


    activity_1 = pd.DataFrame(pd.read_csv('Data/IvoryCoast/CDR/metrics/total_activity_adm_1.csv'))
    activity_1['normed'] = activity_1['Vol'] / max(activity_1['Vol'])
    activity_1['dur_normed'] = activity_1['Dur'] / max(activity_1['Dur'])

    print pearsonr(activity_1['Vol'][2:], hiv_rate.groupby('Adm_1')['rate'].mean()[2:])
    print pearsonr(activity_1['Dur'][2:], hiv_rate.groupby('Adm_1')['rate'].mean()[2:])

    activity_2 = pd.DataFrame(pd.read_csv('Data/IvoryCoast/CDR/metrics/total_activity_adm_2.csv'))
    activity_2['normed'] = activity_2['Vol'] / max(activity_2['Vol'])
    activity_2['dur_normed'] = activity_2['Dur'] / max(activity_2['Dur'])

    print pearsonr(activity_2['Vol'][5:], hiv_rate.groupby('Adm_2')['rate'].mean()[5:])
    print pearsonr(activity_2['Dur'][5:], hiv_rate.groupby('Adm_2')['rate'].mean()[5:])

    ''' adm_3 '''
    activity_3 = pd.DataFrame(pd.read_csv('Data/IvoryCoast/CDR/metrics/total_activity_adm_3.csv'))
    activity_3['normed'] = activity_3['Vol'] / max(activity_3['Vol'])
    activity_3['dur_normed'] = activity_3['Dur'] / max(activity_3['Dur'])

    for i in np.setdiff1d(activity_3['Adm_3'], hiv_rate.groupby('Adm_3')['rate'].mean().reset_index()['Adm_3']):
        activity_3 = activity_3[activity_3['Adm_3'] != i]

    print pearsonr(activity_3['Vol'][10:], hiv_rate.groupby('Adm_3')['rate'].mean()[10:])
    print pearsonr(activity_3['Dur'][10:], hiv_rate.groupby('Adm_3')['rate'].mean()[10:])

    plt.scatter(activity_3['Vol'], hiv_rate.groupby('Adm_3')['rate'].mean())
    plt.show()
    plt.scatter(activity_3['Dur'], hiv_rate.groupby('Adm_3')['rate'].mean())
    plt.show()

    ''' adm_4 '''
    activity_4 = pd.DataFrame(pd.read_csv('Data/IvoryCoast/CDR/metrics/total_activity_adm_4.csv'))
    activity_4['normed'] = activity_4['Vol'] / max(activity_4['Vol'])
    activity_4['dur_normed'] = activity_4['Dur'] / max(activity_4['Dur'])

    for i in np.setdiff1d(activity_4['Adm_4'], hiv_rate.groupby('Adm_4')['rate'].mean().reset_index()['Adm_4']):
        activity_4 = activity_4[activity_4['Adm_4'] != i]

    for i in np.setdiff1d(hiv_rate.groupby('Adm_4')['rate'].mean().reset_index()['Adm_4'], activity_4['Adm_4']):
        hiv_rate = hiv_rate[hiv_rate['Adm_4'] != i]

    print pearsonr(activity_4['Vol'], hiv_rate.groupby('Adm_4')['rate'].mean())
    print pearsonr(activity_4['Dur'], hiv_rate.groupby('Adm_4')['rate'].mean())

    plt.scatter(activity_4['Vol'], hiv_rate.groupby('Adm_4')['rate'].mean())
    plt.show()
    plt.scatter(activity_4['Dur'], hiv_rate.groupby('Adm_4')['rate'].mean())
    plt.show()









