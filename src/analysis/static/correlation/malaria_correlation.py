import pandas as pd
import numpy as np
import sys
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # DHS data
    child_mort = pd.DataFrame(pd.read_csv('Data/IvoryCoast/DHS/Extracted/deathPerBirth.csv'))
    # Population data
    pop = pd.DataFrame(pd.read_csv('Data/IvoryCoast/Essential/Pop_Per_Adm_1234.csv', usecols=['DHSClust', 'UrbRur', 'Pop_1km', 'Adm_1', 'Adm_2', 'Adm_3', 'Adm_4']))

    for i in np.setdiff1d(child_mort['Adm_1'], pop.groupby('Adm_1')['Pop_1km'].sum().reset_index()['Adm_1']):
        child_mort = child_mort[child_mort['Adm_1'] != i]

    rate = np.array(child_mort['six2OneYearPB']) / np.array(pop.groupby('Adm_4')['Pop_1km'].sum())
    child_mort['rate'] = rate

    activity_1 = pd.DataFrame(pd.read_csv('Data/IvoryCoast/CDR/Metrics/total_activity_adm_1.csv'))
    activity_1['normed'] = activity_1['Vol'] / max(activity_1['Vol'])
    activity_1['dur_normed'] = activity_1['Dur'] / max(activity_1['Dur'])

    for i in np.setdiff1d(activity_1['Adm_1'], child_mort['Adm_1']):
        activity_1 = activity_1[activity_1['Adm_1'] != i]

    for i in np.setdiff1d(child_mort['Adm_1'], activity_1['Adm_1']):
        child_mort = child_mort[child_mort['Adm_1'] != i]

    print pearsonr(activity_1['Vol'], child_mort.groupby('Adm_1')['rate'].mean())
    print pearsonr(activity_1['Dur'], child_mort.groupby('Adm_1')['rate'].mean())

    plt.scatter(activity_1['Vol'], child_mort.groupby('Adm_1')['rate'].mean())
    plt.show()
    plt.scatter(activity_1['Dur'], child_mort.groupby('Adm_1')['rate'].mean())
    plt.show()

    # for i in np.setdiff1d(child_mort['Adm_4'], pop.groupby('Adm_4')['Pop_1km'].sum().reset_index()['Adm_4']):
    #     child_mort = child_mort[child_mort['Adm_4'] != i]
    #
    # # rate = np.array(child_mort['six2OneYearPB']) / np.array(pop.groupby('Adm_4')['Pop_1km'].sum())
    # # child_mort['rate'] = rate
    #
    # activity_4 = pd.DataFrame(pd.read_csv('Data/IvoryCoast/CDR/Metrics/total_activity_adm_4.csv'))
    # activity_4['normed'] = activity_4['Vol'] / max(activity_4['Vol'])
    # activity_4['dur_normed'] = activity_4['Dur'] / max(activity_4['Dur'])
    #
    # for i in np.setdiff1d(activity_4['Adm_4'], child_mort['Adm_4']):
    #     activity_4 = activity_4[activity_4['Adm_4'] != i]
    #
    # for i in np.setdiff1d(child_mort['Adm_4'], activity_4['Adm_4']):
    #     child_mort = child_mort[child_mort['Adm_4'] != i]
    #
    #
    # print pearsonr(activity_4['Vol'], child_mort.groupby('Adm_4')['neoNatalPB'].mean())
    # print pearsonr(activity_4['Dur'], child_mort.groupby('Adm_4')['neoNatalPB'].mean())
    #
    # plt.scatter(activity_4['Vol'], child_mort.groupby('Adm_4')['neoNatalPB'].mean())
    # plt.show()
