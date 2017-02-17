import pandas as pd
import numpy as np
import sys
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt

if __name__ == "__main__":

    path = "/Users/JackShipway/Desktop/UCLProject/Project1-Health"

    # DHS data
    dhs = pd.DataFrame(pd.read_csv('DHSData.csv'))

    # CDR data (for each adm)
    entropy_adm_1 = pd.DataFrame(pd.read_csv(path+"/entropy_adm_1.csv"))
    entropy_adm_2 = pd.DataFrame(pd.read_csv(path + "/entropy_adm_2.csv"))

    # Malaria
    malaria_adm_1 = dhs.groupby('Adm_1')['MalariaPerPop'].mean().reset_index()
    malaria_adm_2 = dhs.groupby('Adm_2')['MalariaPerPop'].mean().reset_index()



    # Adm_1
    # non_outliers = np.intersect1d(np.where(entropy['Entropy'] > 150000), np.where(entropy['Entropy'] < 1500000))
    # plt.scatter(malaria_adm_1['MalariaPerPop'].ix[non_outliers], entropy['Entropy'].ix[non_outliers])
    # plt.show()
    # print pearsonr(malaria_adm_1['MalariaPerPop'].ix[non_outliers], entropy['Entropy'].ix[non_outliers])

    plt.scatter(np.array(entropy_adm_2['Entropy']),
                np.array(malaria_adm_2['MalariaPerPop']))
    plt.show()

    # Adm_2
    non_outliers = np.intersect1d(np.where(entropy_adm_2['Entropy'] > 40000), np.where(entropy_adm_2['Entropy'] < 1500000))

    plt.scatter(np.array(np.log(entropy_adm_2['Entropy'])[non_outliers]), np.array(malaria_adm_2['MalariaPerPop'])[non_outliers])
    plt.show()
    plt.scatter(np.array(entropy_adm_2['Entropy'])[non_outliers], np.array(malaria_adm_2['MalariaPerPop'])[non_outliers])
    plt.show()

    print pearsonr(np.log(entropy_adm_2['Entropy'].ix[non_outliers]), malaria_adm_2['MalariaPerPop'].ix[non_outliers])