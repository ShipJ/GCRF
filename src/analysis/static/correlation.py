"""
TODO: Model 1: No data transformation, minimal outlier detection applied, no distance weighted function,
no sparsification applied either - i.e. using all data, all links, at all times. 
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from src.config import config

def mean_absolute_deviation(df, i, j):
    z_score_i = np.array((0.6745*(df[i]-np.median(df[i]))) / np.median(np.abs(df[i]-np.median(df[i]))))
    z_score_j = np.array((0.6745 * (df[j] - np.median(df[j]))) / np.median(np.abs(df[j] - np.median(df[j]))))
    outliers_i = np.where(np.abs(z_score_i) > 3.5)
    outliers_j = np.where(np.abs(z_score_j) > 3.5)
    df[i].ix[outliers_i] = np.nan
    df[j].ix[outliers_j] = np.nan
    return df


def z_outliers(df):
    outliers = df[df > 3*np.mean(df)]
    return outliers

def z_score(df):
    z_scores = np.array((df-np.mean(df))/np.std(df))
    return z_scores

source = config.get_dir()
data = pd.DataFrame(pd.read_csv(source+'/final/civ/master.csv')).dropna()
data['log_vol'] = np.log(data['Vol'])
data['vol_pp'] = data['Vol']/data['Pop_2010']
data['log_pop_dense'] = np.log(data['Pop_2010']/data['Area_km2'])

for i in ['Adm_4']:

    dhs_i = data.groupby(i)['BloodPosRate', 'RapidPosRate', 'HIVPosRate',
                            'DeathRate', 'HealthAccessDifficulty', 'Z_Med'].mean().reset_index()
    cdr_sum_i = data.groupby(i)['Vol', 'Vol_in', 'Vol_out', 'Dur', 'Dur_in', 'Dur_out'].sum().reset_index()
    cdr_mean_i = data.groupby(i)['Entropy', 'Introversion', 'Med_degree', 'Degree_centrality',
                               'EigenvectorCentrality','Residuals', 'log_vol',
                                 'vol_pp', 'log_pop_dense'].mean().reset_index()
    cdr_i = cdr_sum_i.merge(cdr_mean_i, on=i)

    for j in ['BloodPosRate', 'RapidPosRate', 'HIVPosRate', 'DeathRate', 'HealthAccessDifficulty', 'Z_Med']:

        for k in ['Vol', 'Vol_in', 'Vol_out', 'Entropy', 'Introversion', 'Med_degree', 'Degree_centrality',
                  'EigenvectorCentrality','Residuals', 'log_vol', 'vol_pp', 'log_pop_dense']:
            a = cdr_i.groupby(i)[k].sum().reset_index()
            b = a.merge(dhs_i.groupby(i)[j].sum().reset_index(), on=i).dropna().as_matrix()

            print b

            plt.scatter(np.log(b[:, 1]), b[:, 2]+1)
            print pearsonr(np.log(b[:, 1]+1), b[:, 2]+1)
            plt.show()

            a = z_score(a)
            b = z_score(b)


            plt.scatter(a[:, 1],b[:, 2])
            plt.show()





            # a = cdr_i.groupby(i)[k].sum().reset_index()
            # b = a.merge(dhs_i.groupby(i)[j].sum().reset_index(), on=i)
            # compare = mean_absolute_deviation(b, j, k).dropna()
            #
            # c = np.log(np.array(compare[j]))
            # d = np.log(np.array(compare[k]))
            #
            # # Get rid of zero data
            # zero_c = np.where(c != 0)
            # c = c[zero_c]
            # d = d[zero_c]
            # zero_d = np.where(d!=0)
            # c = c[zero_d]
            # d = d[zero_d]
            #
            # c = z_score(c)
            # d = z_score(d)
            #
            # print i, j, k
            # print pearsonr(c, d)
            # plt.scatter(c, d)
            # plt.show()




