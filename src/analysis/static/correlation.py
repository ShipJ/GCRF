"""
TODO: Model 1: No data transformation, minimal outlier detection applied, no distance weighted function,
no sparsification applied either - i.e. using all data, all links, at all times. 
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from src.config import config


def outliers(df):
    outliers = np.where(df < 1.5*np.mean(df))
    return outliers

def z_score(df):
    z_scores = np.array((df-np.mean(df))/np.std(df))
    return z_scores

if __name__ == '__main__':
    source = config.get_dir()
    country = config.get_country()
    adms = config.get_headers(country, 'adm')
    cdr_features = config.get_headers(country, 'cdr')
    dhs_features = config.get_headers(country, 'dhs')

    data = pd.DataFrame(pd.read_csv(source+'/final/%s/master_2.0.csv' % country))

    for i in adms:

        dhs_i = data.groupby(i)[dhs_features].mean().reset_index()
        cdr_sum_i = data.groupby(i)[cdr_features[:6]].sum().reset_index()
        cdr_mean_i = data.groupby(i)[cdr_features[6:]].mean().reset_index()
        cdr_i = cdr_sum_i.merge(cdr_mean_i, on=i)

        for j in dhs_features:
            dvs = dhs_i[[i, j]]

            for k in cdr_features:
                ivs = cdr_i[[i, k]]
                merged = dvs.merge(ivs, on=i, how='outer').dropna()

                print 'No outliers, no transformation ', i, j, k
                iv1 = np.abs(merged[k])
                dv1 = np.abs(merged[j])
                iv1 = z_score(iv1)
                dv1 = z_score(dv1)
                print pearsonr(iv1, dv1)
                plt.scatter(iv1, dv1)
                plt.show()

                print 'Outliers, no transformation ', i, j, k
                iv2 = np.abs(np.array(merged[k]))
                dv2 = np.abs(np.array(merged[j]))
                non_outliers_iv2 = outliers(iv2)
                iv2 = iv2[non_outliers_iv2]
                dv2 = dv2[non_outliers_iv2]
                non_outliers_dv2 = outliers(dv2)
                iv2 = iv2[non_outliers_dv2]
                dv2 = dv2[non_outliers_dv2]

                iv2 = z_score(iv2)
                dv2 = z_score(dv2)

                print pearsonr(iv2, dv2)
                plt.scatter(iv2, dv2)
                plt.show()

                print 'No outliers, log-log-transformation ', i, j, k
                iv3 = np.log(np.abs(merged[k]))
                dv3 = np.log(np.abs(merged[j]))
                iv3 = z_score(iv3)
                dv3 = z_score(dv3)
                print pearsonr(iv3, dv3)
                plt.scatter(iv3, dv3)
                plt.show()

                print 'Outliers, log-log-transformation ', i, j, k
                iv4 = np.log(np.array(np.abs(merged[k]))+1)
                dv4 = np.log(np.array(np.abs(merged[j]))+1)
                non_outliers_iv4 = outliers(iv4)
                iv4 = iv4[non_outliers_iv4]
                dv4 = dv4[non_outliers_iv4]
                non_outliers_dv4 = outliers(dv4)
                iv4 = iv4[non_outliers_dv4]
                dv4 = dv4[non_outliers_dv4]
                iv4 = z_score(iv4)
                dv4 = z_score(dv4)
                print pearsonr(iv4, dv4)
                plt.scatter(iv4, dv4)
                plt.show()

                print 'Outliers, log-linear-transformation ', i, j, k
                iv5 = np.log(np.array(np.abs(merged[k]))+1)
                dv5 = np.array(np.abs(merged[j]))
                non_outliers_iv5 = outliers(iv5)
                iv5 = iv5[non_outliers_iv5]
                dv5 = dv5[non_outliers_iv5]
                non_outliers_dv5 = outliers(dv5)
                iv5 = iv5[non_outliers_dv5]
                dv5 = dv5[non_outliers_dv5]
                iv5 = z_score(iv5)
                dv5 = z_score(dv5)
                print pearsonr(iv5, dv5)
                plt.scatter(iv5, dv5)
                plt.show()

                print 'Outliers, linear-log-transformation ', i, j, k
                iv6 = np.array(np.abs(merged[k]))
                dv6 = np.log(np.array(np.abs(merged[j]))+1)
                non_outliers_iv6 = outliers(iv6)
                iv6 = iv6[non_outliers_iv6]
                dv6 = dv6[non_outliers_iv6]
                non_outliers_dv6 = outliers(dv6)
                iv6 = iv6[non_outliers_dv6]
                dv6 = dv6[non_outliers_dv6]
                iv6 = z_score(iv6)
                dv6 = z_score(dv6)
                print pearsonr(iv6, dv6)
                plt.scatter(iv6, dv6)
                plt.show()
