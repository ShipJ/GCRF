"""
TODO: Model 1: No data transformation, minimal outlier detection applied, no distance weighted function,
no sparsification applied either - i.e. using all data, all links, at all times. 
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, norm
import matplotlib.pyplot as plt
from src.config import config
import sys


def outliers(df):
    outliers = np.where(df < 3.29 *np.mean(df))
    return outliers

def z_score(df):
    return np.array((df-np.mean(df))/np.std(df))

def corr_pvalue_conf(df):

    numeric_df = df.dropna()._get_numeric_data()
    cols = numeric_df.columns
    mat = numeric_df.values
    arr = np.zeros((len(cols),len(cols)), dtype=object)

    for xi, x in enumerate(mat.T):
        for yi, y in enumerate(mat.T[xi:]):
            pmcc_pval = np.array(map(lambda _: round(_,3), pearsonr(x,y)))
            z = np.arctanh(pmcc_pval[0])
            sigma = (1 / ((len(numeric_df.index) - 3) ** 0.5))
            cint = z + np.array([-1, 1]) * sigma * norm.ppf((1 + 0.95) / 2)
            cint = [round(i, 3) for i in cint]
            pmcc_pval_conf = np.concatenate([pmcc_pval, cint])
            arr[xi, yi+xi] = pmcc_pval_conf
            arr[yi+xi, xi] = arr[xi, yi+xi]
    return pd.DataFrame(arr, index=cols, columns=cols)

def aggregate_sum(data, features, adm):
    return data.groupby(adm)[features].sum().reset_index()

def aggregate_mean(data, features, adm):
    return data.groupby(adm)[features].mean().reset_index()

if __name__ == '__main__':
    PATH = config.get_dir()
    country = config.get_country()
    adms = config.get_headers(country, 'adm')
    cdr_features = config.get_headers(country, 'cdr')
    dhs_features = config.get_headers(country, 'dhs')
    group_features = config.get_headers(country, 'group')

    # Grab data
    data = pd.DataFrame(pd.read_csv(PATH+'/final/%s/master_3.0.csv' % country))

    # For each adm
    for adm in adms:

        # Aggregate at that level
        dhs_adm = aggregate_mean(data, adm, dhs_features)
        cdr_sum_adm = aggregate_sum(data, cdr_features[:6], adm)
        cdr_mean_adm = aggregate_mean(data, cdr_features[6:], adm)
        cdr_adm = cdr_sum_adm.merge(cdr_mean_adm, on=adm)

        # For grouping variables at a later stage
        group_adm =

        print cdr_adm
        print dhs_adm


        sys.exit()

        # For all DHS metrics
        for j in dhs_features:
            dvs = dhs_adm[[adm, j]]

            # For all CDR metrics
            for k in cdr_features:

                # Grab independent variable (CDR)
                ivs = cdr_adm[[adm, k]]
                # Merge with dependent variable, remove non-existent data
                merged = dvs.merge(ivs, on=adm, how='outer').dropna()

                # Correlation 1
                print 'No outliers, no transformation ', adm, j, k
                iv1 = np.abs(merged[k])
                dv1 = np.abs(merged[j])
                iv1 = z_score(iv1)
                dv1 = z_score(dv1)
                print pearsonr(iv1, dv1)
                plt.scatter(iv1, dv1)
                plt.show()

                # Correlation 2
                print 'Outliers, no transformation ', adm, j, k
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
                plt.scatter(iv2, dv2)
                plt.show()

                # Correlation 3
                print 'No outliers, log-log-transformation ', adm, j, k
                iv3 = np.log(np.abs(merged[k]))
                dv3 = np.log(np.abs(merged[j]))
                iv3 = z_score(iv3)
                dv3 = z_score(dv3)
                print pearsonr(iv3, dv3)
                plt.scatter(iv3, dv3)
                plt.show()

                # Correlation 4
                print 'Outliers, log-log-transformation ', adm, j, k
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

                # Correlation 5
                print 'Outliers, log-linear-transformation ', adm, j, k
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

                # Correlation 6
                print 'Outliers, linear-log-transformation ', adm, j, k
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
