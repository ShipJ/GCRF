"""
This module
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, norm, boxcox
import matplotlib.pyplot as plt
from src.config import config
import sys

# Figure parameters
plt.rcParams['figure.facecolor']='white'


def outliers(df, iv, dv, adm):
    outliers_iv = list(df[df[iv] > 3.5 * np.mean(df[iv])][adm])
    outliers_dv = list(df[df[dv] > 3.5 * np.mean(df[dv])][adm])
    for i in (outliers_iv+outliers_dv):
        df = df[df[adm] != i]
    return df


def outliers2(df, iv, dv, adm, thresh=3.5):

    if len(df.shape) == 1:
        df = df[:, None]
    median1 = np.median(df[iv])
    median2 = np.median(df[dv])
    diff1 = (df[iv] - median1)**2
    diff2 = (df[iv] - median2)**2
    diff1 = np.sqrt(diff1)
    diff2 = np.sqrt(diff2)
    med_abs_deviation1 = np.median(diff1)
    med_abs_deviation2 = np.median(diff2)
    modified_z_score1 = 0.6745 * diff1 / med_abs_deviation1
    modified_z_score2 = 0.6745 * diff2 / med_abs_deviation2
    a = pd.DataFrame()
    a[adm] = df[adm]
    a[iv] = modified_z_score1
    a[dv] = modified_z_score2
    outliers_iv = list(a[a[iv] > thresh][adm])
    outliers_dv = list(a[a[dv] > thresh][adm])
    for i in (outliers_iv+outliers_dv):
        df = df[df[adm] != i]
    return df


def z_score(df):
    return (df-np.mean(df))/np.std(df)


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

def aggregate_median(data, features, adm):
    return data.groupby(adm)[features].mean().reset_index()

def transform(df, cdr, dhs, model, adm):
    df = df[df[dhs] > 0]
    df[cdr], df[dhs] = df[cdr]+abs(min(df[cdr]))+1, df[dhs]+abs(min(df[dhs]))+1
    df = df.dropna()

    if model == 1:
        iv, dv = df[cdr], df[dhs]
        df[cdr], df[dhs] = z_score(iv), z_score(dv)
        return df
    elif model == 2:
        df = outliers2(df, k, j, adm)
        df[cdr], df[dhs] = z_score(df[cdr]), z_score(df[dhs])
        return df
    elif model == 3:
        df[cdr], df[dhs] = np.log(df[cdr]+1), np.log(df[dhs]+1)
        df[cdr], df[dhs] = z_score(df[cdr]), z_score(df[dhs])
        return df
    elif model == 4:
        df[cdr], df[dhs] = np.log(df[cdr]+1), np.log(df[dhs]+1)
        df = outliers2(df, k, j, adm)
        df[cdr], df[dhs] = z_score(df[cdr]), z_score(df[dhs])
        return df
    elif model == 5:
        df[cdr] = np.log(df[cdr]+1)
        df = outliers2(df, k, j, adm)
        df[cdr], df[dhs] = z_score(df[cdr]), z_score(df[dhs])
        return df
    elif model == 6:
        df[dhs] = np.log(df[dhs]+1)
        df = outliers2(df, k, j, adm)
        df[cdr], df[dhs] = z_score(df[cdr]), z_score(df[dhs])
        return df
    elif model == 7:
        df = outliers2(df, k, j, adm)
        df[cdr], df[dhs] = z_score(np.log(df[cdr])), z_score(np.log(df[dhs]))
        return df


def group_scatter(iv, dv, lower, upper, group):
    plt.scatter(iv[lower], dv[lower], c='white', label='Non-%s'%group)
    plt.scatter(iv[upper], dv[upper], c='black', label=group)
    plt.grid(), plt.legend(scatterpoints=1), plt.xlabel(k), plt.ylabel(j)
    plt.show()


if __name__ == '__main__':
    PATH = config.get_dir()
    country = config.get_country()
    adms = config.get_headers(country, 'adm')
    cdr_features = config.get_headers(country, 'cdr')
    dhs_features = config.get_headers(country, 'dhs')
    group_features = config.get_headers(country, 'group')

    group = 'None'
    models = ['No outliers, no transformation ', 'Outliers, no transformation ',
              'No outliers, log-log-transformation ', 'Outliers, log-log-transformation ',
              'Outliers, log-linear-transformation ', 'Outliers, linear-log-transformation ',
              'Outliers, then log-log ', 'root root', 'noth root', 'root noth', 'root log']

    data = pd.DataFrame(pd.read_csv(PATH+'/final/%s/master_SL.csv' % country))

    for adm in np.setdiff1d(adms, ['Adm_1']):
        dhs_adm = aggregate_mean(data, dhs_features, adm)
        cdr_sum_adm = aggregate_sum(data, cdr_features[:6], adm)
        cdr_mean_adm = aggregate_mean(data, cdr_features[6:], adm)
        cdr_adm = cdr_sum_adm.merge(cdr_mean_adm, on=adm)

        if group != 'None':
            if country == 'civ':
                group_adm = aggregate_median(data, group_features, adm)
            else:
                group_adm = []
        else:
            group_adm = []

        for j in dhs_features:
            dhs_adm_j = dhs_adm[[adm, j]]
            for k in cdr_features:
                cdr_adm_k = cdr_adm[[adm, k]]
                merged = dhs_adm_j.merge(cdr_adm_k, on=adm, how='outer')
                if group != 'None':
                    if country == 'civ':
                        merged = merged.merge(group_adm, on=adm).dropna()

                for i in range(7):
                    print models[i], adm, j, k
                    transformed = transform(merged.copy(), k, j, i+1, adm)
                    iv, dv = transformed[k], transformed[j]
                    print pearsonr(iv, dv)
                    if group != 'None':
                        if country == 'civ':
                            lower_half = np.array(transformed[group] <= np.median(transformed[group]))
                            upper_half = np.array(transformed[group] > np.median(transformed[group]))
                            group_scatter(iv, dv, lower_half, upper_half, group)
                    else:
                        plt.scatter(iv, dv,c='black')
                        plt.xlabel(k), plt.ylabel(j), plt.title(k+' vs. '+j+' : '+adm+'\n'+'('+models[i]+')')
                        plt.grid()
                        plt.show()
