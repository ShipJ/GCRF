import pandas as pd
import numpy as np
from src.config import config
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import sys


def combine_cdr_dhs(a, b):
    return pd.DataFrame(a.merge(b, on=['Adm_1', 'Adm_2', 'Adm_3', 'Adm_4']))




if __name__ == '__main__':

    country = config.get_country()
    cdr = config.get_master_cdr(country)
    dhs = config.get_master_dhs(country)
    cdr_dhs = combine_cdr_dhs(cdr, dhs).dropna()

    for i in ['Adm_1', 'Adm_2', 'Adm_3', 'Adm_4']:
        cdr = cdr_dhs.groupby(i)['Vol', 'Vol_in', 'Vol_out'].sum().reset_index()
        cdr_2 = cdr_dhs.groupby(i)['Entropy', 'Med_degree', 'Pagerank', 'Introversion'].mean().reset_index()
        cdr_3 = cdr.merge(cdr_2, on=i)
        dhs = cdr_dhs.groupby(i)['BloodPosRate', 'RapidPosRate', 'DeathRate'].median().reset_index()
        cdr_dhs_2 = cdr_3.merge(dhs, on=i)

        for j in ['Vol', 'Vol_in', 'Vol_out', 'Entropy', 'Med_degree', 'Pagerank', 'Introversion']:
            for k in ['BloodPosRate', 'DeathRate']:
                a = np.array(cdr_dhs_2[j])
                b = np.array(cdr_dhs_2[k])

                outliers = np.where(a > 0)
                a = a[outliers]
                b = b[outliers]

                outliers2 = np.where(b > 0)
                a = a[outliers2]
                b = b[outliers2]

                a = a[~is_outlier(a)]
                b = b[~is_outlier(a)]

                a = a[~is_outlier(b)]
                b = b[~is_outlier(b)]

                print a, b

                print i, j, k

                print pearsonr(a, b)
                plt.scatter(a, b)
                plt.show()

                print pearsonr(a, b)
                plt.scatter(np.log(a), b)
                plt.show()








