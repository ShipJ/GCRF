"""
This module aggregated
"""

import pandas as pd
import numpy as np
from src.config import config
import sys


def model_1(path, cdr_ct, country, sum_metrics, avg_metrics):
    """
    Aggregate in the way outlined in literature (under model 1)
    :param path:
    :param cdr_ct:
    :param country:
    :return:
    """

    intersect_pop = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/pop/intersect_pop.csv' % country))
    adm4_pop = intersect_pop.groupby('Adm_4')['Pop_2010'].sum().reset_index()

    vol_sum_total = []
    vol_avg_total = []
    for adm in range(1, 192):
        adm_i = intersect_pop[intersect_pop['Adm_4'] == adm]
        metric_sum_vol, metric_avg_vol = np.zeros(8), np.zeros(7)
        if len(adm_i) > 0:
            i = 0
            # Metrics to be summed
            for metric in sum_metrics:
                vol = 0
                for index, row in adm_i.iterrows():
                    ct_vol = np.array(cdr_ct[cdr_ct['CellTowerID'] == row.CellTowerID][metric])
                    ct_pop = np.sum(intersect_pop[intersect_pop['CellTowerID'] == row.CellTowerID]['Pop_2010'])
                    if (len(ct_vol) > 0) and (ct_pop > 0):
                        proportion = row.Pop_2010/ct_pop
                        vol += (proportion * ct_vol[0])
                metric_sum_vol[i] = vol
                i += 1
            i = 0
            # Metrics to be averaged
            for metric in avg_metrics:
                vol = 0
                for index, row in adm_i.iterrows():
                    num_ct = len(adm_i)
                    ct_vol = np.array(cdr_ct[cdr_ct['CellTowerID'] == row.CellTowerID][metric])
                    ct_pop = np.sum(intersect_pop[intersect_pop['CellTowerID'] == row.CellTowerID]['Pop_2010'])
                    if (len(ct_vol) > 0) and (ct_pop > 0):
                        proportion = row.Pop_2010/ct_pop
                        vol += (proportion * ct_vol[0]) / num_ct
                metric_avg_vol[i] = vol
                i += 1
        vol_sum_total.append(metric_sum_vol)
        vol_avg_total.append(metric_avg_vol)

    sum_total = np.array(vol_sum_total)
    avg_total = np.array(vol_avg_total)


    adm_locations = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/cdr/ct_locations/adm_1234.csv' % country))

    aggregated_metrics = pd.DataFrame(pd.concat([adm_locations,
                                                 pd.DataFrame(sum_total, columns=sum_metrics),
                                                 pd.DataFrame(avg_total, columns=avg_metrics)], axis=1))
    return aggregated_metrics

if __name__ == '__main__':
    # System path to data directory
    PATH = config.get_dir()
    # Ask user for country
    country = config.get_country()

    # Column headers for metrics to be summed vs. averaged
    sum_metrics = ['Vol', 'Vol_in', 'Vol_out', 'Vol_self', 'Dur', 'Dur_in', 'Dur_out', 'Dur_self']
    avg_metrics = ['Entropy', 'Med_deg', 'Introversion', 'Pagerank', 'Eig_central', 'Deg_central', 'G_resids']

    # Currently only for all hours. Future -> add 'working' to the array to include metrics for working hours only
    for i in ['all']:
        # Get fundamental metrics
        cdr_ct = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/cdr/metrics/cdr_fundamentals_ct_%s.csv' % (country, i)))

        # Using Model 1 (reference literature), aggregate CT level data to administrative levels
        aggregated_metrics = model_1(PATH, cdr_ct, country, sum_metrics, avg_metrics)

        # Save to csv
        aggregated_metrics.to_csv(PATH+'/processed/%s/cdr/metrics/cdr_aggregate_adm_%s.csv' % (country, i), index=None)