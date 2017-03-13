import pandas as pd
import numpy as np
from src.config import config
import sys


if __name__ == '__main__':
    PATH = config.get_dir()
    country = config.get_country()

    if country == 'civ':
        dhs = pd.DataFrame(
            pd.read_csv(PATH + '/final/%s/master_1.0.csv' % country, usecols=['Adm_1', 'Adm_2', 'Adm_3', 'Adm_4',
                                                                              'BloodPosRate',
                                                                              'RapidPosRate',
                                                                              'PosRate',
                                                                              'DeathRate',
                                                                              'DifficultyScore']))
        dist = pd.DataFrame(pd.read_csv(PATH + '/processed/%s/distance_area/dist_matrix_adm4.csv' % country))
        adm = dhs[['Adm_1', 'Adm_2', 'Adm_3', 'Adm_4']]

        spatial_lag = []
        for response in ['BloodPosRate', 'RapidPosRate', 'PosRate', 'DeathRate', 'DifficultyScore']:
            spatial_score = []
            for i in dhs['Adm_4']:
                response_i = dhs[dhs['Adm_4'] == i]

                a = dist[dist['Source'] == i]
                a_dist = a[a['Distance_km'] > 0]
                sum_weights = np.sum(np.reciprocal(a_dist['Distance_km'] ** 2))

                sum_lag = 0
                for j in a['Target']:
                    b = a[a['Target'] == j]
                    sum_lag += ((np.array(b['Distance_km'])[0] / float(sum_weights)) *
                                np.array(response_i[response])[0])
                spatial_score.append(sum_lag)
            spatial_lag.append(spatial_score)
        spatial = pd.DataFrame(np.transpose(spatial_lag), columns=['BloodPosRateSL', 'RapidPosRateSL', 'DeathRateSL',
                                                                   'PosRateSL', 'DifficultyScoreSL'])
        spatial = pd.DataFrame(pd.concat([adm, spatial], axis=1))
        spatial.to_csv(PATH + '/processed/%s/cdr/metrics/spatial_lag.csv' % country, index=None)

    elif country == 'sen':
        dhs = pd.DataFrame(
            pd.read_csv(PATH + '/final/%s/master_1.0.csv' % country, usecols=['Adm_1', 'Adm_2', 'Adm_3', 'Adm_4',
                                                                              'BloodPosRate',
                                                                              'RapidPosRate',
                                                                              'DeathRate']))

        dist = pd.DataFrame(pd.read_csv(PATH + '/processed/%s/distance_area/dist_matrix_adm4.csv' % country))
        adm = dhs[['Adm_1', 'Adm_2', 'Adm_3', 'Adm_4']]

        spatial_lag = []
        for response in ['BloodPosRate', 'RapidPosRate', 'DeathRate']:
            spatial_score = []
            for i in dhs['Adm_4']:
                response_i = dhs[dhs['Adm_4'] == i]

                a = dist[dist['Source'] == i]
                a_dist = a[a['Distance_km'] > 0]
                sum_weights = np.sum(np.reciprocal(a_dist['Distance_km'] ** 2))

                sum_lag = 0
                for j in a['Target']:
                    b = a[a['Target'] == j]
                    sum_lag += ((np.array(b['Distance_km'])[0] / float(sum_weights)) *
                                np.array(response_i[response])[0])
                spatial_score.append(sum_lag)
            spatial_lag.append(spatial_score)
        spatial = pd.DataFrame(np.transpose(spatial_lag), columns=['BloodPosRateSL', 'RapidPosRateSL', 'DeathRateSL'])
        spatial = pd.DataFrame(pd.concat([adm, spatial], axis=1))
        spatial.to_csv(PATH+'/final/%s/cdr/metrics/spatial_lag.csv' % country, index=None)











