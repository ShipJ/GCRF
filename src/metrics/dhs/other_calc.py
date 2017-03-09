import pandas as pd
import numpy as np

data = pd.DataFrame(pd.read_csv('../../../data/processed/sen/dhs/dhs_locations/dhs_wealth_poverty.csv'))

data2 = pd.DataFrame(pd.read_csv('../../../data/processed/sen/dhs/dhs_locations/dhs_adm1234.csv'))



cdr = pd.DataFrame(pd.read_csv('../../../data/final/sen/cdr.csv'))

other = data.groupby('Adm_3')['poverty_rate', 'z_median', 'pop_1km'].mean().reset_index()

cdr_sum = cdr.groupby('Adm_3')['Vol', 'Vol_in', 'Vol_out', 'Dur', 'Dur_in', 'Dur_out'].sum().reset_index()
cdr_mean = cdr.groupby('Adm_3')['Entropy', 'Introversion', 'Med_degree', 'Degree_centrality',
                                'EigenvectorCentrality', 'Pagerank'].mean().reset_index()
cdr_grav = pd.DataFrame(pd.read_csv('../../../data/processed/sen/cdr/metrics/gravity.csv')).dropna()
cdr_grav = cdr_grav.groupby('Adm_3')['Residuals'].mean().reset_index()

cdr = cdr_sum.merge(cdr_mean, on='Adm_3').merge(cdr_grav, on='Adm_3').merge(other, on='Adm_3')
dhs = pd.DataFrame(pd.read_csv('../../../data/final/sen/dhs.csv'))
dhs = dhs.groupby('Adm_3')['BloodPosRate', 'RapidPosRate', 'DeathRate'].mean().reset_index()


cdr = pd.DataFrame(cdr.merge(dhs, on='Adm_3'))
# cdr.to_csv('../../../data/final/sen/master.csv', index=None)

adm_3 = data2[['Adm_1', 'Adm_2', 'Adm_3', 'Adm_4']].drop('Adm_4', axis=1).drop_duplicates().reset_index().drop('index', axis=1)
cdr = cdr.merge(adm_3, on='Adm_3')
cdr.to_csv('../../../data/final/sen/master.csv', index=None)