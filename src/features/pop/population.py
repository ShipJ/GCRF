import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = '/Users/JackShipway/Desktop/UCLProject/Data/IvoryCoast'

pop = pd.DataFrame(pd.read_csv(path+'/Population/IntersectPop.csv'))
pop_10_1 = pop.groupby('Adm_1')['Pop_2010'].sum().reset_index()
pop_10_2 = pop.groupby('Adm_2')['Pop_2010'].sum().reset_index()
pop_10_3 = pop.groupby('Adm_3')['Pop_2010'].sum().reset_index()
pop_10_4 = pop.groupby('Adm_4')['Pop_2010'].sum().reset_index()
pop_14_1 = pop.groupby('Adm_1')['Pop_2014'].sum().reset_index()
pop_14_2 = pop.groupby('Adm_2')['Pop_2014'].sum().reset_index()
pop_14_3 = pop.groupby('Adm_3')['Pop_2014'].sum().reset_index()
pop_14_4 = pop.groupby('Adm_4')['Pop_2014'].sum().reset_index()

dhs = pd.DataFrame(pd.read_csv(path+'/DHS/metrics/DHSData.csv'))
malaria_1 = dhs.groupby('Adm_1')['MalariaPerPop'].mean().reset_index()

activity = pd.DataFrame(pd.read_csv(path+'/CDR/static_metrics/Activity/total_activity.csv'))
entropy = pd.DataFrame(pd.read_csv(path+'/CDR/static_metrics/Entropy/entropy_adm1.csv'))

print entropy

