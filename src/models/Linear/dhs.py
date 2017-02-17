import pandas as pd
import numpy as np
from scipy import stats

dhs = pd.DataFrame(pd.read_csv('/Users/JackShipway/Desktop/dhs_attributes.csv'))

print dhs
urb_rur = pd.get_dummies(dhs['UrbRur'])
dhs['UrbRur'] = urb_rur['Urban']

poverty = dhs.groupby('Adm_4')['Poverty'].median()
wealth = dhs.groupby('Adm_4')['Z_Med'].median()
urb_rur = dhs.groupby('Adm_4')['UrbRur'].median()

data = pd.DataFrame()
data['poverty'] = poverty
data['wealth'] = wealth
data['urb_rur'] = urb_rur

data = data.reset_index()

data.to_csv('/Users/JackShipway/Desktop/add.csv', index=None)






