import pandas as pd
import numpy as np
from src.config import config






if __name__ == '__main__':
    country = config.get_country()
    master = pd.DataFrame(pd.read_csv('../../data/processed/%s/dhs/master_dhs.csv' % country))

    mode = lambda x: x.value_counts().index[0]



    other = pd.DataFrame(pd.read_csv('../../data/processed/%s/dhs/wealth/DHSData.csv' % country,
                                     usecols=['UrbRur', 'Poverty', 'Z_Med', 'Capital', 'Pop_1km',
                                              'Adm_1', 'Adm_2', 'Adm_3', 'Adm_4']))

    urbrurcapital = other.groupby('Adm_4')['UrbRur', 'Capital'].agg(mode).reset_index()
    pov_z_med = other.groupby('Adm_4')['Poverty', 'Z_Med', 'Pop_1km'].mean().reset_index()

    master_cdr_dhs_other = master.merge(urbrurcapital,
                                        on='Adm_4', how='outer').merge(pov_z_med,
                                                                       on='Adm_4', how='outer')
    pop = pd.DataFrame(pd.read_csv('../../data/processed/%s/pop/IntersectPop.csv' % country))
    pop = pop.groupby('Adm_4')['Pop_2010', 'Pop_2014'].sum().reset_index()

    master = pd.DataFrame(master_cdr_dhs_other.merge(pop, on='Adm_4', how='outer'))

    master.to_csv('../../data/processed/%s/correlation/master_cdr_dhs_other.csv' % country, index=None)









