import pandas as pd
from src.config import config

def merge_other(dhs, other):
    """

    :param cdr_dhs:
    :param other:
    :return:
    """
    mode = lambda x: x.value_counts().index[0]

    urbrurcapital = other.groupby('Adm_4')['UrbRur', 'Capital'].agg(mode).reset_index()
    pov_z_med = other.groupby('Adm_4')['Poverty', 'Z_Med', 'Pop_1km'].mean().reset_index()
    master_dhs_other = dhs.merge(urbrurcapital, on='Adm_4', how='outer').merge(pov_z_med, on='Adm_4', how='outer')
    return pd.DataFrame(master_dhs_other)

if __name__ == '__main__':
    PATH = config.get_dir()
    country = config.get_country()

    dhs = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/dhs/dhs_fundamentals_adm.csv' % country))
    other = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/dhs/wealth/dhs_wealth_poverty.csv' % country))

    dhs_other = dhs.merge(other, on=['Adm_1', 'Adm_2', 'Adm_3', 'Adm_4'])
    dhs_other.to_csv(PATH+'/processed/%s/dhs/dhs_derived_adm.csv' % country, index=None)