import pandas as pd
from src.config import config

country = config.get_country()
master = config.get_master(country)
dhs = pd.DataFrame(pd.read_csv('../../data/processed/%s/dhs/Master.csv' % country))

master = pd.DataFrame(pd.concat([master, dhs[['Area(m^2)', 'Area(km^2)', 'Poverty', 'Wealth', 'Blood_pos',
                                              'Blood_neg', 'Blood_tot', 'Pos_rate', 'Rapid_pos', 'Rapid_neg',
                                              'Rapid_tot', 'Rapid_rate', 'Num_members', 'Num_houses']]], axis=1))

master.to_csv('../../data/processed/%s/Master_cdr_dhs.csv' % country, index=None)

