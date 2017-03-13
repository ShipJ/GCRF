import pandas as pd
from src.config import config

if __name__ == '__main__':
    PATH = config.get_dir()
    country = config.get_country()

    cdr = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/cdr/metrics/cdr_derived_adm.csv' % country))
    dhs = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/dhs/dhs_fundamentals_adm.csv' % country))
    spatial_lag = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/cdr/spatial_lag.csv' % country))
    # other = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/dhs/other_fundamentals_adm.csv' % country))

    master = pd.DataFrame(cdr.merge(dhs, on=['Adm_1', 'Adm_2', 'Adm_3', 'Adm_4'], how='outer'))
    master.to_csv(PATH+'/final/%s/master_1.0.csv' % country, index=None)