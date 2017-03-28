"""
Run once to obtain a merged CDR/DHS file ready for modelling
"""


import pandas as pd
from src.config import config

if __name__ == '__main__':


    master = pd.DataFrame(pd.read_csv('../../data/processed/%s/master_pop.csv' % country))
    print master


    gravity = pd.DataFrame(pd.read_csv('../../data/processed/%s/cdr/metrics/gravity.csv' % country)).dropna()

    g_residual_adm4 = gravity.groupby('Adm_4')['Residuals'].mean().reset_index()

    grav = pd.DataFrame()
    grav['Adm_4'] = range(1,192)
    grav = grav.merge(g_residual_adm4, on='Adm_4', how='outer')

    master = pd.DataFrame(master.merge(grav, on='Adm_4'))

    master.to_csv('../../data/processed/%s/master_pop2.csv' % country, index=None)

    import pandas as pd
    from src.config import config

    if __name__ == '__main__':
        PATH = config.get_dir()
        country = config.get_country()

        cdr = pd.DataFrame(pd.read_csv(PATH + '/processed/%s/cdr/metrics/cdr_derived_bts_all.csv' % country))
        dhs = pd.DataFrame(pd.read_csv(PATH + '/processed/%s/dhs/dhs_fundamentals_adm.csv' % country))
        spatial_lag = pd.DataFrame(pd.read_csv(PATH + '/processed/%s/cdr/metrics/spatial_lag_adm_all.csv' % country))
        # be sure to increment x in master_x.0.csv if you include 'other' data below
        # other = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/dhs/other_fundamentals_adm.csv' % country))

        master = pd.DataFrame(cdr.merge(dhs,
                                        on=['Adm_1', 'Adm_2', 'Adm_3', 'Adm_4'],
                                        how='outer')).merge(spatial_lag,
                                                            on=['Adm_1', 'Adm_2', 'Adm_3', 'Adm_4'],
                                                            how='outer')
        master.to_csv(PATH + '/final/%s/master_2.0.csv' % country, index=None)












