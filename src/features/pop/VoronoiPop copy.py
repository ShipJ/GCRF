import pandas as pd
import numpy as np

if __name__ == "__main__":

    ''' Ivory Coast '''

    # Proportion of each cell tower associated with each administrative region
    IntersectPop = pd.DataFrame(pd.read_csv('Data/Temporary/IvoryCoastIntersectPop.csv'))
    # Total activity of each cell tower
    Activity = pd.DataFrame(pd.read_csv('Data/IvoryCoast/CDR/metrics/total_activity.csv'))

    adm = [np.zeros((14, 6)), np.zeros((33, 6)), np.zeros((113, 6)), np.zeros((191, 6))]

    for i in range(4):
        for index, row in IntersectPop.iterrows():
            pop = np.sum(IntersectPop[IntersectPop['CellTowerID'] == row.CellTowerID]['Pop_2010'])
            if pop > 0:
                if i == 0:
                    adm[i][row.Adm_1_1-1, :] += (row.Pop_2010 / pop) * \
                                                  Activity[Activity['CellTowerID'] == row.CellTowerID][['Vol', 'Vol_in', 'Vol_out', 'Dur', 'Dur_in', 'Dur_out']].iloc[0]
                elif i == 1:
                    adm[i][row.Adm_2_2-1, :] += (row.Pop_2010 / pop) * \
                                                  Activity[Activity['CellTowerID'] == row.CellTowerID][['Vol', 'Vol_in', 'Vol_out', 'Dur', 'Dur_in', 'Dur_out']].iloc[0]
                elif i == 2:
                    adm[i][row.Adm_3_3-1, :] += (row.Pop_2010 / pop) * \
                                                  Activity[Activity['CellTowerID'] == row.CellTowerID][['Vol', 'Vol_in', 'Vol_out', 'Dur', 'Dur_in', 'Dur_out']].iloc[0]
                else:
                    adm[i][row.Adm_4_4-1, :] += (row.Pop_2010 / pop) * \
                                                  Activity[Activity['CellTowerID'] == row.CellTowerID][['Vol', 'Vol_in', 'Vol_out', 'Dur', 'Dur_in', 'Dur_out']].iloc[0]

    pop_1 = IntersectPop.groupby('Adm_1')['Pop_2010'].sum()
    activity_pp_1 = adm[0]/pop_1[:, None]

    pop_2 = IntersectPop.groupby('Adm_2')['Pop_2010'].sum()
    activity_pp_2 = adm[1] / pop_2[:, None]

    # 2 regions do not have population data -> remove
    pop_3 = IntersectPop.groupby('Adm_3')['Pop_2010'].sum()
    adm[2] = np.delete(adm[2], [22, 107], axis=0)
    activity_pp_3 = adm[2] / pop_3[:, None]

    # 13 regions do not have population data -> remove
    pop_4 = IntersectPop.groupby('Adm_4')['Pop_2010'].sum()
    adm[3] = np.delete(adm[3], [33, 35, 37, 39, 40, 94, 123, 124, 129, 133, 149, 168, 183], axis=0)
    activity_pp_4 = adm[3] / pop_4[:, None]


    # np.savetxt('Data/IvoryCoast/CDR/metrics/total_activity_adm_1.csv', activity_pp_1, delimiter=',')
    # np.savetxt('Data/IvoryCoast/CDR/metrics/total_activity_adm_2.csv', activity_pp_2, delimiter=',')
    # np.savetxt('Data/IvoryCoast/CDR/metrics/total_activity_adm_3.csv', activity_pp_3, delimiter=',')
    # np.savetxt('Data/IvoryCoast/CDR/metrics/total_activity_adm_4.csv', activity_pp_4, delimiter=',')


    ''' Senegal '''


