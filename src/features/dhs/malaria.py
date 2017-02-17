''' This file shows how to take engineered DHS data, and extract various metrics such as number of malaria cases '''
# Inputs:
# Outputs:

import pandas as pd
import numpy as np

if __name__ == "__main__":

    # Data set selected, trimmed down, and weighted from SPSS
    malaria_dhs = pd.DataFrame(pd.read_csv("Data/IvoryCoast/DHS/Extracted/malaria_weighted.csv"))
    # Convert strings to numeric values for all data
    malaria_dhs = malaria_dhs.applymap(lambda x: 0 if isinstance(x, basestring) and x.isspace() else x)
    data = malaria_dhs.as_matrix().astype(int)

    # values of 6,7,9 would indicate problems with the sampling
    data = np.delete(data, np.where(data[:, 4:40] == 6)[0], axis=0)
    data = np.delete(data, np.where(data[:, 4:40] == 7)[0], axis=0)
    data = np.delete(data, np.where(data[:, 4:40] == 9)[0], axis=0)
    total = np.sum(data[:, 4:40], axis=1)

    malaria_cases = pd.DataFrame(data[:, [0,1,3]], columns=['ClustID', 'HouseID', 'Members'])
    malaria_cases['Cases'] = pd.Series(total)

    cluster_cases = pd.DataFrame(malaria_cases.groupby('ClustID')['Members', 'Cases'].sum())

    cluster_cases['CasePerPerson'] = cluster_cases['Cases'] / cluster_cases['Members']

    cluster_adm4 = pd.DataFrame(pd.read_csv("Data/IvoryCoast/DHS/dhs_aggregate_subpref.csv", usecols=['DHSCLUST', 'ID_4']))

    cluster_cases = cluster_cases.ix[cluster_adm4['DHSCLUST']]

    cluster_cases['ADM_4'] = np.array(cluster_adm4['ID_4'])


    cluster_cases = cluster_cases.groupby('ADM_4')['CasePerPerson'].mean()
    cluster_cases = pd.DataFrame(cluster_cases.reindex(range(192), fill_value=0))

    cluster_cases.to_csv('Data/IvoryCoast/DHS/Extracted/malaria_cases.csv')



