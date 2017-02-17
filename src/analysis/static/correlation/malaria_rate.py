import pandas as pd
import numpy as np
import sys
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # DHS data
    malaria = pd.DataFrame(pd.read_csv('malaria_dhs.csv'))
    # Transform string values to ints
    malaria_dhs = malaria.applymap(lambda x: 0 if isinstance(x, basestring) and x.isspace() else x)
    # Convert to a numpy array
    data = malaria_dhs.as_matrix().astype(int)

    # Remove inconclusive, other and other data entries (only interested in positive or negative results)
    data = np.delete(data, np.where(data[:, 4:40] == 6)[0], axis=0)
    data = np.delete(data, np.where(data[:, 4:40] == 7)[0], axis=0)
    data = np.delete(data, np.where(data[:, 4:40] == 9)[0], axis=0)

    total = np.sum(data[:, 4:40], axis=1)

    malaria_cases = pd.DataFrame(data[:, [0, 1, 3]], columns=['ClustID', 'HouseID', 'Members'])
    malaria_cases['Cases'] = pd.Series(total)

    # Proportional number of cases per cluster (aligned with UN estimated sample weight)
    cluster_cases = pd.DataFrame(malaria_cases.groupby('ClustID')['Members', 'Cases'].sum()).reset_index()

    # Remove DHS clusters for which we have no geographical knowledge
    for i in [149, 166, 174, 186, 206, 294, 315, 330, 342, 345]:
        cluster_cases = cluster_cases[cluster_cases['ClustID'] != i]

    # Cases per household members
    cluster_cases['CasePerPerson'] = cluster_cases['Cases'] / cluster_cases['Members']

    cluster_cases.to_csv('malaria_cases_pp.csv', index=None)