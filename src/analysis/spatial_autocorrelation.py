from src.config import config
import pandas as pd
import networkx as nx
import numpy as np

def moran(data, *args):

    dist = pd.DataFrame(pd.read_csv(PATH+'/processed/%s/distance_area/dist_matrix_adm%s.csv' % (country,
                                                                                                list(args[0])[-1])))
    response = data.groupby(args[0])[args[1]].mean().reset_index().dropna()

    non_missing = np.array(response[args[0]])
    dist = dist[dist['Source'].isin(non_missing)]
    dist = dist[dist['Target'].isin(non_missing)]

    G = nx.from_pandas_dataframe(dist, 'Source', 'Target', 'Distance_km')
    H = nx.adjacency_matrix(G, weight='Distance_km').todense()

    H[H>500] = 0

    w = np.reciprocal(H+1)
    response = np.array(response[args[1]])

    x_bar = np.mean(response)
    adms = len(w)
    c = np.zeros((adms, adms))
    for i in range(adms):
        for j in range(adms):
            c[i, j] = (response[i]-x_bar)*(response[j]-x_bar)

    mysum = np.sum(np.multiply(c, w))
    s2 = np.sum(np.power(response-x_bar, 2))/adms
    sumw = np.sum(w)
    I = mysum/(s2*sumw)
    print I


if __name__ == '__main__':
    PATH = config.get_dir()
    country = config.get_country()
    data = pd.DataFrame(pd.read_csv(PATH+'/final/%s/master_2.0.csv' % country))

    print moran(data, 'Adm_4', 'DeathRate')

