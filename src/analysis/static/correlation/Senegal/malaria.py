import pandas as pd
import numpy as np
import math
import os.path

if __name__ == "__main__":

    path = '/Users/JackShipway/Desktop/UCLProject/Data/Senegal/DHS'

    malaria = pd.DataFrame(pd.read_csv(path+'/Extracted/malaria.csv'))

    dhs = pd.DataFrame(pd.read_csv(path + '/SPSS/DHS_Adm_1234.csv'))

    print malaria

    malaria = malaria.applymap(lambda x: -1 if isinstance(x, basestring) and x.isspace() else int(x))

    print malaria

    bloodtest = malaria.iloc[:, 3:39]
    bloodtest['DHS'] = malaria['DHSClust']
    # bloodtest['NumMembers'] = malaria['NumMembers']
    #
    # rapidtest = malaria.iloc[:, 39:]
    # rapidtest['DHS'] = malaria['DHSClust']
    # rapidtest['NumMembers'] = malaria['NumMembers']
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # malaria = pd.DataFrame(pd.read_csv(path+'/malaria.csv'))
    # malaria = malaria.applymap(lambda x: 0 if isinstance(x, basestring) and x.isspace() else x)
    # malaria = malaria_dhs.as_matrix().astype(int)
    #
    # bloodtest = malaria.iloc[:, 3:39]
    # bloodtest['DHS'] = malaria['DHSClust']
    # bloodtest['NumMembers'] = malaria['NumMembers']
    #
    # rapidtest = malaria.iloc[:, 39:]
    # rapidtest['DHS'] = malaria['DHSClust']
    # rapidtest['NumMembers'] = malaria['NumMembers']
    #
    # pos = []
    # neg = []
    # mem = []
    # house = []
    #
    # for i in range(1, 353):
    #     data = rapidtest[rapidtest['DHS'] == i].iloc[:, 0:36]
    #     members = rapidtest[rapidtest['DHS'] == i].iloc[:, 37]
    #     freq = data.stack().value_counts().tolist()
    #     if len(freq) > 2:
    #         pos.append(freq[2] / float(np.sum(members)))
    #         neg.append(freq[1] / float(np.sum(members)))
    #     elif len(freq) == 2:
    #         pos.append(0)
    #         neg.append(freq[1] / float(np.sum(members)))
    #     else:
    #         pos.append(0)
    #         neg.append(0)
    #     mem.append(np.sum(members))
    #     house.append(len(data))
    #
    # prop = np.multiply(np.multiply(np.divide(pos, neg), np.divide(mem, 37265.0)), np.divide(house, 6871.0))
    # np.savetxt(path + '/Malaria/rapidtest.csv', prop, delimiter=',')