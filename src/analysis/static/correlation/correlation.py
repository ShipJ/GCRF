import pandas as pd
import numpy as np
import sys
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # DHS data
    dhs = pd.DataFrame(pd.read_csv('DHSData.csv'))

    # CDR data (for each adm)
    activity_adm_1 = pd.DataFrame(pd.read_csv('activity_1.csv'))
    activity_adm_2 = pd.DataFrame(pd.read_csv('activity_2.csv'))
    activity_adm_3 = pd.DataFrame(pd.read_csv('activity_3.csv'))
    activity_adm_4 = pd.DataFrame(pd.read_csv('activity_4.csv'))

    # Normalise data (purely for visualisation)
    activity_adm_1['normed'] = activity_adm_1['Vol'] / max(activity_adm_1['Vol'])
    activity_adm_1['dur_normed'] = activity_adm_1['Dur'] / max(activity_adm_1['Dur'])
    activity_adm_2['normed'] = activity_adm_2['Vol'] / max(activity_adm_2['Vol'])
    activity_adm_2['dur_normed'] = activity_adm_2['Dur'] / max(activity_adm_2['Dur'])
    activity_adm_3['normed'] = activity_adm_3['Vol'] / max(activity_adm_3['Vol'])
    activity_adm_3['dur_normed'] = activity_adm_3['Dur'] / max(activity_adm_3['Dur'])
    activity_adm_4['normed'] = activity_adm_4['Vol'] / max(activity_adm_4['Vol'])
    activity_adm_4['dur_normed'] = activity_adm_4['Dur'] / max(activity_adm_4['Dur'])

    ''' HIV '''
    hiv_adm_1 = dhs.groupby('Adm_1')['HIV_rate'].mean().reset_index()
    hiv_adm_1['HIV_rate'] = hiv_adm_1['HIV_rate'] / max(hiv_adm_1['HIV_rate'])
    hiv_adm_2 = dhs.groupby('Adm_2')['HIV_rate'].mean().reset_index()
    hiv_adm_3 = dhs.groupby('Adm_3')['HIV_rate'].mean().reset_index()
    hiv_adm_4 = dhs.groupby('Adm_4')['HIV_rate'].mean().reset_index()

    ''' Malaria '''
    malaria_adm_1 = dhs.groupby('Adm_1')['MalariaPerPop'].mean().reset_index()
    malaria_adm_2 = dhs.groupby('Adm_2')['MalariaPerPop'].mean().reset_index()
    malaria_adm_3 = dhs.groupby('Adm_3')['MalariaPerPop'].mean().reset_index()
    malaria_adm_4 = dhs.groupby('Adm_4')['MalariaPerPop'].mean().reset_index()

    ''' Child Mortality '''
    child_mort_adm_1 = dhs.groupby('Adm_1')['neoNatalPB'].mean().reset_index()
    child_mort_adm_2 = dhs.groupby('Adm_2')['neoNatalPB'].mean().reset_index()
    child_mort_adm_3 = dhs.groupby('Adm_3')['neoNatalPB'].mean().reset_index()
    child_mort_adm_4 = dhs.groupby('Adm_4')['neoNatalPB'].mean().reset_index()

    ''' Tuberculosis '''
    #
    ''' Morning Sickness '''
    #

    ''' align adm_3 '''
    for i in np.setdiff1d(activity_adm_3['Adm_3'], dhs.groupby('Adm_3')['HIV_rate'].mean().reset_index()['Adm_3']):
        activity_adm_3 = activity_adm_3[activity_adm_3['Adm_3'] != i]

    ''' align adm_4'''
    for i in np.setdiff1d(activity_adm_4['Adm_4'], dhs.groupby('Adm_4')['HIV_rate'].mean().reset_index()['Adm_4']):
        activity_adm_4 = activity_adm_4[activity_adm_4['Adm_4'] != i]

    for i in np.setdiff1d(dhs.groupby('Adm_4')['HIV_rate'].mean().reset_index()['Adm_4'], activity_adm_4['Adm_4']):
        hiv_adm_4 = hiv_adm_4[hiv_adm_4['Adm_4'] != i]
        malaria_adm_4 = malaria_adm_4[malaria_adm_4['Adm_4'] != i]
        child_mort_adm_4 = child_mort_adm_4[child_mort_adm_4['Adm_4'] != i]


    ''' Correlation testing using PMCC '''

    # list = [0,2,3,4,5,6,7,8,9,10,11,12,13]
    #
    # # Adm_1: hiv
    # print pearsonr(np.log(activity_adm_1['normed'].ix[list]), hiv_adm_1['HIV_rate'].ix[list])
    # plt.scatter(np.log(activity_adm_1['normed'].ix[list]), hiv_adm_1['HIV_rate'].ix[list])
    # plt.show()
    #
    # print pearsonr(np.log(activity_adm_1['normed']), hiv_adm_1['HIV_rate'])
    # plt.scatter(np.log(activity_adm_1['normed']), hiv_adm_1['HIV_rate'])
    # plt.show()
    #
    # sys.exit()
    # print pearsonr(activity_adm_1['dur_normed'].ix[list], hiv_adm_1['HIV_rate'].ix[list])
    # print pearsonr(activity_adm_1['Vol_in'].ix[list], hiv_adm_1['HIV_rate'].ix[list])
    # print pearsonr(activity_adm_1['Vol_out'].ix[list], hiv_adm_1['HIV_rate'].ix[list])
    # print pearsonr(activity_adm_1['Dur_in'].ix[list], hiv_adm_1['HIV_rate'].ix[list])
    # print pearsonr(activity_adm_1['Dur_out'].ix[list], hiv_adm_1['HIV_rate'].ix[list])
    #
    #
    # # Adm_1: malaria
    # print pearsonr(activity_adm_1['normed'].ix[list], malaria_adm_1['MalariaPerPop'].ix[list])
    # print pearsonr(activity_adm_1['dur_normed'].ix[list], malaria_adm_1['MalariaPerPop'].ix[list])
    # print pearsonr(activity_adm_1['Vol_in'].ix[list], malaria_adm_1['MalariaPerPop'].ix[list])
    # print pearsonr(activity_adm_1['Vol_out'].ix[list], malaria_adm_1['MalariaPerPop'].ix[list])
    # print pearsonr(activity_adm_1['Dur_in'].ix[list], malaria_adm_1['MalariaPerPop'].ix[list])
    # print pearsonr(activity_adm_1['Dur_out'].ix[list], malaria_adm_1['MalariaPerPop'].ix[list])
    #
    # plt.scatter(activity_adm_1['normed'], malaria_adm_1['MalariaPerPop'])
    # plt.show()

    # # # Adm_1: Child mortality
    # # print pearsonr(activity_adm_1['normed'], child_mort_adm_1['neoNatalPB'])
    # # print pearsonr(activity_adm_1['dur_normed'], child_mort_adm_1['neoNatalPB'])
    # # print pearsonr(activity_adm_1['Vol_in'], child_mort_adm_1['neoNatalPB'])
    # # print pearsonr(activity_adm_1['Vol_out'], child_mort_adm_1['neoNatalPB'])
    # # print pearsonr(activity_adm_1['Dur_in'], child_mort_adm_1['neoNatalPB'])
    # # print pearsonr(activity_adm_1['Dur_out'], child_mort_adm_1['neoNatalPB'])
    #
    # # Adm_2: hiv

    # list = np.array(np.where(activity_adm_2['Vol'] < 85))
    # list2 = np.array(np.where(activity_adm_2['Vol'] > 10))
    # list3 = np.intersect1d(list, list2)
    # list4 = np.array(np.where(malaria_adm_2['MalariaPerPop'] > 0.001))
    # list5 = np.intersect1d(list3, list4)
    #
    # print pearsonr(np.array(activity_adm_2['Vol'])[list5], np.array(malaria_adm_2['MalariaPerPop'])[list5])
    # plt.scatter(np.array(activity_adm_2['Vol'])[list5], np.array(malaria_adm_2['MalariaPerPop'])[list5])
    # plt.show()
    #
    # sys.exit()
    # print pearsonr(activity_adm_2['dur_normed'].ix[list], hiv_adm_2['HIV_rate'].ix[list])
    # print pearsonr(activity_adm_2['Vol_in'].ix[list], hiv_adm_2['HIV_rate'].ix[list])
    # print pearsonr(activity_adm_2['Vol_out'].ix[list], hiv_adm_2['HIV_rate'].ix[list])
    # print pearsonr(activity_adm_2['Dur_in'].ix[list], hiv_adm_2['HIV_rate'].ix[list])
    # print pearsonr(activity_adm_2['Dur_out'].ix[list], hiv_adm_2['HIV_rate'].ix[list])


    # plt.scatter(activity_adm_2['normed'][4:], hiv_adm_2['HIV_rate'][4:])
    # plt.show()
    #
    # print 'polyfit:', np.polyfit(activity_adm_2['normed'].ix[4:], hiv_adm_2['HIV_rate'].ix[4:], 2)
    # print 'polyfit:', np.polyfit(activity_adm_2['normed'].ix[4:], hiv_adm_2['HIV_rate'].ix[4:], 1)
    #
    # x = np.linspace(0, 0.5, 50)
    # y = 0.23511516 * (x ** 2) + (-0.1438877 * x) + 0.02231236
    #
    # a = np.linspace(0, 0.5, 50)
    # b = -0.0539* a + 0.01589
    #
    # plt.plot(x, y)
    # plt.plot(a, b)
    #
    # plt.scatter(activity_adm_2['normed'].ix[4:], hiv_adm_2['HIV_rate'].ix[4:])
    # plt.plot()
    # plt.show()
    #
    # sys.exit()










    # # Adm_2: malaria
    # print pearsonr(activity_adm_2['normed'].ix[list], malaria_adm_2['MalariaPerPop'].ix[list])
    # print pearsonr(activity_adm_2['dur_normed'].ix[list], malaria_adm_2['MalariaPerPop'].ix[list])
    # print pearsonr(activity_adm_2['Vol_in'].ix[list], malaria_adm_2['MalariaPerPop'].ix[list])
    # print pearsonr(activity_adm_2['Vol_out'].ix[list], malaria_adm_2['MalariaPerPop'].ix[list])
    # print pearsonr(activity_adm_2['Dur_in'].ix[list], malaria_adm_2['MalariaPerPop'].ix[list])
    # print pearsonr(activity_adm_2['Dur_out'].ix[list], malaria_adm_2['MalariaPerPop'].ix[list])





    # dat = dhs.groupby('Adm_3')['Z_Med'].mean().reset_index()
    #
    # list = list(dat[dat['Z_Med'] < 0]['Adm_3'])
    #
    # a = activity_adm_3.ix[list]
    # b = hiv_adm_3.ix[list]
    #
    #
    # c = activity_adm_3.ix[np.setdiff1d(range(1,34),list)]
    # d = hiv_adm_3.ix[np.setdiff1d(range(1,34),list)]
    #
    # plt.scatter(a['Vol'],b['HIV_rate'],c='b')
    # plt.scatter(c['Vol'],d['HIV_rate'],c='r')
    # plt.show()




    # a = activity_adm_1.ix[[0,3,6,9,10,12]]
    # b = hiv_adm_1.ix[[0,3,6,9,10,12]]
    #
    # c = activity_adm_1.ix[[1,2,4,5,7,8,11,13]]
    # d = hiv_adm_1.ix[[1,2,4,5,7,8,11,13]]
    #
    # plt.scatter(a['Vol'],b['HIV_rate'],c='b')
    # plt.scatter(c['Vol'],d['HIV_rate'],c='r')
    # plt.show()



    # Adm_3: hiv
    # print pearsonr(activity_adm_3['normed'], hiv_adm_3['HIV_rate'])
    # print pearsonr(activity_adm_3['dur_normed'], hiv_adm_3['HIV_rate'])

    # Adm_3: malaria
    # list = np.where(activity_adm_3['Vol'] > 10)
    # list2 = np.where(activity_adm_3['Vol'] < 85)
    # list3 = np.intersect1d(list, list2)
    #
    # list4 = np.where(malaria_adm_3['MalariaPerPop'] > 0.01)
    # list5 = np.intersect1d(list3, list4)
    # #
    # print pearsonr(np.log(np.array(activity_adm_3['Vol'])[list5]), malaria_adm_3['MalariaPerPop'].ix[list5])
    # # print pearsonr(activity_adm_3['dur_normed'], malaria_adm_3['MalariaPerPop'])
    # plt.scatter(np.log(activity_adm_3['Vol'].ix[list5]), malaria_adm_3['MalariaPerPop'].ix[list5])
    # plt.show()

    # # Adm_4: hiv
    # print pearsonr(activity_adm_4['normed'], hiv_adm_4['HIV_rate'])
    # print pearsonr(activity_adm_4['dur_normed'], hiv_adm_4['HIV_rate'])
    # # Adm_4: malaria
    # print pearsonr(activity_adm_4['normed'], malaria_adm_4['MalariaPerPop'])
    # print pearsonr(activity_adm_4['dur_normed'], malaria_adm_4['MalariaPerPop'])
    #
    #
    # a = np.median(activity_adm_4['Vol_in'])
    # b = list(activity_adm_4[activity_adm_4['Vol_in'] < a]['Adm_4'])
    # c = np.setdiff1d(activity_adm_4['Adm_4'], b)
    #

    list1 = np.where(activity_adm_4['Vol'] > 0)
    list2 = np.where(activity_adm_4['Vol'] < 100)
    list3 = np.intersect1d(list1, list2)
    list4 = np.where(malaria_adm_4['MalariaPerPop'] > 0.01)
    list5 = np.where(malaria_adm_4['MalariaPerPop'] < 0.2)
    list6 = np.intersect1d(list4, list5)

    list7 = np.intersect1d(list3, list6)

    print pearsonr(np.log(np.array(activity_adm_4['Vol'])[list7]), np.array(malaria_adm_4['MalariaPerPop'])[list7])
    plt.scatter(np.log(np.array(activity_adm_4['Vol'])[list7]), np.array(malaria_adm_4['MalariaPerPop'])[list7])
    plt.show()




