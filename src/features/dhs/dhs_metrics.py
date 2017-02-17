import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats.stats import pearsonr

if __name__ == "__main__":

    ####### Added from other file, to keep dhs metrics all in one place ##########
    # Get wealth index per cluster
    wealth_index = pd.DataFrame(pd.read_csv('Data/IvoryCoast/DHS/WealthIndex.csv', encoding='utf-8-sig'))
    cluster_wealth = wealth_index.groupby(by=['ClustNum'])['Score'].median()

    # Get total activity per cell tower
    cell_activity = pd.DataFrame(np.load('Data/IvoryCoast/CDR/Metrics/activity.npy'), columns=['Volume'])

    # Get regions corresponding to each cell tower
    corresponding_regions = pd.DataFrame(pd.read_csv('Data/CorrespondingSubPref.csv', usecols=['InputID', 'TargetID']))

    # Population data per cell tower
    pop_data = pd.DataFrame(pd.read_csv('Data/CorrespondingSubPref.csv'))

    activity = cell_activity['Volume']
    population = pop_data['Population']

    for i in range(1238):
        pop = population[i]
        if pop > 0:
            activity[i] /= pop

    corresponding_regions['ActivityPP'] = pd.Series(activity, index=corresponding_regions.index)

    cluster_activity = np.zeros(351)
    for i in range(351):
        region_i = np.array(corresponding_regions['InputID'][corresponding_regions['TargetID'] == i + 1])
        for j in range(len(region_i)):
            cluster_activity[i] += corresponding_regions['ActivityPP'][region_i[j]]

    # pop_data = pd.DataFrame(pd.read_csv("Data/voronoipop4.csv"))
    # print np.setdiff1d(range(1238), pop_data['ID'])

    # for subpref in sorted(pd.unique(pop_data['TargetID'])):
    #     cluster_activity[subpref] /= cluster_pop[subpref]
    #
    #
    a = (cluster_wealth + abs(min(cluster_wealth)))
    cluster_wealth = a / max(a)

    print pearsonr(cluster_activity, cluster_wealth)

    ######### ##############




    ''' HIV-based spatial metrics '''
    hiv_dhs = pd.DataFrame(pd.read_csv('Data/IvoryCoast/DHS/Extracted/hiv_per_dhs_clust.csv'))
    positive_pop = hiv_dhs.groupby(by='DHSClustID')['Result'].sum().reset_index()

    hiv_data = pd.DataFrame()
    hiv_data['DHSClust'] = np.setdiff1d(range(1, 353), [209, 225, 324])
    hiv_data['Num_Pos'] = np.array(positive_pop['Result'])

    clust_pop = pd.DataFrame(pd.read_csv('Data/IvoryCoast/DHS/Extracted/DHSClusterPop.csv'))

    for i in [149, 166, 174, 186, 206, 294, 315, 330, 342, 345]:
        hiv_data = hiv_data[hiv_data['DHSClust'] != i]

    for j in [209, 324]:
        clust_pop = clust_pop[clust_pop['DHSClust'] != j]

    hiv_data['Pop'] = np.array(clust_pop['Pop'])

    pos_pp = []
    for k in range(len(hiv_data)):
        pos_pp.append(hiv_data['Num_Pos'].iloc[k] / hiv_data['Pop'].iloc[k])

    hiv_data['Num_Pos_PP'] = np.array(pos_pp)

    DHSClust_adm = pd.DataFrame(pd.read_csv('Data/IvoryCoast/CDR/CellTowerInfo/ICSubpref_Adm_1234.csv'))

    missing_1 = np.setdiff1d(DHSClust_adm['DHSClust'], hiv_data['DHSClust'])
    missing_2 = np.setdiff1d(hiv_data['DHSClust'], DHSClust_adm['DHSClust'])

    for l in missing_2:
        hiv_data = hiv_data[hiv_data['DHSClust'] != l]

    for m in missing_1:
        DHSClust_adm = DHSClust_adm[DHSClust_adm['DHSClust'] != m]

    hiv_data['ID_1'] = np.array(DHSClust_adm['ID_1'])
    hiv_data['ID_2'] = np.array(DHSClust_adm['ID_2'])
    hiv_data['ID_3'] = np.array(DHSClust_adm['ID_3'])
    hiv_data['ID_4'] = np.array(DHSClust_adm['ID_4'])

    hiv_adm_1 = hiv_data.groupby(by='ID_1')['Num_Pos_PP'].mean().reset_index()
    hiv_adm_2 = hiv_data.groupby(by='ID_2')['Num_Pos_PP'].mean().reset_index()
    hiv_adm_3 = hiv_data.groupby(by='ID_3')['Num_Pos_PP'].mean().reset_index()
    hiv_adm_4 = hiv_data.groupby(by='ID_4')['Num_Pos_PP'].mean().reset_index()

    activity_adm_1234 = pd.DataFrame(pd.read_csv('Data/IvoryCoast/CDR/Metrics/activity_bts_level.csv'))

    vor_prop = pd.DataFrame(pd.read_csv('Data/IvoryCoast/RegionBoundaries/VorPropAdm1234.csv'))

    active = np.array(np.unique(vor_prop['cellID']))

    activity = []
    for i in active:
        data_active = vor_prop[vor_prop['cellID'] == i]

        act = 0
        for j in range(len(data_active)):
            act += ((data_active['vorPopClust'].iloc[j] / data_active['vorPop'].iloc[j]) * activity_adm_1234[activity_adm_1234['CellTowerID'] == i].iloc[0]['Activity'])
        activity.append(act)


    data = pd.DataFrame()
    data['CellID'] = active
    data['Activity'] = activity

    new = pd.DataFrame(pd.read_csv('Data/IvoryCoast/CDR/CellTowerInfo/temp.csv'))
    data['ID_1'] = np.array(new['ID_1'])
    data['ID_2'] = np.array(new['ID_2'])
    data['ID_3'] = np.array(new['ID_3'])
    data['ID_4'] = np.array(new['ID_4'])

    activity_adm_1 = data.groupby(by='ID_1')['Activity'].sum().reset_index()
    activity_adm_2 = data.groupby(by='ID_2')['Activity'].sum().reset_index()
    # activity_adm_3 = data.groupby(by='ID_3')['Activity'].sum().reset_index()
    activity_adm_4 = data.groupby(by='ID_4')['Activity'].sum().reset_index()


    need = np.setdiff1d(activity_adm_4['ID_4'], hiv_adm_4['ID_4'])

    for i in need:
        activity_adm_4 = pd.DataFrame(activity_adm_4[activity_adm_4['ID_4'] != i])
        hiv_adm_4 = pd.DataFrame(hiv_adm_4[hiv_adm_4['ID_4'] != i])

    need2 = np.setdiff1d(hiv_adm_4['ID_4'], activity_adm_4['ID_4'])

    for j in need2:
        activity_adm_4 = pd.DataFrame(activity_adm_4[activity_adm_4['ID_4'] != j])
        hiv_adm_4 = pd.DataFrame(hiv_adm_4[hiv_adm_4['ID_4'] != j])


    zeros = np.array(hiv_adm_4[hiv_adm_4['Num_Pos_PP'] == 0]['ID_4'])

    for i in zeros:
        hiv_adm_4 = hiv_adm_4[hiv_adm_4['ID_4'] != i]
        activity_adm_4 = activity_adm_4[activity_adm_4['ID_4'] != i]

    print hiv_adm_4
    print activity_adm_4



    print pearsonr(activity_adm_4['Activity'].iloc[1:], hiv_adm_4['Num_Pos_PP'].iloc[1:])

    plt.scatter(activity_adm_4['Activity'].iloc[1:], hiv_adm_4['Num_Pos_PP'].iloc[1:])
    plt.show()


    #
    # pop = pd.DataFrame(pd.read_csv('Data/Temporary/dhs_Pop.csv'))
    #
    #
    # for k in [32, 38, 61, 124, 130]:
    #     pop = pop[pop['ID_4'] != k]

    # pop = pop.groupby(by='ID_4')['pop_1km'].sum().reset_index()

    # activity_adm_4['Activity_PP'] = np.array(activity_adm_4['Activity']) / np.array(pop['pop_1km'])
    #
    # print pearsonr(hiv_adm_4['Num_Pos_PP'], activity_adm_4['Activity_PP'])
    #
    # plt.scatter(hiv_adm_4['Num_Pos_PP'], activity_adm_4['Activity_PP'])
    # plt.show()
    #
    # pop_adm = pd.DataFrame(pd.read_csv('Data/Temporary/pop_per_adm.csv')).dropna()
    #
    # pop_adm1 = pop_adm.groupby(by='ID_1')['pop_1km'].sum().reset_index()
    #
    # activity_adm_1['Activity_PP'] = np.array(activity_adm_1['Activity']) / np.array(pop_adm1['pop_1km'])
    #
    # print pearsonr(hiv_adm_1['Num_Pos_PP'], activity_adm_1['Activity_PP'])


    #
    # new_data = pd.DataFrame(pd.read_csv('Data/IvoryCoast/DHS/Extracted/FinalWealthIndex.csv'))
    #
    # need1 = np.setdiff1d(new_data['Adm_4ID'], activity_adm_4['ID_4'])
    # need2 = np.setdiff1d(activity_adm_4['ID_4'], new_data['Adm_4ID'])
    #
    # for k in need1:
    #     new_data = new_data[new_data['Adm_4ID'] != k]
    #
    # for l in need2:
    #     activity_adm_4 = activity_adm_4[activity_adm_4['ID_4'] != l]
    #
    # print new_data
    # print activity_adm_4
    #
    # print pearsonr(new_data['Poverty_Rate'], activity_adm_4['Activity'])



    # pearsonr(hiv_adm_3, activity_adm_3)
    # print pearsonr(hiv_adm_4['Num_Pos_PP'], activity_adm_4['Activity'])
    #
    #
    # plt.scatter(hiv_adm_4['Num_Pos_PP'].iloc[1:], activity_adm_4['Activity'].iloc[1:])
    # plt.show()




    sys.exit()







    # activity_adm_4 = adm.groupby(by='ID_4')['Activity'].sum().reset_index()
    #
    # for m in [4, 7, 9, 12, 23, 29, 30, 31, 37, 42, 48, 50, 53, 55, 56, 63, 66, 72, 76, 81, 82, 85, 100, 105, 113, 116, 117, 120, 122, 141, 143, 144, 147, 152, 153, 160, 165, 168, 171, 174, 175, 178]:
    #     activity_adm_4 = activity_adm_4[activity_adm_4['ID_4'] != m]
    #
    # for n in [38, 124, 130]:
    #     unnormalised = unnormalised[unnormalised['ID_4'] != n]
    #
    # print pearsonr(activity_adm_4['Activity'], unnormalised['Num_Pos_PP'])









    # ''' Child-Mortality-based spatial metrics '''
    # # Get child mortality data set
    # child_mortality_dhs = pd.DataFrame(pd.read_csv("Data/IvoryCoast/DHS/Extracted/ICChildHealth.csv", usecols=['clustID', 'houseID', 'numBirths', 'death'], encoding="utf-8-sig"))
    # sample_weight = pd.DataFrame(pd.read_csv("Data/IvoryCoast/DHS/Extracted/SampleWeight.csv", usecols=['WomenChildWeight', 'WomenChildClust', 'WomenChildHouse'], encoding="utf-8-sig")).dropna()
    #
    # # Different categories of ages at death - neo-natal is 0 to 28 days after birth etc, up to age 16
    # up_to_1_month, one_to_6_month, six_month_to_year, one_to_5_year, five_to_10_year, ten_to_16_year = [], [], [], [], [], []
    #
    # for i in range(len(child_mortality_dhs)):
    #     # Flags as a 3-digit number if death occurs, else no death
    #     death = child_mortality_dhs['death'][i].split()
    #     if death != []:
    #         unit = death[0]
    #         # neo_natal death measured in days corresponds to a 1 in this column
    #         if unit[0] == '1':
    #
    #             # Not one of these deaths
    #             one_to_6_month.append(0)
    #             six_month_to_year.append(0)
    #             one_to_5_year.append(0)
    #             five_to_10_year.append(0)
    #             ten_to_16_year.append(0)
    #
    #             # Neo-natal death recorded
    #             up_to_1_month.append(1)
    #
    #         elif unit[0] == '2':
    #             # Death between 1 month and 12 months
    #             up_to_1_month.append(0)
    #             one_to_5_year.append(0)
    #             five_to_10_year.append(0)
    #             ten_to_16_year.append(0)
    #
    #             if unit[1] == '0':
    #                 if unit[2] in ['0', '1', '2', '3', '4', '5']:
    #                     one_to_6_month.append(1)
    #                     six_month_to_year.append(0)
    #                 else:
    #                     one_to_6_month.append(0)
    #                     six_month_to_year.append(1)
    #             else:
    #                 one_to_6_month.append(0)
    #                 six_month_to_year.append(0)
    #
    #         else:
    #             up_to_1_month.append(0)
    #             one_to_6_month.append(0)
    #             six_month_to_year.append(0)
    #
    #             if unit[0] == '3':
    #                 if unit[1] == '0':
    #                     if unit[2] in ['0', '1', '2', '3', '4']:
    #                         one_to_5_year.append(1)
    #                         five_to_10_year.append(0)
    #                         ten_to_16_year.append(0)
    #                     else:
    #                         five_to_10_year.append(1)
    #                         one_to_5_year.append(0)
    #                         ten_to_16_year.append(0)
    #                 else:
    #                     ten_to_16_year.append(1)
    #                     five_to_10_year.append(0)
    #                     one_to_5_year.append(0)
    #             else:
    #                 five_to_10_year.append(0)
    #                 one_to_5_year.append(0)
    #                 ten_to_16_year.append(0)
    #     else:
    #         up_to_1_month.append(0)
    #         one_to_6_month.append(0)
    #         six_month_to_year.append(0)
    #         one_to_5_year.append(0)
    #         five_to_10_year.append(0)
    #         ten_to_16_year.append(0)
    #
    # child_mortality_dhs['NeoNatal'] = pd.Series(up_to_1_month)
    # child_mortality_dhs['One2Six'] = pd.Series(one_to_6_month)
    # child_mortality_dhs['Six2OneYear'] = pd.Series(six_month_to_year)
    # child_mortality_dhs['One25Year'] = pd.Series(one_to_5_year)
    #
    #
    # neo_natal = child_mortality_dhs.groupby(by='clustID')['NeoNatal'].sum()
    # one_to_6_month = child_mortality_dhs.groupby(by='clustID')['One2Six'].sum()
    # six_month_to_year = child_mortality_dhs.groupby(by='clustID')['Six2OneYear'].sum()
    # one_to_5_year = child_mortality_dhs.groupby(by='clustID')['One25Year'].sum()
    # births = child_mortality_dhs[['clustID', 'numBirths']].groupby(by='clustID')['numBirths'].sum()
    #
    # new = pd.DataFrame()
    # new['neoNatal'] = pd.Series(neo_natal)
    # new['one2Six'] = pd.Series(one_to_6_month)
    # new['six2One'] = pd.Series(six_month_to_year)
    # new['one2Five'] = pd.Series(one_to_5_year)
    # new['births'] = pd.Series(births)
    # new['weight'] = pd.Series(sample_weight.groupby(by='WomenChildClust')['WomenChildWeight'].mean())
    # new['neoNatalPB'] = (new['neoNatal'] / new['births']) * (new['weight'] / 1000000)
    # new['one2SixPB'] = (new['one2Six'] / new['births']) * (new['weight'] / 1000000)
    # new['six2OneYearPB'] = (new['six2One'] / new['births']) * (new['weight'] / 1000000)
    # new['oneTo5YearPB'] = (new['one2Five'] / new['births']) * (new['weight'] / 1000000)
    # new['clustID'] = pd.Series(range(353))
    #
    # hiv_dhs = pd.DataFrame(pd.read_csv("Data/IvoryCoast/DHS/dhs_aggregate_subpref.csv", usecols=['DHSCLUST', 'ID_4']))
    #
    # new = new.ix[hiv_dhs['DHSCLUST']]
    # new['ID_4'] = np.array(hiv_dhs['ID_4'])
    #
    # death_per_birth = new.groupby(by='ID_4')['neoNatalPB','one2SixPB','six2OneYearPB','oneTo5YearPB'].mean()
    # death_per_birth = pd.DataFrame(death_per_birth.reindex(range(192), fill_value=0))
    #
    # print death_per_birth
    #
    # death_per_birth.to_csv('Data/IvoryCoast/DHS/Extracted/deathPerBirth.csv')
    #
    #




    # compute the proportion of each respective death as compared to the total number of births per household, sum them up for all
    # households within a particular cluster, and then multiply by the sample weight. The idea overall therefore is to have 349 (number of clusters) and a value that is low or high
    # depending on how likely it is to have neo-natal deaths, and other age groups. Plot this data through QGIS and then see if there are any correlations amongst the data. Delmiro seems
    # to think that there will be a stronger signal for child mortality perhaps more so than HIV. Once this is complete, do the same for malaria, then maybe a few more of the features
    # Also don't forget to aggregate to the shapefile level because they're the only points we're able to plot. Again, this will have to be done per population of the cluster or whatever
    # is likely to make it fairer...






                # if unit[1] == '0':
                #     if unit[2] in ['0', '1', '2', '3', '4', '5']:
                #         one_to_6_month.append(1)
                #         six_month_to_year.append(0)
                #     else:
                #         one_to_6_month.append(0)
                #         six_month_to_year.append(1)
                # else:
                #     one_to_6_month.append(0)
                #     six_month_to_year.append(0)
                #
                # # Deaths between 6m and 1y
                # if unit[1] == '1':
                #     if unit[2] in ['0', '1', '2']:
                #         six_month_to_year.append(1)
                #     else:
                #         six_month_to_year.append(0)
                # else:
                #     six_month_to_year.append(0)




    #
    #         else:
    #             up_to_1_month.append(0)
    #             one_to_6_month.append(0)
    #             six_month_to_year.append(0)
    #
    #         # Deaths between 1m and 12m
    #         if unit[0] == '2':
    #
    #         else:
    #             one_to_6_month.append(0)
    #             six_month_to_year.append(0)
    #     else:
    #         up_to_1_month.append(0)
    #         one_to_6_month.append(0)
    #         six_month_to_year.append(0)
    #
    # print sum(up_to_1_month), sum(one_to_6_month), sum(six_month_to_year)

            # # Deaths between 1m and 12m
            # if unit[0] == '2':
            #     # Deaths between 1 month and 6 months
            #     if unit[1] == '0':
            #         if unit[2] in ['0', '1', '2', '3', '4', '5']:
            #             one_to_6_month.append(1)
            #         else:
            #             six_month_to_year.append(1)
            #     # Deaths between 6m and 1y
            #     if unit[1] == '1':
            #         if unit[2] in ['0', '1', '2']:
            #             six_month_to_year.append(1)
            # else:
            #     one_to_6_month.append(0)
            #     six_month_to_year.append(0)
            #
            # if unit[0] == '3':
            #     if unit[1] == '0':
            #         if unit[2] in ['0', '1', '2', '3', '4']:
            #             one_to_5_year.append(1)
            #         else:
            #             five_to_10_year.append(1)
            #     if unit[1] == '1':
            #         ten_to_16_year.append(1)
            # else:
            #     one_to_5_year.append(0)
            #     five_to_10_year.append(0)
            #     ten_to_16_year.append(0)





        #
        #
        # else:
        #     up_to_1_month.append(0)
        #
        #

























