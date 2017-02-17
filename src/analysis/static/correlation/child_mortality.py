import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
from scipy.stats.stats import pearsonr

if __name__ == "__main__":

    # Get child mortality data set
    child_mortality_dhs = pd.DataFrame(pd.read_csv("ICChildHealth.csv", usecols=['clustID', 'houseID', 'numBirths', 'death'], encoding="utf-8-sig"))

    # Different categories of ages at death - neo-natal is 0 to 28 days after birth etc, up to age 16
    up_to_1_month, one_to_6_month, six_month_to_year, one_to_5_year, five_to_10_year, ten_to_16_year = [], [], [], [], [], []

    for i in range(len(child_mortality_dhs)):
        # Flags as a 3-digit number if death occurs, else no death
        death = child_mortality_dhs['death'][i].split()
        if death != []:
            unit = death[0]
            # neo_natal death measured in days corresponds to a 1 in this column
            if unit[0] == '1':

                # Not one of these deaths
                one_to_6_month.append(0)
                six_month_to_year.append(0)
                one_to_5_year.append(0)
                five_to_10_year.append(0)
                ten_to_16_year.append(0)

                # Neo-natal death recorded
                up_to_1_month.append(1)

            elif unit[0] == '2':
                # Death between 1 month and 12 months
                up_to_1_month.append(0)
                one_to_5_year.append(0)
                five_to_10_year.append(0)
                ten_to_16_year.append(0)

                if unit[1] == '0':
                    if unit[2] in ['0', '1', '2', '3', '4', '5']:
                        one_to_6_month.append(1)
                        six_month_to_year.append(0)
                    else:
                        one_to_6_month.append(0)
                        six_month_to_year.append(1)
                else:
                    one_to_6_month.append(0)
                    six_month_to_year.append(0)

            else:
                up_to_1_month.append(0)
                one_to_6_month.append(0)
                six_month_to_year.append(0)

                if unit[0] == '3':
                    if unit[1] == '0':
                        if unit[2] in ['0', '1', '2', '3', '4']:
                            one_to_5_year.append(1)
                            five_to_10_year.append(0)
                            ten_to_16_year.append(0)
                        else:
                            five_to_10_year.append(1)
                            one_to_5_year.append(0)
                            ten_to_16_year.append(0)
                    else:
                        ten_to_16_year.append(1)
                        five_to_10_year.append(0)
                        one_to_5_year.append(0)
                else:
                    five_to_10_year.append(0)
                    one_to_5_year.append(0)
                    ten_to_16_year.append(0)
        else:
            up_to_1_month.append(0)
            one_to_6_month.append(0)
            six_month_to_year.append(0)
            one_to_5_year.append(0)
            five_to_10_year.append(0)
            ten_to_16_year.append(0)

    child_mortality_dhs['NeoNatal'] = pd.Series(up_to_1_month)
    child_mortality_dhs['One2Six'] = pd.Series(one_to_6_month)
    child_mortality_dhs['Six2OneYear'] = pd.Series(six_month_to_year)
    child_mortality_dhs['One25Year'] = pd.Series(one_to_5_year)


    neo_natal = child_mortality_dhs.groupby(by='clustID')['NeoNatal'].sum()
    one_to_6_month = child_mortality_dhs.groupby(by='clustID')['One2Six'].sum()
    six_month_to_year = child_mortality_dhs.groupby(by='clustID')['Six2OneYear'].sum()
    one_to_5_year = child_mortality_dhs.groupby(by='clustID')['One25Year'].sum()
    births = child_mortality_dhs[['clustID', 'numBirths']].groupby(by='clustID')['numBirths'].sum()

    new = pd.DataFrame()
    new['clustID'] = np.array(range(1, 352))
    new['neoNatal'] = np.array(neo_natal)
    new['one2Six'] = np.array(one_to_6_month)
    new['six2One'] = np.array(six_month_to_year)
    new['one2Five'] = np.array(one_to_5_year)
    new['births'] = np.array(births)

    for i in [149, 166, 174, 186, 206, 294, 315, 330, 342, 345]:
        new = new[new['clustID'] != i]

    new.to_csv('child_mortality.csv', index=None)






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
