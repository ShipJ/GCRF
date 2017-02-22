from scipy.stats.stats import pearsonr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# total_activity = pd.DataFrame(pd.read_csv('Data/IvoryCoast/CDR/metrics/activityPerAdm_4.csv'))
#
# print total_activity
#
#
# vorProp = pd.DataFrame(pd.read_csv('Data/IvoryCoast/CDR/CellTowerInfo/VoronoiProportion.csv'))
# new = np.array(vorProp['vorPopClust'] / vorProp['vorPop'])
# vorProp['Prop'] = new
#
# proportional_activity = np.zeros(178)
#
# tower = 1
# count = 0
# for i in range(178):
#     activity = 0
#     while vorProp['clustID'].iloc[count] == tower:
#         activity += vorProp['Prop'].iloc[count] * total_activity['activity'].iloc[i]
#         count += 1
#
#     proportional_activity[i] = activity
#     tower = vorProp['clustID'].iloc[count]
#
# print len(total_activity)
# print len(proportional_activity)
#
# total_activity['proportional'] = proportional_activity
# print total_activity
#
#
#
# # poverty_rate = pd.DataFrame(pd.read_csv('Data/IvoryCoast/DHS/Extracted/FinalWealthIndex.csv'))
# #
# # print len(proportional_activity)
#
# # print pearsonr(proportional_activity, hiv['Mean'])
#






data = np.load('Data/IvoryCoast/CDR/metrics/total_activity.npy')
correspond = pd.DataFrame(pd.read_csv('Data/IvoryCoast/CDR/CellTowerInfo/CellTowerAdm_4.csv', encoding="utf-8-sig"),
                          columns=['CellTowerID', 'Adm_4ID'])

print correspond.sort('CellTowerID')
sys.exit()

index = sorted(correspond['CellTowerID'])
data = data[index]
print data

# data = data[index]


















#
# corresponding_subpref = pd.DataFrame(pd.read_csv("Data/IvoryCoast/CDR/CellTowers/CorrespondingSubpref.csv"), index=None)
# cell_tower_activity = pd.DataFrame({'Activity': cell_tower_activity, 'ID': range(1238)}, index=None)
# cell_tower_activity.drop([0, 573, 749, 1061, 1200, 1205, 1208, 1213])
# print corresponding_subpref
# print cell_tower_activity
#
# # data = pd.concat([cell_tower_activity, corresponding_subpref['TargetID']],
# #                  axis=1, ignore_index=True)
# #
# # print cell_tower_activity, corresponding_subpref