import pandas as pd
import numpy as np
import sys
import os

from sklearn.linear_model import LinearRegression

from scipy.stats import pearsonr

path = '/Users/JackShipway/Desktop/UCLProject/Data/IvoryCoast'
path2 = '/Project1-Health'

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)


list_files('/Users/JackShipway/Desktop/UCLProject')

sys.exit()

data = pd.DataFrame(pd.read_csv(path+'/Master.csv')).dropna()
#
# pop_1 = np.array(data.groupby('Adm_1')['Pop_2010'].sum())
# pop_2 = np.array(data.groupby('Adm_2')['Pop_2010'].sum())
# pop_3 = np.array(data.groupby('Adm_3')['Pop_2010'].sum())
# pop_4 = np.array(data.groupby('Adm_4')['Pop_2010'].sum())
#
# area_km2_1 = np.array(data.groupby('Adm_1')['Area(km^2)'].sum())
# area_km2_2 = np.array(data.groupby('Adm_2')['Area(km^2)'].sum())
# area_km2_3 = np.array(data.groupby('Adm_3')['Area(km^2)'].sum())
# area_km2_4 = np.array(data.groupby('Adm_4')['Area(km^2)'].sum())
#
# pop_dense_1 = np.array(data.groupby('Adm_1')['Pop_density_2010'].mean())
# pop_dense_2 = np.array(data.groupby('Adm_2')['Pop_density_2010'].mean())
# pop_dense_3 = np.array(data.groupby('Adm_3')['Pop_density_2010'].mean())
# pop_dense_4 = np.array(data.groupby('Adm_4')['Pop_density_2010'].mean())
#
# act_1 = np.array(data.groupby('Adm_1')['Activity'].mean())
# act_2 = np.array(data.groupby('Adm_2')['Activity'].mean())
# act_3 = np.array(data.groupby('Adm_3')['Activity'].mean())
# act_4 = np.array(data.groupby('Adm_4')['Activity'].mean())
#
# ent_1 = np.array(data.groupby('Adm_1')['Entropy'].mean())
# ent_2 = np.array(data.groupby('Adm_2')['Entropy'].mean())
# ent_3 = np.array(data.groupby('Adm_3')['Entropy'].mean())
# ent_4 = np.array(data.groupby('Adm_4')['Entropy'].mean())
#
# int_1 = np.array(data.groupby('Adm_1')['Introversion'].mean())
# int_2 = np.array(data.groupby('Adm_2')['Introversion'].mean())
# int_3 = np.array(data.groupby('Adm_3')['Introversion'].mean())
# int_4 = np.array(data.groupby('Adm_4')['Introversion'].mean())
#
# med_1 = np.array(data.groupby('Adm_1')['Med_degree'].mean())
# med_2 = np.array(data.groupby('Adm_2')['Med_degree'].mean())
# med_3 = np.array(data.groupby('Adm_3')['Med_degree'].mean())
# med_4 = np.array(data.groupby('Adm_4')['Med_degree'].mean())
#
mal_1 = data.groupby('Adm_1')[['Blood_pos', 'Blood_tot']].sum().reset_index()
mal_2 = data.groupby('Adm_2')[['Blood_pos', 'Blood_tot']].sum().reset_index()
mal_3 = data.groupby('Adm_3')[['Blood_pos', 'Blood_tot']].sum().reset_index()
mal_4 = data.groupby('Adm_4')[['Blood_pos', 'Blood_tot']].sum().reset_index()
mal_1 = np.divide(mal_1['Blood_pos'], mal_1['Blood_tot'])
mal_2 = np.divide(mal_2['Blood_pos'], mal_2['Blood_tot'])
mal_3 = np.divide(mal_3['Blood_pos'], mal_3['Blood_tot'])
mal_4 = np.array(np.divide(mal_4['Blood_pos'], mal_4['Blood_tot']))


class Something:

    def __init__(self):
        self.something = 'Okay, this is something!'

    def somethingelse(self):
        a = 'Okay, this is something else!'
        return a


myThing = Something()
print myThing.something
print myThing.somethingelse()

sys.exit()

#
# wealth_1 = np.array(data.groupby('Adm_1')['Wealth'].median())
# wealth_2 = np.array(data.groupby('Adm_2')['Wealth'].median())
# wealth_3 = np.array(data.groupby('Adm_3')['Wealth'].median())
# wealth_4 = np.array(data.groupby('Adm_4')['Wealth'].median())

# urb = np.where(data['UrbRur'] > 0.5)
# rur = np.where(data['UrbRur'] < 0.5)
#
# a = pop_4[urb][1:]
# b = wealth_4[urb][1:]
#
# c = pop_4[rur]
# d = wealth_4[rur]
#
# print pearsonr(a, b)
# print pearsonr(c, d)
#
# plt.scatter(a, b)
# # plt.show()
# plt.scatter(c, d, c='r')
# plt.show()

data = np.genfromtxt('/Users/JackShipway/Desktop/new.csv', delimiter=',', usecols=(0,1,2,3,4))
# data = data[~np.isnan(data).any(axis=1)][1:]



mae = []
for i in range(5):
    mae_1000 = []
    train_prop = 16 + (i * 3)
    for j in range(100):
        np.random.shuffle(data_normalised)
        X_train = data_normalised[0:train_prop, 0:3]
        y_train = data_normalised[0:train_prop, 4]
        X_test = data_normalised[train_prop:, 0:3]
        y_test = data_normalised[train_prop:, 4]
        lm = LinearRegression()
        lm.fit(X_train, y_train)
        mae_1000.append(np.mean(np.abs(lm.predict(X_test) - y_test)))
    mae.append(np.median(mae_1000))
plt.plot(range(5), mae, '-x', c='b', label='Population+CDR')

# ''' '''
# data = np.genfromtxt(path+'/Master.csv', delimiter=',', skiprows=1,
#                      usecols=(5, 9, 28))
# data = data[~np.isnan(data).any(axis=1)][1:]
#
# data_normalised = np.zeros(data.shape)
# for i in range(data.shape[1]):  # Normalise data by subtracting feature means from each value
#     feature = data[:, i]
#     mean = np.mean(feature)  # Compute feature mean
#     std = np.std(feature)  # Compute feature standard deviation
#     feature_z = (feature - mean) / std  # Compute z-score based on computed mean and std
#     data_normalised[:, i] = feature_z
#
# mae = []
# for i in range(5):
#     mae_1000 = []
#     train_prop = 70 + (i * 13)
#     for l in range(1000):
#         np.random.shuffle(data_normalised)
#         X_train = data_normalised[0:train_prop, 0:1]
#         y_train = data_normalised[0:train_prop, 2]
#         X_test = data_normalised[train_prop:, 0:1]
#         y_test = data_normalised[train_prop:, 2]
#         lm = LinearRegression()
#         lm.fit(X_train, y_train)
#         mae_1000.append(np.mean(np.abs(lm.predict(X_test) - y_test)))
#     mae.append(np.median(mae_1000))
# plt.plot(range(5), mae, '-x', c='g', label='Population + Area')
#
# ''' '''
#
# ''' '''
# data = np.genfromtxt(path+'/Master.csv', delimiter=',', skiprows=1, #16, 21, 22, 23,
#                      usecols=(15, 21, 22, 23, 34, 35, 36, 37, 28))
# data = data[~np.isnan(data).any(axis=1)][1:]
#
# data_normalised = np.zeros(data.shape)
# for i in range(data.shape[1]):  # Normalise data by subtracting feature means from each value
#     feature = data[:, i]
#     mean = np.mean(feature)  # Compute feature mean
#     std = np.std(feature)  # Compute feature standard deviation
#     feature_z = (feature - mean) / std  # Compute z-score based on computed mean and std
#     data_normalised[:, i] = feature_z
#
# mae = []
# for i in range(5):
#     mae_1000 = []
#     train_prop = 70 + (i * 13)
#     for l in range(1000):
#         np.random.shuffle(data_normalised)
#         X_train = data_normalised[0:train_prop, 0:7]
#         y_train = data_normalised[0:train_prop, 8]
#         X_test = data_normalised[train_prop:, 0:7]
#         y_test = data_normalised[train_prop:, 8]
#         lm = LinearRegression()
#         lm.fit(X_train, y_train)
#         mae_1000.append(np.mean(np.abs(lm.predict(X_test) - y_test)))
#     mae.append(np.median(mae_1000))
# plt.plot(range(5), mae, '-x', c='y', label='CDR')
#
# ''' '''
#
#
# data = np.genfromtxt(path+'/Master.csv', delimiter=',', skiprows=1,#, 21, 22, 23,
#                      usecols=(5, 9, 10, 15, 21, 22, 23, 33, 34, 35, 36, 37, 28))
# data = data[~np.isnan(data).any(axis=1)][1:]
#
# data_normalised = np.zeros(data.shape)
# for i in range(data.shape[1]):  # Normalise data by subtracting feature means from each value
#     feature = data[:, i]
#     mean = np.mean(feature)  # Compute feature mean
#     std = np.std(feature)  # Compute feature standard deviation
#     feature_z = (feature - mean) / std  # Compute z-score based on computed mean and std
#     data_normalised[:, i] = feature_z
#
# mae = []
# for i in range(5):
#     mae_1000 = []
#     mse_1000 = []
#     train_prop = 70 + (i * 13)
#     for k in range(1000):
#         np.random.shuffle(data_normalised)
#         X_train = data_normalised[0:train_prop, 0:11]
#         y_train = data_normalised[0:train_prop, 12]
#         X_test = data_normalised[train_prop:, 0:11]
#         y_test = data_normalised[train_prop:, 12]
#         lm = LinearRegression()
#         lm.fit(X_train, y_train)
#         mae_1000.append(np.mean(np.abs(lm.predict(X_test) - y_test)))
#     mae.append(np.median(mae_1000))
# plt.plot(range(5), mae, '-x', c='r', label='Population + CDR')

plt.xlabel('Train %')
plt.ylabel('MAE')
plt.legend(loc='lower right')
plt.grid(True, which='both')
plt.minorticks_on()
plt.xticks(range(5), [50, 60, 70, 80, 90])
plt.ylim([0,1])
plt.show()



# plt.scatter(range(1000), rmse_1000)
#
# rmse_1000.append(np.sqrt(np.mean((lm.predict(X_test) - y_test) ** 2)))
# mse_1000.append(np.mean((lm.predict(X_test) - y_test) ** 2))

# print('Coefficients: \n', lm.coef_)
# print("Mean squared error: %.2f" % np.mean((lm.predict(X_test) - y_test) ** 2))
# print('Variance score: %.2f' % lm.score(X_test, y_test))

# Plot outputs
# plt.plot(range(len(y_test)), y_test, color='blue')
# plt.plot(range(len(y_test)), lm.predict(X_test), color='red')
# plt.show()

















