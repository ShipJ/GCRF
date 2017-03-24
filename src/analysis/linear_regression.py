import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from src.config import config
import sys
import matplotlib.pyplot as plt

def z_score(df):
    return (df - np.mean(df))/np.std(df)

if __name__ == '__main__':
    PATH = config.get_dir()
    country = config.get_country()
    adm = config.get_headers(country, 'adm')
    cdr = config.get_headers(country, 'cdr')
    dhs = config.get_headers(country, 'dhs')
    data = pd.DataFrame(pd.read_csv(PATH+'/final/%s/master_2.0.csv'%country,
                                    usecols=['Pagerank', 'G_residuals',
                                             'EigenvectorCentrality',
                                             'BloodPosRateSL', 'Log_pop_density', 'BloodPosRate'])).dropna()

    data = data[data['BloodPosRate'] > 0]
    data = data.ix[1:]
    z_data = pd.DataFrame(z_score(data)).as_matrix()

    mse_all = []
    for i in range(5):
        mse_1, mse_2, mse_3, mse_4, mse_5 = [], [], [], [], []
        print 'Training set %d' % i
        prop = np.floor(len(data) / 2) + (i * 7)
        for j in range(1000):
            np.random.shuffle(z_data)

            # Baseline
            X_train1 = z_data[:prop, [3]]
            y_train1 = z_data[:prop, 4]
            X_test1 = z_data[prop:, [3]]
            y_test1 = z_data[prop:, 4]
            lm1 = LinearRegression()
            lm1.fit(X_train1, y_train1)
            mse_1.append(mean_squared_error(y_test1, lm1.predict(X_test1)))
            # CDR
            X_train2 = z_data[:prop, :3]
            y_train2 = z_data[:prop, 4]
            X_test2 = z_data[prop:, :3]
            y_test2 = z_data[prop:, 4]
            lm2 = LinearRegression()
            lm2.fit(X_train2, y_train2)
            mse_2.append(mean_squared_error(y_test2, lm2.predict(X_test2)))
            # Baseline+CDR
            X_train3 = z_data[:prop, :4]
            y_train3 = z_data[:prop, 4]
            X_test3 = z_data[prop:, :4]
            y_test3 = z_data[prop:, 4]
            lm3 = LinearRegression()
            lm3.fit(X_train3, y_train3)
            mse_3.append(mean_squared_error(y_test3, lm3.predict(X_test3)))
            # Baseline+Lag
            X_train4 = z_data[:prop, [3, 5]]
            y_train4 = z_data[:prop, 4]
            X_test4 = z_data[prop:, [3, 5]]
            y_test4 = z_data[prop:, 4]
            lm4 = LinearRegression()
            lm4.fit(X_train4, y_train4)
            mse_4.append(mean_squared_error(y_test4, lm4.predict(X_test4)))
            # All
            X_train5 = z_data[:prop, np.setdiff1d(range(6), [4])]
            y_train5 = z_data[:prop, 4]
            X_test5 = z_data[prop:, np.setdiff1d(range(6), [4])]
            y_test5 = z_data[prop:, 4]
            lm5 = LinearRegression()
            lm5.fit(X_train5, y_train5)
            mse_5.append(mean_squared_error(y_test5, lm5.predict(X_test5)))

        mse_all.append([np.median(mse_1), np.median(mse_2), np.median(mse_3), np.median(mse_4), np.median(mse_5)])

    a = np.array(mse_all)

    plt.plot(range(5), a[:, 0], label='baseline')
    plt.plot(range(5), a[:, 1], label='cdr')
    plt.plot(range(5), a[:, 2], label='baseline+cdr')
    plt.plot(range(5), a[:, 3], label='baseline+lag')
    plt.plot(range(5), a[:, 4], label='all')
    plt.legend(), plt.grid()
    plt.show()
