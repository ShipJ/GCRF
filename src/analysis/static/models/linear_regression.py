import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from src.config import config
import sys
import matplotlib.pyplot as plt




if __name__ == '__main__':
    PATH = config.get_dir()
    country = config.get_country()
    adm = config.get_headers(country, 'adm')
    cdr = config.get_headers(country, 'cdr')
    dhs = config.get_headers(country, 'dhs')
    data = pd.DataFrame(pd.read_csv(PATH+'/final/%s/master_2.0.csv'%country,
                                    usecols=cdr+dhs)).dropna()
    print data.columns

    data = data.as_matrix()


    mae = []
    for i in range(5):
        prop = 49 + i*10
        mae_1000 = []
        for j in range(1000):
            np.random.shuffle(data)
            X_train = data[0:prop, [0,6,7,8,9,10,11,12,14,15]]
            y_train = data[0:prop, 16]
            X_test = data[prop:, [0,6,7,8,9,10,11,12,14,15]]
            y_test = data[prop:, 16]

            lm = LinearRegression()
            lm.fit(X_train, y_train)
            mae_1000.append(np.mean(np.abs(lm.predict(X_test) - y_test)))
        mae.append(np.mean(mae_1000))

    plt.plot(range(5), mae, '-x')
    plt.show()

