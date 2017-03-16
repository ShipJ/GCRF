from src.config import config
from src.analysis.static.feature_selection import stepwise_regression
import numpy as np


if __name__ == '__main__':
    country = config.get_country()
    dhs_features = config.get_headers(country, 'dhs')

    for response in dhs_features:
        print 'Running feature-selection for: %s' % response
        if country == 'civ':
            a = [stepwise_regression(country, response).get_string()]
        elif country == 'sen':
            a = [stepwise_regression(country, response).get_string()]
        else:
            a = ['']

        # np.savetxt('../../../reports/results/%s/r_squared/%s.txt' % response, a, fmt='%s')















