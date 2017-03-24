from src.config import config
from src.analysis.feature_selection import stepwise_regression

if __name__ == '__main__':
    country = config.get_country()
    dhs_features = config.get_headers(country, 'dhs')

    for response in dhs_features:
        print 'Running feature-selection for: %s' % response
        if country == 'civ':
            a = [stepwise_regression(country, response).get_string()]
        elif country == 'sen':
            a = [stepwise_regression(country, response).get_string()]
















