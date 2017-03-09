from src.config import config
from src.analysis.static.feature_selection import stepwise_regression


if __name__ == '__main__':
    country = config.get_country()

    if country == 'civ':
        for response in ['BloodPosRate', 'RapidPosRate', 'HIVPosRate', 'DeathRate', 'HealthAccessDifficulty']:
            print 'Running feature-selection for: %s' % response
            print stepwise_regression(country, response)

    elif country == 'sen':
        for response in ['BloodPosRate', 'RapidPosRate', 'DeathRate']:
            print 'Running feature-selection for: %s' % response
            print stepwise_regression(country, response)










