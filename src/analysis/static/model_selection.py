from src.config import config
from src.analysis.static.feature_selection import stepwise_regression
from src.analysis.static.outliers import is_outlier


if __name__ == '__main__':
    country = config.get_country()

    print 'hey'

    # for response in ['BloodPosRate', 'RapidPosRate', 'HIVPosRate', 'DeathRate', 'HealthAccessDifficulty']:
    #     print 'Running feature-selection for: %s' % response
    #     print stepwise_regression(country, response)











