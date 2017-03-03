from src.config import config
from src.analysis.static import feature_selection


if __name__ == '__main__':
    country = config.get_country()

    results = feature_selection.stepwise_regression(country)




