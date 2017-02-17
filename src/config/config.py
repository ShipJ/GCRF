"""
This file contains various functions that used by various scripts, as well as providing constant values for
different countries - i.e. data that will not change.
"""

def get_country():
    """
    Ask user for country code.

    :return: str - country for which there is data.
    """
    print "Process data for which country? ['sen': Senegal, 'civ': Ivory Coast]: "
    input_country = raw_input()
    if input_country == 'sen':
        country = 'sen'
    elif input_country == 'civ':
        country = 'civ'
    else:
        print "Please type the country abbreviation (lower case): "
        return get_country()
    return country


def get_constants(country):
    if country == 'sen':
        constants = {'country' : 'civ', 'num_towers' : 1240, 'hours' : 3278}
        return constants
    elif country == 'civ':
        constants = {'country': 'civ', 'num_towers': 1240, 'hours': 3278}
        return constants
    else:
        print "Please type the country abbreviation (lower case): "
        return get_constants(country)
