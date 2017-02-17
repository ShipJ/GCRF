import pandas as pd
import tarfile
import os
import sys


def process_raw(source, target, country):
    """
    From raw zip/tar files containing CDR data, extract tsv's,
    split files by timestamp, store in separate files.

    :param source: str - file path to raw data.
    :param target: str - file path to save processed data.
    :param country: str - country code.

    :return: None.
    """
    if country == 'civ':
        file = tarfile.open(source+'.zip', 'r')
        for f in file:
            print "Reading %s..." % f.name
            df = pd.DataFrame.from_csv(file.extractfile(f), sep='\t', header=None).reset_index()
            df.columns = ['datetime', 'source', 'target', 'activity', 'duration']
            # Split file by timestamp
            time_stamped = df.groupby('datetime')
            save_timestamp(time_stamped, target)

    elif country == 'sen':
        file = [i for i in os.listdir(source) if i.startswith('SET1V')]
        for f in file:
            print "Reading %s..." % f
            df = pd.DataFrame.from_csv(source+'/'+f, sep=',', header=None).reset_index()
            df.columns = ['datetime', 'source', 'target', 'activity', 'duration']
            # Split file by timestamp
            time_stamped = df.groupby('datetime')
            save_timestamp(time_stamped, target)


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


def get_data_set():
    """
    Ask user for data set to process.

    :return: str - data set reference
    """
    print "Process which data set? [1, 2, 3 or 4?]: "
    data_set = raw_input()
    if data_set in ['1', '2', '3', '4']:
        return data_set
    else:
        print "Please type an actual data set. \n"
        return get_data_set()


def save_timestamp(time_stamped, target):
    """
    Take time-stamped data and store in target file

    :param time-stamped: pandas groupby object - from process_raw()
    :param target: str - store in interim data files

    :return: None
    """
    for name, group in time_stamped:
        # Grab current datetime
        current = group['datetime'].iloc[0]
        # Extract datetime as strings
        y, m, d, h = str(current.year), str(current.month), str(current.day), str(current.hour)
        print "Reading %s-%s-%s-%s data" % (y, m, d, '{0:0>2}'.format(h))
        # Store as a time-stamped file
        group.to_csv(target + '/%s-%s-%s-%s.csv' % (y, m, d, '{0:0>2}'.format(h)), index=None)


if __name__ == '__main__':
    # Ask user for country code and data set
    country = get_country()
    data_set = get_data_set()

    # Grab data from raw, process, save to interim
    source = '../../data/raw/%s/CDR%s' % (country, data_set)
    target = '../../data/interim/%s/CDR/timestamp/' % country

    process_raw(source, target, country)
