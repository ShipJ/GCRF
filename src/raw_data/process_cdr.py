"""
This module takes raw csv/tsv data containing call detail records
and converts it into individual time-stamped csv files
"""

import pandas as pd
import tarfile as tf
import os
from src.config import config


def process_raw(source, target, country):
    """
    From raw zip/tar files containing CDR data, extract tsv's,
    split files by timestamp, store in individual files.

    :param source: str - file path to raw data.
    :param target: str - file path to save processed data.
    :param country: str - country code.

    :return: None.
    """
    if country == 'civ':
        file = tf.open(source+'.zip', 'r')
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


def which_data():
    """
    Ask user for particular data set to process - there are four from the D4D challenge

    :return: str - data set reference
    """
    print "Process which data set? [1, 2, 3 or 4?]: "
    data_set = raw_input()
    if data_set in ['1', '2', '3', '4']:
        return data_set
    else:
        print "Please type an actual data set. \n"
        return which_data()


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
    country = config.get_country()
    data_source = config.get_dir()
    data_set = which_data()

    # Grab data from raw, process, save to interim
    source = data_source+'/raw/%s/CDR%s' % (country, data_set)
    target = source+'/interim/%s/CDR/timestamp/' % country

    process_raw(source, target, country)
