"""
This module takes raw csv/tsv data containing call detail records
and converts it into individual time-stamped csv files
"""

import os
import pandas as pd
import tarfile as tf
from src.config import config


def process_raw(path, country):
    """
    From raw zip/tar files containing CDR data, extract tsv's,
    split files by timestamp, store in individual files.

    :param path: str - system path to data directory
    :param country: str - country location tag

    :return:
    """
    if country == 'civ':
        files = tf.open(path+'/raw/%s/cdr/SET1TSV.tgz' % country, 'r')
        for f in files:
            print "Reading %s..." % f.name
            df = pd.DataFrame.from_csv(files.extractfile(f), sep='\t', header=None).reset_index()
            df.columns = ['datetime', 'source', 'target', 'activity', 'duration']
            # Split file by timestamp
            time_stamped = df.groupby('datetime')
            save_timestamp(PATH, time_stamped, country)

    elif country == 'sen':
        files = [i for i in os.listdir(path+'/raw/%s/cdr' % country) if i.startswith('SET1V')]
        for f in files:
            print "Reading %s..." % f
            df = pd.DataFrame.from_csv(path+'/raw/%s/cdr/'% country+f, sep=',', header=None).reset_index()
            df.columns = ['datetime', 'source', 'target', 'activity', 'duration']
            # Split file by timestamp
            time_stamped = df.groupby('datetime')
            save_timestamp(PATH, time_stamped, country)


def save_timestamp(path, time_stamped, country):
    """
    Take time-stamped data and store in target file

    :param path: str - system path to data directory
    :param time_stamped: pandas groupby object - from process_raw()
    :param country: str - country location tag

    :return:
    """
    for name, group in time_stamped:
        # Grab current datetime
        current = group['datetime'].iloc[0]
        # Extract datetime as strings
        y, m, d, h = str(current.year), str(current.month), str(current.day), str(current.hour)
        print "Reading %s-%s-%s-%s data" % (y, m, d, '{0:0>2}'.format(h))
        # Store as a time-stamped file
        group.to_csv(path+'/interim/%s/cdr/%s-%s-%s-%s.csv' % (country, y, m, d, '{0:0>2}'.format(h)), index=None)


if __name__ == '__main__':
    # Path to data
    PATH = config.get_dir()
    # Ask for country code
    country = config.get_country()
    # Grab raw, save to interim
    process_raw(PATH, country)
