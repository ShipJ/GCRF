import pandas as pd
import tarfile
import zipfile
import glob
import sys
import os


def process_raw(source, target):
    """
    From raw zip/tar files containing CDR data, extract tsv's,
    split files by timestamp, store in separate files.

    :param source: string - file path to raw data.
    :param target: string - file path to save processed data.
    :param country: string - country code.

    :return: None.
    """
    tar = tarfile.open(source, 'r')

    for file in tar:
        print "Reading %s ..." % tar.name

        df = pd.DataFrame.from_csv(source+file.name, sep='\t', header=None).reset_index()
        df.columns = ['datetime', 'source', 'target', 'activity', 'duration']

        # Split file by timestamp
        time_stamped = df.groupby('datetime')

        # Save each split to a time-stamped file
        for name, group in time_stamped:
            # Grab current datetime
            current = group['datetime'].iloc[0]
            # Extract datetime as strings
            y, m, d, h = str(current.year), str(current.month), str(current.day), str(current.hour)
            print "Reading %s-%s-%s-%s data" % (y, m, d, '{0:0>2}'.format(h))
            # Save time-stamped file
            group.to_csv(target+'/%s-%s-%s-%s.csv' % (y, m, d, '{0:0>2}'.format(h)), index=None)


def get_country():
    """
    Ask for user to input a country.

    :return: string - country for which there is data.
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
    Ask the user which data set they would like to process

    :return: string - data set reference
    """
    print "Process which data set? [1, 2, 3 or 4?]: "
    data_set = raw_input()
    if data_set in ['1', '2', '3', '4']:
        return data_set
    else:
        print "Please type an actual data set. \n"
        return get_data_set()


if __name__ == '__main__':

    # Retrieve country code and data set to process
    country = get_country()
    data_set = get_data_set()

    # Grab from source, process, save to target
    source = '../../data/raw/%s/CDR%s.zip' % (country, data_set)
    target = '../../data/processed/%s/CDR/timestamp/' % country

    process_raw(source, target)
