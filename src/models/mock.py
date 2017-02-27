import pandas as pd
import os

for _file in os.listdir('/data/raw/SET1'):
    print "Reading %s ..." % _file

    df = pd.DataFrame.from_csv('/data/raw/SET1/SET1TSV_0.TSV/%s' % _file, sep='\t', header=None).reset_index()
    df.columns = ['datetime', 'source', 'target', 'activity', 'duration']

    # Split file by timestamp
    time_stamped = df.groupby('datetime')

    for name, group in time_stamped:
        # Grab current datetime
        current = group['datetime'].iloc[0]
        # Extract datetime as strings
        y, m, d, h = str(current.year), str(current.month), str(current.day), str(current.hour)
        print "Reading %s-%s-%s-%s data" % (y, m, d, '{0:0>2}'.format(h))
        # Save time-stamped file
        group.to_csv('temporal/%s-%s-%s-%s.csv' % (y, m, d, '{0:0>2}'.format(h)), index=None)
