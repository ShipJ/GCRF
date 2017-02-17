import pandas as pd
import arrow as ar
import time
import sys
import math
import numpy as np

if __name__ == "__main__":

    # Start timer
    t0 = time.time()

    # Pass data set number as parameter
    print "Which data set would you like to load? Type 01, 02 etc"
    set_number = raw_input()

    PATH = 'Data/Senegal/CDR/SET1/SET1V_%s.CSV' % set_number

    # Read Senegal SET1 data sets (can run in parallel)
    print "Reading Data Set: %s" % set_number
    data = pd.DataFrame(pd.read_csv(PATH, header=None))
    data.columns = ['datetime', 'source', 'target', 'volume', 'duration']
    print "Loaded Data Set: %s" % set_number

    # Initial datetime (1st Jan 2013)
    start_datetime = ar.get(data['datetime'][0], 'YYYY-MM-DD HH')
    # Current datetime
    current_datetime = ar.get(data['datetime'][0], 'YYYY-MM-DD HH')

    # Directory
    m, w, d, h = int(set_number)-1, 0, 0, 0

    # Track indices where hour changes - store hourly chunks in separate files
    slice_ix, slices = 0, [0]

    # Used to split data by hour
    count = 0

    # Iterate through data set
    for index, row in data.iterrows():

        # Increment slicer until hour changes
        if start_datetime == current_datetime:
            print slice_ix
            slice_ix += 1


        # Hour has changed
        else:
            # Store slice indices
            slices.append(slice_ix)

            print "The hour has changed, the first hour of data is from: %s to %s" % (slices[count], slices[count+1])

            # Check for missing data ~ hours skipped (eg. 22:00:00 -> 00:00:00)
            hours_missed = (24 * (current_datetime - start_datetime).days) +\
                           ((current_datetime - start_datetime).seconds / 3600) - 1

            # No missing hours
            if hours_missed == 0:
                print "Data OK - Saving data[%s:%s]" % (slices[count], slices[count + 1]), "to: %s, %s, %s, %s" % (m, w, d, h)

                # Save data split by slices to csv
                data[:][slices[count]:slices[count+1]].to_csv(
                    "Data/Senegal/CDR/Temporal/Month_%s/Week_%s/Day_%s/Hour_%s/graph.csv" % (
                     m, w, d, h), index=None)

            # If missing hours, must save empty files
            else:
                print "There are %s missing hours between %s and %s" % (hours_missed, start_datetime, current_datetime)
                # for j in range(hours_missed):
                #     # Placeholder for no interaction
                #     empty_data = pd.DataFrame(np.array([row['datetime'], 1, 1, 0, 0]).reshape(1, 5))
                #     empty_data.columns = ['datetime', 'source', 'target', 'volume', 'duration']
                #     empty_data.to_csv("SenegalData/CDR/TemporalData/Month_%s/Week_%s/Day_%s/Hour_%s/graph.csv" % (
                #                        m, w, d, h), index=None, header=None)

            # Increment w, d, h
            h = int(math.fmod(h + hours_missed + 1, 24))
            if h == 0:
                d = int(math.fmod(d + 1, 7))
                if d == 0:
                    w += 1

            # Increment slice and count
            slice_ix += 1
            count += 1

        # Update respective datetime
        start_datetime = current_datetime
        current_datetime = ar.get(data['datetime'][slice_ix], 'YYYY-MM-DD HH')

    # End timer
    t1 = time.time()
    # Time elapsed
    print "\nRuntime: %s seconds, %s minutes, %s hours" % (t1 - t0, (t1 - t0) / 60, (t1 - t0) / 3600)
