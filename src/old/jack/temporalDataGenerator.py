import pandas as pd
import numpy as np
import arrow as ar
import time
import sys
import math

#Define constants
DATA_SETS = 10
MONTHS = 5
WEEKS = 20
DAYS = 140
HOURS = 3360
CELL_TOWERS = 1238

# Start timer
t0 = time.time()

if __name__ == "__main__":

    current = ar.Arrow(2012, 4, 19, 22, 0, 0)

    # datetime identifiers
    month, week, day, hour = 4, 3, 3, 22

    for i in range(1):
        print "Reading Data Set %s" % i
        data = pd.DataFrame(pd.read_csv("IvoryCoastData/CDR/ICS1/ICS1AntToAnt_9.TSV", sep='\t', header=None))
        data.columns = ['datetime', 'source', 'target', 'weight', 'duration']
        print "Loaded Data Set %s" % i

        slice_ix, slices = 12440688, [12440688]
        count = 0

        # Iterate through whole data set
        for index, row in data[:][12440688:].iterrows():

            # Increment slicer until hour changes
            if ar.get(row['datetime']) == current:
                slice_ix += 1

            # When hour changes
            else:
                # Increment slicer, also save index
                slices.append(slice_ix)
                slice_ix += 1

                # Checking for hours skipped: i.e. 22:00:00 -> 00:00:00... 1 hour is normal
                if ((ar.get(row['datetime'])-current).seconds/3600) == 1:
                    # Save data split by slices to csv
                    data[:][slices[count]:slices[count+1]].to_csv("IvoryCoastData/CDR/TemporalData/Month_%s/Week_%s/Day_%s/Hour_%s/graph.csv" % (month, week, day, hour), sep=',', index=None)

                # If there is more than 1 hour skipped, still need to save empty csv's for missed hours
                else:
                    # How many hours were skipped
                    print ar.get(row['datetime']), current
                    num_missing = ((ar.get(row['datetime'])-current).seconds/3600)-1
                    print "There are %s missing hours between: " % num_missing, ar.get(row['datetime']), 'and', ar.get(row['datetime']).replace(hours=-(num_missing+1))
                    # Save an empty csv for the missing graphs
                    for j in range(num_missing):
                        hour = int(math.fmod(hour + 1, 24))
                        if hour == 0:
                            day = int(math.fmod(day + 1, 7))
                            if day == 0:
                                week = int(math.fmod(week + 1, 4))
                                if week == 0:
                                    month = int(math.fmod(month + 1, 5))
                        # Get datetime of first, second, ... missing data
                        date = ar.get(row['datetime']).replace(hours=-(num_missing - j))
                        nothing = np.array([date, 15, 15, 0, 0]).reshape(1, 5)
                        nothing = pd.DataFrame(nothing)
                        nothing.to_csv("IvoryCoastData/CDR/TemporalData/Month_%s/Week_%s/Day_%s/Hour_%s/graph.csv" % (month, week, day, hour), sep=',', index=None, header=None)

                current = ar.get(data['datetime'][slice_ix])
                count += 1

                # Keep track of directories
                hour = int(math.fmod(hour + 1, 24))
                if hour == 0:
                    day = int(math.fmod(day + 1, 7))
                    if day == 0:
                        week = int(math.fmod(week + 1, 4))
                        if week == 0:
                            month = int(math.fmod(month + 1, 5))
                print month, week, day, hour


        del data
        current.replace(hours=1)

# End timer
t1 = time.time()
# Time elapsed
print "\nRuntime: %s seconds, %s minutes, %s hours" % (t1-t0, (t1-t0)/60, (t1-t0)/3600)
