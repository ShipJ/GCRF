# Jack Shipway, 6/10/16, UCL GCRF Project
#
# Run once to generate csv files: takes ~ 5 hours
#
# Input: CDR Set 1 - Antenna to Antenna
# Output: csv file of edges in each time-step
# Data type: tsv
# Data coverage: 5/12/2011 -> 22/04/12, 4.6 months, 20 weeks, 140 days, 3360 hours

import pandas as pd
import numpy as np
import arrow as ar
import time
import sys
import math

#Define constants
MONTHS = 5
WEEKS = 20
DAYS = 140
HOURS = 3360
CELL_TOWERS = 1238

# Start timer
t0 = time.time()

if __name__ == "__main__":

    # datetime identifiers
    month, week, day, hour = 0, 0, 0, 0

    for i in range(1):
        # Run for each data set
        print "Reading Data Set %s" % i
        data = pd.DataFrame(pd.read_csv("Data/IvoryCoast/CDR/SET1/ICS1AntToAnt_0.TSV", sep='\t', header=None))
        data.columns = ['datetime', 'source', 'target', 'weight', 'duration']
        print "Loaded Data Set %s successfully" % i

        slice_ix, slices = 0, [0]
        current = ar.get(data['datetime'][0])
        count = 0
        for index, row in data.iterrows():
            if ar.get(row['datetime']) == current:
                pass
            else:
                slices.append(slice_ix)
                data[:][slices[count]:slices[count+1]].to_csv("Data/IvoryCoast/CDR/TemporalData/Month_%s/Week_%s/Day_%s/Hour_%s/graph.csv" % (month, week, day, hour), sep=',', index=None)
                current = ar.get(data['datetime'][slice_ix])
                hour = int(math.fmod(hour + 1, 24))
                if hour == 0:
                    day = int(math.fmod(day + 1, 7))
                    if day == 0:
                        week = int(math.fmod(week + 1, 4))
                        if week == 0:
                            month = int(math.fmod(month + 1, 5))
                print month, week, day, hour
                count += 1
            slice_ix += 1


    # datetime identifiers
    week, day, hour = 0, 0, 0

    for i in range(14):
        daily_data = []
        for j in range(23):
            hourly_data = pd.DataFrame(pd.read_csv("Data/IvoryCoast/CDR/TemporalData/Month_0/Week_%s/Day_%s/Hour_%s/graph.csv" % (week, day, hour), sep=',', header=None))
            daily_data.append(hourly_data)
            hour += 1

        daily_data = pd.DataFrame(pd.concat(daily_data), index=None)
        daily_data.to_csv("Data/IvoryCoast/CDR/TemporalData/Month_0/Week_%s/Day_%s/graph.csv" % (week, day), sep=',', index=None)
        hour = 0
        day += 1
        if i == 6:
            week += 1
            day = 0


# End timer
t1 = time.time()
# Time elapsed
print "\nRuntime: %s seconds, %s minutes, %s hours" % (t1-t0, (t1-t0)/60, (t1-t0)/3600)