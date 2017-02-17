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

    month, week, day, hour = 0, 0, 0, 0

    for i in range(5):
        for j in range(4):
            week_list = []
            for k in range(7):
                day_list = []
                for l in range(24):
                    print "Reading M: %s W: %s, D: %s, H: %s .csv data" % (month, week, day, hour)
                    hour_data = pd.DataFrame(pd.read_csv("IvoryCoastData/CDR/TemporalData/Month_%s/Week_%s/Day_%s/Hour_%s/graph.csv" % (month, week, day, hour), sep=','))
                    day_list.append(hour_data)
                    hour = int(math.fmod(hour + 1, 24))
                day_data = pd.DataFrame(pd.concat(day_list), index=None)
                print "Saving data to M: %s, W: %s, D: %s, H: %s" % (month, week, day, hour)
                day_data.to_csv("IvoryCoastData/CDR/TemporalData/Month_%s/Week_%s/Day_%s/graph.csv" % (month, week, day), sep=',', index=None)
                day = int(math.fmod(day + 1, 7))
                #week_list.append(day_data)
            # week_data = pd.DataFrame(pd.concat(week_list), index=None)
            # week_data.to_csv("IvoryCoastData/CDR/TemporalData/Month_%s/Week_%s/graph.csv" % (month, week), sep=',', index=None)
            week = int(math.fmod(week + 1, 4))
        month = int(math.fmod(month + 1, 5))


