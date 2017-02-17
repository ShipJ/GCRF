# Jack Shipway, 6/10/16, UCL GCRF Project
#
#
# Input: csv files over a user defined period of time
# Output: mean incoming/outgoing volume/duration +- std

# Data coverage: 5/12/2011 -> 22/04/12, 4.6 months, 20 weeks, 140 days, 3360 hours

import pandas as pd
import numpy as np
import time
import math
import matplotlib.pyplot as plt

# Start timer
t0 = time.time()

if __name__ == "__main__":

     # datetime identifiers
    hour, day, week, month = 0, 0, 0, 0
    print month, week, day, hour

    # Store sum of each hour in separate spot
    total_incoming_vol = np.zeros((168, 1238))
    total_outgoing_vol = np.zeros((168, 1238))
    total_incoming_dur = np.zeros((168, 1238))
    total_outgoing_dur = np.zeros((168, 1238))

    for i in range(168):
        data = pd.DataFrame(pd.read_csv("Data/IvoryCoast/CDR/TemporalData/Month_%s/Week_%s/Day_%s/Hour_%s/graph.csv" % (month, week, day, hour), sep=','))
        data.columns = ['datetime', 'source', 'target', 'volume', 'duration']
        data = data[['source', 'target', 'volume', 'duration']]

        # Get IDs of towers with outgoing calls
        nodes_out = data.source.unique()
        num_out = len(nodes_out)
        nodes_in = data.target.unique()
        num_in = len(nodes_in)

        # Aggregate call volume (weight) at monthly level
        #print "Aggregating outgoing call volume"
        vol_out = data.groupby(['source'], as_index=False)['volume'].sum()
        for index, row in vol_out.iterrows():
            source, volume = row['source'], row['volume']
            total_outgoing_vol[i][source] += volume

        #print "Aggregating incoming call volume"
        vol_in = data.groupby(['target'], as_index=False)['volume'].sum()
        for index, row in vol_in.iterrows():
            source, volume = row['target'], row['volume']
            total_incoming_vol[i][source] += volume

        #print "Aggregating outgoing call duration"
        dur_out = data.groupby(['source'], as_index=False)['duration'].sum()
        for index, row in dur_out.iterrows():
            source, duration = row['source'], row['duration']
            total_outgoing_dur[i][source] += duration

        #print "Aggregating outgoing call duration"
        dur_in = data.groupby(['target'], as_index=False)['duration'].sum()
        for index, row in dur_in.iterrows():
            source, duration = row['target'], row['duration']
            total_incoming_dur[i][source] += duration

        hour = int(math.fmod(hour + 1, 24))
        if hour == 0:
            day = int(math.fmod(day + 1, 7))
        if hour == 0 and day == 0:
            week = int(math.fmod(week + 1, 4))
        if hour == 0 and day == 0 and week == 0:
            month = int(math.fmod(month + 1, 5))
        print month, week, day, hour

    # Average
    incoming_vol_avg = np.mean(total_incoming_vol, axis=0)
    #print incoming_vol_avg, incoming_vol_avg.size, incoming_vol_avg[15]
    outgoing_vol_avg = np.mean(total_outgoing_vol, axis=0)
    #print outgoing_vol_avg, outgoing_vol_avg.size, outgoing_vol_avg[15]
    incoming_dur_avg = np.mean(total_incoming_dur, axis=0)
    #print incoming_vol_avg, incoming_vol_avg.size, incoming_vol_avg[15]
    outgoing_dur_avg = np.mean(total_outgoing_dur, axis=0)
    #print incoming_vol_avg, incoming_vol_avg.size, incoming_vol_avg[15]

    # Standard deviation
    incoming_vol_std = np.std(total_incoming_vol, axis=0)
    outgoing_vol_std = np.std(total_outgoing_vol, axis=0)
    incoming_dur_std = np.std(total_incoming_dur, axis=0)
    outgoing_dur_std = np.std(total_outgoing_dur, axis=0)

    plt.errorbar(range(100), outgoing_vol_avg[0:100], yerr=outgoing_vol_std[0:100])
    plt.show()

    # End timer
    t1 = time.time()
    # Time elapsed
    print "\nRuntime: %s seconds, %s minutes, %s hours" % (t1-t0, (t1-t0)/60, (t1-t0)/3600)





