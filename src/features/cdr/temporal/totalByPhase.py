# Jack Shipway, October 2016, UCL GCRF Research Project
#
# This file combines hourly CDR data files into a large np array, corresponding to different phases of the day.
# I.e. total_morn is an np array containing all the 'morning' hours (5am - 12pm), for each day over the entire
# 5-month period - repeated for afternoon, evening, and night. It also saves np arrays for working/non-working
# hours, rather than into morn/aft/eve/night. This is because upon interpreting results, there seemed to be
# little difference between morn/aft/eve. So I decided to combine them into one.
#
# Save phase data for whole time period - run once to obtain numpy arrays

import pandas as pd
import numpy as np
import math

# Hours per daily phase
MORN = range(5, 12)
AFT = range(12, 17)
EVE = range(17, 21)
NIGHT = [int(math.fmod(i, 24)) for i in range(21, 30)]
# Hours for working/non-working phases
WORK = range(5, 21)
NON_WORK = [int(math.fmod(j, 24)) for j in range(21, 30)]
PHASES = [MORN, AFT, EVE, NIGHT, WORK, NON_WORK]

if __name__ == "__main__":

    # Initial month/week/day: 5th Dec 2011
    m, w, d = 0, 0, 0
    # Track current day
    day_count = 0
    # 4 attributes x 1238 nodes x 140 days
    total_morn = np.zeros((4, 1238, 140))
    total_aft = np.zeros((4, 1238, 140))
    total_eve = np.zeros((4, 1238, 140))
    total_night = np.zeros((4, 1238, 140))
    total_work = np.zeros((4, 1238, 140))
    total_nwork = np.zeros((4, 1238, 140))
    totals = [total_morn, total_aft, total_eve, total_night, total_work, total_nwork]

    # For each month 
    for i in range(5):
        # For each week 
        for j in range(4):
            # For each day 
            for k in range(7):
                print "Aggregating data for day: %s" % day_count
                # Store daily data within each phase
                day_phases = [[], [], [], [], [], []]

                # For each phase & work/non-work
                for l in range(6):
                    for hour in PHASES[l]:
                        # Grab data from csv's corresponding to each phase
                        hour_data = pd.DataFrame(pd.read_csv("IvoryCoastData/CDR/TemporalData/Month_%s/Week_%s/Day_%s/Hour_%s/graph.csv" % (m, w, d, hour)))
                        # Collect for all days
                        day_phases[l].append(hour_data)

                    # Combine into 1 data frame
                    data = pd.DataFrame(pd.concat(day_phases[l], ignore_index=True))
                    # Get IDs of actively calling nodes
                    nodes_out = data.source.unique()
                    # Get IDs of actively receiving nodes
                    nodes_in = data.target.unique()

                    # CDR incoming/outgoing call volume/duration
                    vol_in = np.zeros((1, 1238))
                    vol_out = np.zeros((1, 1238))
                    dur_in = np.zeros((1, 1238))
                    dur_out = np.zeros((1, 1238))

                    # Incoming volume
                    vol_aggregate_in = data.groupby(['target'], as_index=False)['weight'].sum()
                    for index, row in vol_aggregate_in.iterrows():
                        source, volume = row['target'], row['weight']
                        vol_in[0][source] += volume
                    # Update daily phase total incoming volume
                    totals[l][0, :, day_count] = vol_in

                    # Outgoing volume
                    vol_aggregate_out = data.groupby(['source'], as_index=False)['weight'].sum()
                    for index, row in vol_aggregate_out.iterrows():
                        source, volume = row['source'], row['weight']
                        vol_out[0][source] += volume
                    # Update daily phase total outgoing volume
                    totals[l][1, :, day_count] = vol_out

                    # Incoming duration
                    dur_aggregate_in = data.groupby(['target'], as_index=False)['duration'].sum()
                    for index, row in dur_aggregate_in.iterrows():
                        source, duration = row['target'], row['duration']
                        dur_in[0][source] += duration
                    # Update daily phase total incoming duration
                    totals[l][2, :, day_count] = dur_in

                    # Outgoing duration
                    dur_aggregate_out = data.groupby(['source'], as_index=False)['duration'].sum()
                    for index, row in dur_aggregate_out.iterrows():
                        source, duration = row['source'], row['duration']
                        dur_out[0][source] += duration
                    # Update daily phase total outgoing duration
                    totals[l][3, :, day_count] = dur_out

                # Increment day
                day_count += 1
                # Next day of the week
                d = int(math.fmod(d + 1, 7))
            # Next week of the month
            w = int(math.fmod(w + 1, 4))
        # Next month of the year
        m = int(math.fmod(m + 1, 5))

    print "Saving Total Morning Data"
    np.save("IvoryCoastData/CDR/AggregateData/total_morn", totals[0])
    print "Saving Total Afternoon Data"
    np.save("IvoryCoastData/CDR/AggregateData/total_aft", totals[1])
    print "Saving Total Evening Data"
    np.save("IvoryCoastData/CDR/AggregateData/total_eve", totals[2])
    print "Saving Total Night Data"
    np.save("IvoryCoastData/CDR/AggregateData/total_night", totals[3])
    print "Saving Total Working Hour Data"
    np.save("IvoryCoastData/CDR/AggregateData/total_working", totals[4])
    print "Saving Total Non-working Hour Data"
    np.save("IvoryCoastData/CDR/AggregateData/total_nworking", totals[5])