# Jack Shipway, October 2016, UCL GCRF Research Project
#
# This file combines hourly CDR data files into a large np array, corresponding to different phases of the day.
# I.e. total_morn is an np array containing all the 'morning' hours (5am - 12pm), for each day over the entire
# 5-month period - repeated for afternoon, evening, and night
#
# Save phase data for 1 month (28 day) periods, starting 5/12/11, ending 22/4/12

import pandas as pd
import numpy as np
import math

# Hours in each daily phase
MORN = range(5, 12)
AFT = range(12, 17)
EVE = range(17, 21)
NIGHT = [int(math.fmod(i, 24)) for i in range(21, 30)]
PHASES = [MORN, AFT, EVE, NIGHT]

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
    totals = [total_morn, total_aft, total_eve, total_night]

    # For each month
    for i in range(5):
        # For each week
        for j in range(4):
            # For each day
            for k in range(7):
                print "Aggregating data for day: %s" % day_count
                # Store daily data within each phase
                day_phases = [[], [], [], []]

                # For each phase
                for l in range(4):
                    for hour in PHASES[l]:
                        # Grab data from csv's corresponding to each phases
                        hour_data = pd.DataFrame(pd.read_csv(
                            "Data/IvoryCoast/CDR/TemporalData/Month_%s/Week_%s/Day_%s/Hour_%s/graph.csv" % (
                            m, w, d, hour)))
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
                        vol_out[0][source] += volume
                    # Update daily phase total incoming volume
                    totals[l][0, :, day_count] = vol_in
                    # Outgoing volume
                    vol_aggregate_out = data.groupby(['source'], as_index=False)['weight'].sum()
                    for index, row in vol_aggregate_out.iterrows():
                        source, volume = row['source'], row['weight']
                        vol_in[0][source] += volume
                    # Update daily phase total outgoing volume
                    totals[l][1, :, day_count] = vol_out
                    # Incoming duration
                    dur_aggregate_in = data.groupby(['target'], as_index=False)['duration'].sum()
                    for index, row in dur_aggregate_in.iterrows():
                        source, duration = row['target'], row['duration']
                        dur_out[0][source] += duration
                    # Update daily phase total incoming duration
                    totals[l][2, :, day_count] = dur_in
                    # Outgoing duration
                    dur_aggregate_out = data.groupby(['source'], as_index=False)['duration'].sum()
                    for index, row in dur_aggregate_out.iterrows():
                        source, duration = row['source'], row['duration']
                        dur_in[0][source] += duration
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







#
    # # Dec
    # print "saving 1"
    # np.save("IvoryCoastData/CDR/AggregateData/jan_morn", total_morn[:, :, 0:28])
    # print "saving 2"
    # np.save("IvoryCoastData/CDR/AggregateData/jan_aft", total_aft[:, :, 0:28])
    # print "saving 3"
    # np.save("IvoryCoastData/CDR/AggregateData/jan_eve", total_eve[:, :, 0:28])
    # print "saving 4"
    # np.save("IvoryCoastData/CDR/AggregateData/jan_night", total_night[:, :, 0:28])
    #
    # # Jan
    # print "saving 1"
    # np.save("IvoryCoastData/CDR/AggregateData/feb_morn", total_morn[:, :, 28:56])
    # print "saving 2"
    # np.save("IvoryCoastData/CDR/AggregateData/feb_aft", total_aft[:, :, 28:56])
    # print "saving 3"
    # np.save("IvoryCoastData/CDR/AggregateData/feb_eve", total_eve[:, :, 28:56])
    # print "saving 4"
    # np.save("IvoryCoastData/CDR/AggregateData/feb_night", total_night[:, :, 28:56])
    #
    # # Feb
    # print "saving 1"
    # np.save("IvoryCoastData/CDR/AggregateData/mar_morn", total_morn[:, :, 56:84])
    # print "saving 2"
    # np.save("IvoryCoastData/CDR/AggregateData/mar_aft", total_aft[:, :, 56:84])
    # print "saving 3"
    # np.save("IvoryCoastData/CDR/AggregateData/mar_eve", total_eve[:, :, 56:84])
    # print "saving 4"
    # np.save("IvoryCoastData/CDR/AggregateData/mar_night", total_night[:, :, 56:84])
    #
    # # Mar
    # print "saving 1"
    # np.save("IvoryCoastData/CDR/AggregateData/apr_morn", total_morn[:, :, 84:112])
    # print "saving 2"
    # np.save("IvoryCoastData/CDR/AggregateData/apr_aft", total_morn[:, :, 84:112])
    # print "saving 3"
    # np.save("IvoryCoastData/CDR/AggregateData/apr_morn", total_morn[:, :, 84:112])
    # print "saving 4"
    # np.save("IvoryCoastData/CDR/AggregateData/apr_eve", total_morn[:, :, 84:112])
    #
    # # Apr
    # print "saving 1"
    # np.save("IvoryCoastData/CDR/AggregateData/dec_morn", total_morn[:, :, 112:140])
    # print "saving 2"
    # np.save("IvoryCoastData/CDR/AggregateData/dec_aft", total_aft[:, :, 112:140])
    # print "saving 3"
    # np.save("IvoryCoastData/CDR/AggregateData/dec_eve", total_eve[:, :, 112:140])
    # print "saving 4"
    # np.save("IvoryCoastData/CDR/AggregateData/dec_night", total_night[:, :, 112:140])



    # # Mon
    # print "saving 1"
    # np.save("IvoryCoastData/CDR/AggregateData/mon_morn", total_morn[:, :, ::7])
    # print "saving 2"
    # np.save("IvoryCoastData/CDR/AggregateData/mon_aft", total_aft[:, :, ::7])
    # print "saving 3"
    # np.save("IvoryCoastData/CDR/AggregateData/mon_eve", total_eve[:, :, ::7])
    # print "saving 4"
    # np.save("IvoryCoastData/CDR/AggregateData/mon_night", total_night[:, :, ::7])
    #
    # # Tues
    # print "saving 1"
    # np.save("IvoryCoastData/CDR/AggregateData/tue_morn", total_morn[:, :, 1::7])
    # print "saving 2"
    # np.save("IvoryCoastData/CDR/AggregateData/tue_aft", total_aft[:, :, 1::7])
    # print "saving 3"
    # np.save("IvoryCoastData/CDR/AggregateData/tue_eve", total_eve[:, :, 1::7])
    # print "saving 4"
    # np.save("IvoryCoastData/CDR/AggregateData/tue_night", total_night[:, :, 1::7])
    #
    # # Weds
    # print "saving 1"
    # np.save("IvoryCoastData/CDR/AggregateData/wed_morn", total_morn[:, :, 2::7])
    # print "saving 2"
    # np.save("IvoryCoastData/CDR/AggregateData/wed_aft", total_aft[:, :, 2::7])
    # print "saving 3"
    # np.save("IvoryCoastData/CDR/AggregateData/wed_eve", total_eve[:, :, 2::7])
    # print "saving 4"
    # np.save("IvoryCoastData/CDR/AggregateData/wed_night", total_night[:, :, 2::7])
    #
    # # Thurs
    # print "saving 1"
    # np.save("IvoryCoastData/CDR/AggregateData/thu_morn", total_morn[:, :, 3::7])
    # print "saving 2"
    # np.save("IvoryCoastData/CDR/AggregateData/thu_aft", total_aft[:, :, 3::7])
    # print "saving 3"
    # np.save("IvoryCoastData/CDR/AggregateData/thu_eve", total_eve[:, :, 3::7])
    # print "saving 4"
    # np.save("IvoryCoastData/CDR/AggregateData/thu_night", total_night[:, :, 3::7])
    #
    # # Fri
    # print "saving 1"
    # np.save("IvoryCoastData/CDR/AggregateData/fri_morn", total_morn[:, :, 4::7])
    # print "saving 2"
    # np.save("IvoryCoastData/CDR/AggregateData/fri_aft", total_aft[:, :, 4::7])
    # print "saving 3"
    # np.save("IvoryCoastData/CDR/AggregateData/fri_eve", total_eve[:, :, 4::7])
    # print "saving 4"
    # np.save("IvoryCoastData/CDR/AggregateData/fri_night", total_night[:, :, 4::7])
    #
    # # Sat
    # print "saving 1"
    # np.save("IvoryCoastData/CDR/AggregateData/sat_morn", total_morn[:, :, 5::7])
    # print "saving 2"
    # np.save("IvoryCoastData/CDR/AggregateData/sat_aft", total_aft[:, :, 5::7])
    # print "saving 3"
    # np.save("IvoryCoastData/CDR/AggregateData/sat_eve", total_eve[:, :, 5::7])
    # print "saving 4"
    # np.save("IvoryCoastData/CDR/AggregateData/sat_night", total_night[:, :, 5::7])

    # # Sun
    # print "saving 1"
    # np.save("IvoryCoastData/CDR/AggregateData/sun_morn", totals[0][:, :, 6::7])
    # print "saving 2"
    # np.save("IvoryCoastData/CDR/AggregateData/sun_aft", totals[1][:, :, 6::7])
    # print "saving 3"
    # np.save("IvoryCoastData/CDR/AggregateData/sun_eve", totals[2][:, :, 6::7])
    # print "saving 4"
    # np.save("IvoryCoastData/CDR/AggregateData/sun_night", totals[3][:, :, 6::7])
    #





















    #
    # ''' Plotting of Results '''
    #
    # # take average over time axis, want a 4 x 1238 array for each
    # monthly_avg_morn = np.multiply(np.mean(total_morn, axis=2), (8/7.0))
    # monthly_std_morn = np.multiply(np.std(total_morn, axis=2), (8/7.0))
    # # monthly_avg_morn[0] /= float(np.max(monthly_avg_morn[0]))
    # # monthly_std_morn[0] /= float(np.max(monthly_std_morn[0]))
    # index1 = np.argsort(monthly_avg_morn[0])
    # index2 = np.argsort(monthly_avg_morn[1])
    # index3 = np.argsort(monthly_avg_morn[2])
    # index4 = np.argsort(monthly_avg_morn[3])
    # monthly_avg_morn[0] = monthly_avg_morn[0][index1]
    # monthly_std_morn[0] = monthly_std_morn[0][index1]
    # monthly_avg_morn[1] = monthly_avg_morn[1][index2]
    # monthly_std_morn[1] = monthly_std_morn[1][index2]
    # monthly_avg_morn[2] = monthly_avg_morn[2][index3]
    # monthly_std_morn[2] = monthly_std_morn[2][index3]
    # monthly_avg_morn[3] = monthly_avg_morn[3][index4]
    # monthly_std_morn[3] = monthly_std_morn[3][index4]
    #
    # # take average over time axis, want a 4 x 1238 array for each
    # monthly_avg_aft = np.multiply(np.mean(total_aft, axis=2), 1.6)
    # monthly_std_aft = np.multiply(np.std(total_aft, axis=2), 1.6)
    # # monthly_avg_morn[0] /= float(np.max(monthly_avg_morn[0]))
    # # monthly_std_morn[0] /= float(np.max(monthly_std_morn[0]))
    # index1 = np.argsort(monthly_avg_aft[0])
    # index2 = np.argsort(monthly_avg_aft[1])
    # index3 = np.argsort(monthly_avg_aft[2])
    # index4 = np.argsort(monthly_avg_aft[3])
    # monthly_avg_aft[0] = monthly_avg_aft[0][index1]
    # monthly_std_aft[0] = monthly_std_aft[0][index1]
    # monthly_avg_aft[1] = monthly_avg_aft[1][index2]
    # monthly_std_aft[1] = monthly_std_aft[1][index2]
    # monthly_avg_aft[2] = monthly_avg_aft[2][index3]
    # monthly_std_aft[2] = monthly_std_aft[2][index3]
    # monthly_avg_aft[3] = monthly_avg_aft[3][index4]
    # monthly_std_aft[3] = monthly_std_aft[3][index4]
    #
    #
    # # take average over time axis, want a 4 x 1238 array for each
    # monthly_avg_eve = np.multiply(np.mean(total_eve, axis=2), 2)
    # monthly_std_eve = np.multiply(np.std(total_eve, axis=2), 2)
    # # monthly_avg_morn[0] /= float(np.max(monthly_avg_morn[0]))
    # # monthly_std_morn[0] /= float(np.max(monthly_std_morn[0]))
    # index1 = np.argsort(monthly_avg_eve[0])
    # index2 = np.argsort(monthly_avg_eve[1])
    # index3 = np.argsort(monthly_avg_eve[2])
    # index4 = np.argsort(monthly_avg_eve[3])
    # monthly_avg_eve[0] = monthly_avg_eve[0][index1]
    # monthly_std_eve[0] = monthly_std_eve[0][index1]
    # monthly_avg_eve[1] = monthly_avg_eve[1][index2]
    # monthly_std_eve[1] = monthly_std_eve[1][index2]
    # monthly_avg_eve[2] = monthly_avg_eve[2][index3]
    # monthly_std_eve[2] = monthly_std_eve[2][index3]
    # monthly_avg_eve[3] = monthly_avg_eve[3][index4]
    # monthly_std_eve[3] = monthly_std_eve[3][index4]
    #
    # monthly_avg_night = np.mean(total_night, axis=2)
    # monthly_std_night = np.std(total_night, axis=2)
    # # monthly_avg_night[0] /= float(np.max(monthly_avg_night[0]))
    # # monthly_std_night[0] /= float(np.max(monthly_std_night[0]))
    # index1 = np.argsort(monthly_avg_night[0])
    # index2 = np.argsort(monthly_avg_night[1])
    # index3 = np.argsort(monthly_avg_night[2])
    # index4 = np.argsort(monthly_avg_night[3])
    # monthly_avg_night[0] = monthly_avg_night[0][index1]
    # monthly_std_night[0] = monthly_std_night[0][index1]
    # monthly_avg_night[1] = monthly_avg_night[1][index2]
    # monthly_std_night[1] = monthly_std_night[1][index2]
    # monthly_avg_night[2] = monthly_avg_night[2][index3]
    # monthly_std_night[2] = monthly_std_night[2][index3]
    # monthly_avg_night[3] = monthly_avg_night[3][index4]
    # monthly_std_night[3] = monthly_std_night[3][index4]


    # # Plot of node vs total outgoing volume
    # plt.plot(range(1237), monthly_avg_morn[0][0:1237], '-', label='morn', linewidth=0.1)
    # plt.plot(range(1237), monthly_avg_aft[0][0:1237], '-', label='aft', linewidth=0.2)
    # plt.plot(range(1237), monthly_avg_eve[0][0:1237], '-', label='eve', linewidth=0.5)
    # plt.plot(range(1237), monthly_avg_night[0][0:1237], '-', label='night', linewidth=1)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Outgoing Call Volume", fontsize=16)
    # plt.title("Outgoing call volume for each node - 5 months", fontsize=16)
    # plt.legend()
    # plt.grid()
    # plt.show()
    #
    # plt.figure(1, facecolor='white')
    #
    # plt.subplot(2,2,1)
    # plt.errorbar(range(1237), monthly_avg_morn[0][0:1237], yerr=monthly_std_morn[0][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlim(0, 1237)
    # plt.ylim(-2000, 12000)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Outgoing Call Volume", fontsize=16)
    # plt.title("Outgoing call volume: Morning", fontsize=16)
    # plt.grid()
    #
    # plt.subplot(2,2,2)
    # plt.errorbar(range(1237), monthly_avg_aft[0][0:1237], yerr=monthly_std_aft[0][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Outgoing Call Volume", fontsize=16)
    # plt.title("Outgoing call volume: Afternoon", fontsize=16)
    # plt.xlim(0, 1237)
    # plt.ylim(-2000, 12000)
    # plt.grid()
    #
    # plt.subplot(2,2,3)
    # plt.errorbar(range(1237), monthly_avg_eve[0][0:1237], yerr=monthly_std_eve[0][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Outgoing Call Volume", fontsize=16)
    # plt.title("Outgoing call volume: Evening", fontsize=16)
    # plt.xlim(0, 1237)
    # plt.ylim(-2000, 12000)
    # plt.grid()
    #
    # plt.subplot(2,2,4)
    # plt.errorbar(range(1237), monthly_avg_night[0][0:1237], yerr=monthly_std_night[0][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Volume of Outgoing Calls", fontsize=16)
    # plt.title("Outgoing Call Volume: Night", fontsize=16)
    # plt.xlim(0, 1237)
    # plt.ylim(-2000, 12000)
    # plt.grid()
    #
    # plt.show()
    #
    #
    #
    #
    #
    #
    # ''' Plot 2 '''
    # plt.figure(1, facecolor='white')
    #
    # plt.subplot(2,2,1)
    # plt.errorbar(range(1237), monthly_avg_morn[1][0:1237], yerr=monthly_std_morn[1][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlim(0, 1237)
    # plt.ylim(-2000, 12000)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Incoming Call Volume", fontsize=16)
    # plt.title("Incoming call volume: Morning", fontsize=16)
    # plt.grid()
    #
    # plt.subplot(2,2,2)
    # plt.errorbar(range(1237), monthly_avg_aft[1][0:1237], yerr=monthly_std_aft[1][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Incoming Call Volume", fontsize=16)
    # plt.title("Incoming call volume: Afternoon", fontsize=16)
    # plt.xlim(0, 1237)
    # plt.ylim(-2000, 12000)
    # plt.grid()
    #
    # plt.subplot(2,2,3)
    # plt.errorbar(range(1237), monthly_avg_eve[1][0:1237], yerr=monthly_std_eve[1][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Incoming Call Volume", fontsize=16)
    # plt.title("Incoming call volume: Evening", fontsize=16)
    # plt.xlim(0, 1237)
    # plt.ylim(-2000, 12000)
    # plt.grid()
    #
    # plt.subplot(2,2,4)
    # plt.errorbar(range(1237), monthly_avg_night[1][0:1237], yerr=monthly_std_night[1][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Incoming Call Volume", fontsize=16)
    # plt.title("Incoming Call Volume: Night", fontsize=16)
    # plt.xlim(0, 1237)
    # plt.ylim(-2000, 12000)
    # plt.grid()
    #
    # plt.show()

    # ''' Plot 3 '''
    #
    # plt.figure(1, facecolor='white')
    #
    # plt.subplot(2,2,1)
    # plt.errorbar(range(1237), monthly_avg_morn[2][0:1237], yerr=monthly_std_morn[2][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlim(0, 1237)
    # plt.ylim(-300000, 2000000)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Outgoing Call Duration (seconds)", fontsize=16)
    # plt.title("Outgoing Call Duration: Morning", fontsize=16)
    # plt.grid()
    #
    # plt.subplot(2,2,2)
    # plt.errorbar(range(1237), monthly_avg_aft[2][0:1237], yerr=monthly_std_aft[2][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Outgoing Call Duration (seconds)", fontsize=16)
    # plt.title("Outgoing Call Duration: Afternoon", fontsize=16)
    # plt.xlim(0, 1237)
    # plt.ylim(-300000, 2000000)
    # plt.grid()
    #
    # plt.subplot(2,2,3)
    # plt.errorbar(range(1237), monthly_avg_eve[2][0:1237], yerr=monthly_std_eve[2][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Outgoing Call Duration (seconds)", fontsize=16)
    # plt.title("Outgoing Call Duration: Evening", fontsize=16)
    # plt.xlim(0, 1237)
    # plt.ylim(-300000, 2000000)
    # plt.grid()
    #
    # plt.subplot(2,2,4)
    # plt.errorbar(range(1237), monthly_avg_night[2][0:1237], yerr=monthly_std_night[2][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Outgoing Call Duration (seconds)", fontsize=16)
    # plt.title("Outgoing Call Duration: Night", fontsize=16)
    # plt.xlim(0, 1237)
    # plt.ylim(-300000, 2000000)
    # plt.grid()
    #
    # plt.show()
    #
    # ''' Plot 4 '''
    #
    # plt.figure(1, facecolor='white')
    #
    # plt.subplot(2,2,1)
    # plt.errorbar(range(1237), monthly_avg_morn[3][0:1237], yerr=monthly_std_morn[3][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlim(0, 1237)
    # plt.ylim(-300000, 2000000)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Incoming Call Duration", fontsize=16)
    # plt.title("Incoming Call Duration: Morning", fontsize=16)
    # plt.grid()
    #
    # plt.subplot(2,2,2)
    # plt.errorbar(range(1237), monthly_avg_aft[3][0:1237], yerr=monthly_std_aft[3][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Incoming Call Duration", fontsize=16)
    # plt.title("Incoming Call Duration: Afternoon", fontsize=16)
    # plt.xlim(0, 1237)
    # plt.ylim(-300000, 2000000)
    # plt.grid()
    #
    # plt.subplot(2,2,3)
    # plt.errorbar(range(1237), monthly_avg_eve[3][0:1237], yerr=monthly_std_eve[3][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Incoming Call Duration", fontsize=16)
    # plt.title("Incoming Call Duration: Evening", fontsize=16)
    # plt.xlim(0, 1237)
    # plt.ylim(-300000, 2000000)
    # plt.grid()
    #
    # plt.subplot(2,2,4)
    # plt.errorbar(range(1237), monthly_avg_night[3][0:1237], yerr=monthly_std_night[3][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Volume of Outgoing Calls", fontsize=16)
    # plt.title("Outgoing Call Volume: Night", fontsize=16)
    # plt.xlim(0, 1237)
    # plt.ylim(-300000, 2000000)
    # plt.grid()
    #
    # plt.show()



    #
    # # Plot of node vs total incoming volume
    # plt.plot(range(1237), monthly_avg_morn[1][0:1237], '-', label='morn')
    # plt.plot(range(1237), monthly_avg_aft[1][0:1237], '-', label='aft')
    # plt.plot(range(1237), monthly_avg_eve[1][0:1237], '-', label='eve')
    # plt.plot(range(1237), monthly_avg_night[1][0:1237], '-', label='night')
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Volume of Incoming Calls", fontsize=16)
    # plt.title("Incoming call volume for each node - 5 months", fontsize=16)
    # plt.legend()
    # plt.grid()
    # plt.show()
    #
    # # Plot of node vs total outgoing duration
    # plt.plot(range(1237), monthly_avg_morn[2][0:1237], '-', label='morn')
    # plt.plot(range(1237), monthly_avg_aft[2][0:1237], '-', label='aft')
    # plt.plot(range(1237), monthly_avg_eve[2][0:1237], '-', label='eve')
    # plt.plot(range(1237), monthly_avg_night[2][0:1237], '-', label='night')
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Duration of Outgoing Calls (seconds)", fontsize=16)
    # plt.title("Outgoing call duration for each node - 5 months", fontsize=16)
    # plt.legend()
    # plt.grid()
    # plt.show()
    #
    # # Plot of node vs total incoming duration
    # plt.plot(range(1237), monthly_avg_morn[3][0:1237], '-', label='morn')
    # plt.plot(range(1237), monthly_avg_aft[3][0:1237], '-', label='aft')
    # plt.plot(range(1237), monthly_avg_eve[3][0:1237], '-', label='eve')
    # plt.plot(range(1237), monthly_avg_night[3][0:1237], '-', label='night')
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Duration of Incoming Calls (seconds)", fontsize=16)
    # plt.title("Incoming call duration for each node - 5 months", fontsize=16)
    # plt.legend()
    # plt.grid()
    # plt.show()


    #
    # plt.errorbar(range(1237), monthly_avg_night[0][0:1237], yerr=monthly_std_night[0][0:1237], fmt='--')
    # plt.grid()
    # plt.show()