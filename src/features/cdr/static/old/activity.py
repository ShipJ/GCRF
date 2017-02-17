# Jack Shipway, 3/11/16, UCL GCRF Project
#
# Total activity and duration for each cell tower (incoming + outgoing calls)
#
# Input: hourly time-stamped csv files of CDR data
# Output: a (num_towers) x 1 array containing the total activity and duration of each cell tower

import pandas as pd
import numpy as np
import math

if __name__ == "__main__":

    # country = sys.argv[1]
    country = 'Senegal'

    path = '/Users/JackShipway/Desktop/UCLProject/Data/%s/CDR'

    # Set known data set values (number of towers, and time length of data)
    if country == 'Senegal':
        num_bts, hours = 1668, 8760
    elif country == 'IvoryCoast':
        num_bts, hours = 1240, 3360
    else:
        num_bts, hours = 10000, 100000

    # Starting time-step (month, week, day, hour)
    m, w, d, h = 0, 0, 0, 0

    # Metrics to derive
    volume_total, volume_in, volume_out = np.zeros(num_bts), np.zeros(num_bts), np.zeros(num_bts)
    duration_total, duration_in, duration_out = np.zeros(num_bts), np.zeros(num_bts), np.zeros(num_bts)

    # Sum output for each cell tower, for each hour of data
    for hour in range(hours):
        print 'Reading Hour: %s' % hour, m, w, d, h

        # Read the temporal data sets in turn, convert to multi-dimensional array
        data = pd.read_csv(path+'/Data/Month_%s/Week_%s/Day_%s/Hour_%s/graph.csv' % (m, w, d, h),
                           usecols=['source', 'target', 'volume', 'duration']).as_matrix()

        # Only update cell towers active within that time-step
        active_towers = np.array(np.unique(np.concatenate([data[:, 0], data[:, 1]])))

        for cell_tower in active_towers:
            active_data = data[data[:, 0] == cell_tower]

            in_vol = np.sum(active_data[:, 2])
            out_vol = np.sum(data[data[:, 1] == cell_tower][:, 2])

            in_dur = np.sum(active_data[:, 3])
            out_dur = np.sum(data[data[:, 1] == cell_tower][:, 3])

            # Taking into account (i, i) counted twice, so it must be subtracted once
            if active_data.size == 0:
                self_vol, self_dur = 0, 0
            else:
                self_active = active_data[active_data[:, 1] == cell_tower]
                if self_active.size != 0:
                    self_vol, self_dur = self_active[0][2], self_active[0][3]
                else:
                    self_vol, self_dur = 0, 0

            # Increment activity for respective cell towers
            volume_total[cell_tower] += (in_vol + out_vol) - self_vol
            volume_in[cell_tower] += in_vol
            volume_out[cell_tower] += out_vol
            duration_total[cell_tower] += (in_dur + out_dur) - self_dur
            duration_in[cell_tower] += in_dur
            duration_out[cell_tower] += out_dur

        # Increment CDR directory
        h = int(math.fmod(h + 1, 24))
        if h == 0:
            d = int(math.fmod(d + 1, 7))
            if d == 0:
                w = int(math.fmod(w + 1, 4))
                if w == 0:
                    m += 1

    total_activity = pd.DataFrame()
    total_activity['ID'] = np.array(range(num_bts))
    total_activity['Vol'] = volume_total
    total_activity['Vol_in'] = volume_in
    total_activity['Vol_out'] = volume_out
    total_activity['Dur'] = duration_total
    total_activity['Dur_in'] = duration_in
    total_activity['Dur_out'] = duration_out

    # # Save results
    # total_activity.to_csv(path+'staticmetrics/Activity/total_activity.csv', index=None)
