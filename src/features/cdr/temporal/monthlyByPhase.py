# Jack Shipway, October 2016, UCL GCRF Research Project
#
# Run once to obtain

import numpy as np

MONTHS = ['dec', 'jan', 'feb', 'mar', 'apr']
DAYS = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']

total_morn = np.load("IvoryCoastData/CDR/AggregateData/total_morn.npy")
total_aft = np.load("IvoryCoastData/CDR/AggregateData/total_aft.npy")
total_eve = np.load("IvoryCoastData/CDR/AggregateData/total_eve.npy")
total_night = np.load("IvoryCoastData/CDR/AggregateData/total_night.npy")
total_working = np.load("IvoryCoastData/CDR/AggregateData/total_working.npy")
total_nworking = np.load("IvoryCoastData/CDR/AggregateData/total_nworking.npy")

start, end = 0, 28
for month in MONTHS:
    month_morn = total_morn[:, :, start:end]
    np.save("IvoryCoastData/CDR/AggregateData/%s_morn.npy" % month, month_morn)
    month_aft = total_aft[:, :, start:end]
    np.save("IvoryCoastData/CDR/AggregateData/%s_aft.npy" % month, month_aft)
    month_eve = total_eve[:, :, start:end]
    np.save("IvoryCoastData/CDR/AggregateData/%s_eve.npy" % month, month_eve)
    month_night = total_night[:, :, start:end]
    np.save("IvoryCoastData/CDR/AggregateData/%s_night.npy" % month, month_night)
    month_working = total_working[:, :, start:end]
    np.save("IvoryCoastData/CDR/AggregateData/%s_working.npy" % month, month_working)
    month_nworking = total_nworking[:, :, start:end]
    np.save("IvoryCoastData/CDR/AggregateData/%s_nworking.npy" % month, month_nworking)
    start = end
    end += 28

i = 0
for day in DAYS:
    day_morn = total_morn[:, :, i::7]
    np.save("IvoryCoastData/CDR/AggregateData/%s_morn.npy" % day, day_morn)
    day_aft = total_aft[:, :, i::7]
    np.save("IvoryCoastData/CDR/AggregateData/%s_aft.npy" % day, day_aft)
    day_eve = total_eve[:, :, i::7]
    np.save("IvoryCoastData/CDR/AggregateData/%s_eve.npy" % day, day_eve)
    day_night = total_night[:, :, i::7]
    np.save("IvoryCoastData/CDR/AggregateData/%s_night.npy" % day, day_night)
    day_working = total_working[:, :, i::7]
    np.save("IvoryCoastData/CDR/AggregateData/%s_working.npy" % day, day_working)
    day_nworking = total_nworking[:, :, i::7]
    np.save("IvoryCoastData/CDR/AggregateData/%s_nworking.npy" % day, day_nworking)
    i += 1
