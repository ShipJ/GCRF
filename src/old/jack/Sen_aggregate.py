# Import Senegal Set 1 data, store call volume and duration
# between antennae in two numpy arrays

from __future__ import division
import numpy as np
import arrow
import gzip

if __name__ == "__main__":
    # Numpy arrays of #antennae x #antennae x #days
    vol = np.zeros((1666, 1666, 365), dtype=np.int64)
    dur = np.zeros((1666, 1666, 365), dtype=np.int64)
    # This data set also has volume of SMS messages
    sms = np.zeros((1666, 1666, 365), dtype=np.int64)

    # Start date: 1st Jan 2013
    day_one = arrow.Arrow(2013, 1, 1)

    for fn in ["Desktop/Code/Python/SenegalData//SET1V_%02d.CSV.gz" % i for i in range(1, 13)]:
        print 'opening %s' % fn
        f_in = gzip.open(fn, 'rb')
        n = 0
        for line in f_in:
            row = line.split(',')
            dt = arrow.get(row[0], 'YYYY-MM-DD HH')
            # Exclude connections before 7am, after 7pm
            if dt.time().hour < 7 or dt.time().hour >= 19:
                continue
            day = (dt - day_one).days
            bts_a, bts_b = int(row[1]), int(row[2])
            vol[bts_a-1, bts_b-1, day] += int(row[3])
            dur[bts_a-1, bts_b-1, day] += int(row[4])
            # Completion Incrementer
            n += 1
            if n % 100000 == 0:
                print '%d lines parsed' % n
        f_in.close()
    # Save call volume/duration
    np.save('Desktop/Code/Python/ConfigResults/sen/cdr_data/flows/vol_1day_7-19.npy', vol)
    np.save('Desktop/Code/Python/ConfigResults/sen/cdr_data/flows/dur_1day_7-19.npy', dur)


def total_events(type):
    f = gzip.open('cdr_data/sen/flows/%s_1day_bins.npy.gz' % type)
    W = np.load(f)
    f.close()
    X = W.sum(axis=2)
    X_var = W.var(axis=2) # temporal variance
    np.save('cdr_data/sen/flows/total_%s.npy' % type, X)
    np.save('cdr_data/sen/flows/total_%s_day_variance.npy' % type, X_var)
    return X, X_var
