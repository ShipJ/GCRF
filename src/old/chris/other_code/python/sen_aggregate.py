"""
aggregate D4D data to specified time bins
"""
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import arrow
import gzip
import sys


if __name__ == "__main__":
    vol = np.zeros((1666, 1666, 365), dtype=np.int64)
    dur = np.zeros((1666, 1666, 365), dtype=np.int64)
    # sms = np.zeros((1666, 1666, 365), dtype=np.int64)

    day_one = arrow.Arrow(2013, 1, 1)

    for fn in ["cdr_data/sen/SET1/SET1V_%02d.CSV.gz" % i for i in range(1, 13)]:
        print 'opening %s' % fn
        f_in = gzip.open(fn, 'rb')
        n = 0
        for line in f_in:
            row = line.split(',')
            dt = arrow.get(row[0], 'YYYY-MM-DD HH')
            # sum day time only 7am-7pm
            if dt.time().hour < 7 or dt.time().hour >= 19:
                continue
            day = (dt - day_one).days
            bts_a, bts_b = int(row[1]), int(row[2])
            vol[bts_a-1, bts_b-1, day] += int(row[3])
            dur[bts_a-1, bts_b-1, day] += int(row[4])

            n += 1
            if n % 100000 == 0:
                print '%d lines parsed' % n
        f_in.close()
    np.save('cdr_data/sen/flows/vol_1day_7-19.npy', vol)
    np.save('cdr_data/sen/flows/dur_1day_7-19.npy', dur)


def total_events(type):
    f = gzip.open('cdr_data/sen/flows/%s_1day_bins.npy.gz' % type)
    W = np.load(f)
    f.close()
    X = W.sum(axis=2)
    X_var = W.var(axis=2) # temporal variance
    np.save('cdr_data/sen/flows/total_%s.npy' % type, X)
    np.save('cdr_data/sen/flows/total_%s_day_variance.npy' % type, X_var)
    return X, X_var
