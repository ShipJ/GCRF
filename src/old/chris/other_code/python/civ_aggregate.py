import numpy as np
import arrow
import gzip
import sys


# vol = np.zeros((1239, 1239), dtype=np.int64)
# dur = np.zeros((1239, 1239), dtype=np.int64)

# for i in range(10):
#     fn = 'cdr_data/civ/SET1/SET1TSV_%d.TSV' % i
#     print 'opening', fn
#     f = open(fn)
#     i = 0
#     for line in f:
#         row = line.split('\t')
#         a, b, v, d = [int(row[k]) for k in [1,2,3,4]]
#         vol[a+1, b+1] += v
#         dur[a+1, b+1] += d
#         i += 1
#         if i % 100000 == 0:
#             print 'read %d lines' % i
#     f.close()

if __name__ == "__main__":
    vol = np.zeros((1239, 1239, 140), dtype=np.int64)
    dur = np.zeros((1239, 1239, 140), dtype=np.int64)
    day_one = arrow.Arrow(2011, 12, 05)

    for fn in ["cdr_data/civ/SET1/SET1TSV_%d.TSV" % i for i in range(10)]:
        print 'opening %s' % fn
        f_in = open(fn, 'rb')
        n = 0
        for line in f_in:
            row = line.split('\t')
            dt = arrow.get(row[0])
            if dt.time().hour < 7 or dt.time().hour >= 19:
                continue
            day = (dt - day_one).days
            bts_a, bts_b = int(row[1]), int(row[2])
            vol[bts_a+1, bts_b+1, day] += int(row[3])
            dur[bts_a+1, bts_b+1, day] += int(row[4])

            n += 1
            if n % 100000 == 0:
                print '%d lines parsed' % n
        f_in.close()

    print 'saving data...'
    np.save('cdr_data/civ/SET1/vol_1day_7-19.npy', vol)
    np.save('cdr_data/civ/SET1/dur_1day.7-19.npy', dur)

    print 'fin.'
