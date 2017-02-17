from __future__ import division
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# from radiation import *
import config


def get_model_data(flows):
    mask = np.logical_and(flows['d'].values > 0., flows['vol'].values > 0.)
    X = np.log(flows[mask][['m', 'n', 'd']].values)
    y = np.log(flows[mask]['vol'].values)
    return X, y, mask

def predict_flows(lm, flows):
    beta = np.concatenate(([np.exp(lm.intercept_)], lm.coef_))
    yhat = beta[0] * flows['m']**beta[1] * flows['n']**beta[2] * flows['d']**beta[3]
    return yhat, beta

def fit_split_model(r, X, y):
    lo_ix = X[:,2] <= r
    hi_ix = np.logical_not(lo_ix)
    if lo_ix.sum() > 10 and hi_ix.sum() > 10:
        X_lo, y_lo = X[lo_ix], y[lo_ix]
        X_hi, y_hi = X[hi_ix], y[hi_ix]
        lm_lo, lm_hi = LinearRegression(), LinearRegression()
        lm_lo.fit(X_lo, y_lo)
        lm_hi.fit(X_hi, y_hi)
        _y = np.concatenate((y[lo_ix], y[hi_ix]))
        yhat = np.concatenate((lm_lo.predict(X_lo), lm_hi.predict(X_hi)))
        mse = mean_squared_error(_y, yhat)
        r2 = r2_score(_y, yhat)
        return lm_lo, lm_hi, mse, r2
    return None, None, np.inf

def model_split_flows(X, y):
    qs = np.arange(1,99)
    rs = np.percentile(X[:,2], qs)
    mse = np.array([fit_split_model(r, X, y)[2] for r in rs])
    r = rs[np.where(mse.min()==mse)[0]][0]
    lm_lo, lm_hi, _, r2 = fit_split_model(r, X, y)
    return lm_lo, lm_hi, np.exp(r), r2

def predict_split_flows(lm_lo, lm_hi, flows, split):
    beta_lo = np.concatenate(([np.exp(lm_lo.intercept_)], lm_lo.coef_))
    beta_hi = np.concatenate(([np.exp(lm_hi.intercept_)], lm_hi.coef_))
    yhat_lohi = beta_lo[0] * flows['m']**beta_lo[1] * flows['n']**beta_lo[2] * flows['d']**beta_lo[3]
    yhat_lohi[flows['d']>split] = beta_hi[0] * flows.loc[flows['d']>split,'m']**beta_hi[1] * flows.loc[flows['d']>split,'n']**beta_hi[2] * flows.loc[flows['d']>split,'d']**beta_hi[3]
    return yhat_lohi, beta_lo, beta_hi

def run(points_fn, vol_fn, rad_fn, flows_fn, group='all'):
    print "reading data..."
    points = pd.read_csv(points_fn)
    vol = np.load(vol_fn)
    flows = pd.read_csv(rad_fn)
    # pids = np.unique(np.concatenate((flows['a'].values, flows['b'].values)))
    # points = points[[pid in pids for pid in points['id']]]

    # if not ix_col:
    #     ix = points.id + conf['offset']
    # else:
    ix = points['vol_ix'].values
    flows['vol'] = (vol[ix][:,ix]).flatten()
    internal_flows = flows[flows['a'] == flows['b']].copy()
    flows = flows[flows['a'] != flows['b']]

    print "scaling radiation estimates by total out volume..."
    # (should really be done in radiation.py)
    # total number of calls over total population (N_c/N)
    pop_scale = flows['vol'].sum() / flows[['a','m']].drop_duplicates()['m'].sum()
    # T_i = m_i * (N_c/N)
    total_out_flow = flows[['a','vol']].groupby('a').sum()
    flows.set_index('a', inplace=True)
    flows['rad'] = flows['r'] * flows['m'] * pop_scale # should join by index
    flows.drop('r', axis=1, inplace=True)
    flows.reset_index(inplace=True)

    print "gravity modelling..."
    # population is d^2 weighted mean of raster
    X, y, mask = get_model_data(flows)
    lm = LinearRegression()
    lm.fit(X, y)
    yhat, beta = predict_flows(lm, flows)
    with open(conf['lm_params_fn'](clust_type, size), 'a') as f:
        f.write("\n%s - %s\n" % (loc_tag.upper(), group))
        f.write("===\n")
        f.write("lm beta: %s\n" % beta)
        f.write("lm R^2: %f\n" % lm.score(X, y))

    lm_lo, lm_hi, split, r2 = model_split_flows(X, y)
    yhat_lohi, beta_lo, beta_hi = predict_split_flows(lm_lo, lm_hi, flows, split)
    with open(conf['lm_params_fn'](clust_type, size), 'a') as f:
        f.write('----------------\n')
        f.write("split lm lo beta: %s\n" % beta_lo)
        f.write("split lm hi beta: %s\n" % beta_hi)
        f.write("split lm R^2: %f\n" % r2)
        f.write("split point 2: %.2f\n" % split)

    flows['grav'] = yhat
    flows['grav_split'] = yhat_lohi

    cols = flows.columns.tolist()
    # append internal flows (they will have no value for radiation and gravity flows)
    flows = pd.concat((flows, internal_flows))[cols]
    flows.to_csv(flows_fn, index=False, encoding='utf8')



if __name__ == '__main__':
    loc_tag = sys.argv[1]
    clust_type = sys.argv[2]
    size = int(sys.argv[3])
    conf = config.get(loc_tag)

    # ix_col = None
    # if len(sys.argv) > 6:
    #     ix_col = sys.argv[6]

    print "all points..."
    run(conf['centroids_fn'](clust_type, size),
        conf['vol_fn'](clust_type, size),
        conf['rad_fn'](clust_type, size),
        conf['flows_fn'](clust_type, size)
        )
    print "\nnoncapital points..."
    run(conf['noncapital_centroids_fn'](clust_type, size),
        conf['noncapital_vol_fn'](clust_type, size),
        conf['noncapital_rad_fn'](clust_type, size),
        conf['noncapital_flows_fn'](clust_type, size)
        )
    print "\ncapital points..."
    run(conf['capital_centroids_fn'](clust_type, size),
        conf['capital_vol_fn'](clust_type, size),
        conf['capital_rad_fn'](clust_type, size),
        conf['capital_flows_fn'](clust_type, size)
        )

    print "\nfin."






    # --------
    # OLD CODE
    # --------
    # total_out_flow.columns = ['total_out_vol','r1_sum','r2_sum']
    # total_out_flow['T'] = total_out_flow['total_out_vol'] * pop_scale
    # flows = flows.join(total_out_flow, on='a')

    # flows['rt1'] = flows['r'] * flows['T']
    # flows['rt2'] = flows['r2'] * flows['T']
    # flows['rv1'] = flows['r1'] * flows['total_out_vol']
    # flows['rv2'] = flows['r2'] * flows['total_out_vol']
    # flows['rsv1'] = flows['rv1'] / flows['r1_sum']
    # flows['rsv2'] = flows['rv2'] / flows['r2_sum']
    # population is d weighted mean of raster
    # X, y, mask = get_model_data(flows, 'm', 'n1', 'vol')
    # lm = LinearRegression()
    # lm.fit(X, y)
    # yhat, beta = predict_flows(lm, flows, 'm1', 'n1')
    # lm_lo, lm_hi, r = model_split_flows(X, y)
    # yhat_lohi, beta_lo, beta_hi = predict_split_flows(lm_lo, lm_hi, flows, 'm1', 'n1')

    # flows['g_vol1'] = yhat
    # flows['gr_vol1'] = yhat_lohi
    # print "split point 1: %.2f" % r








#
