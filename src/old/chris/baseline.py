from __future__ import division

import sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

import config
from modelling import *



def go(loc_tag):
    print "location:", loc_tag
    conf = config.get(loc_tag)
    dhs = pd.read_csv(conf['dhs_fn'])
    dhs.set_index('clust_id', inplace=True)


    ####################
    #   RANDOM
    ####################
    # regression
    # ----------
    # wealth
    # approximate bimodal normal distribution
    # (params chosen by visually comparing density
    #  plots to ground truth)
    print "random baselines"
    m = 10000
    n = len(dhs)
    print "wealth"
    if loc_tag == 'sen':
        x = np.concatenate((np.random.normal(-.9, .3, int(m/2)),
                            np.random.normal(1., .5, int(m/2))))
    else:
        x = np.concatenate((np.random.normal(-1., .4, int(m/2)),
                            np.random.normal(.65, .7, int(m/2))))

    y = dhs.z_median.values
    random_mae = np.array(
        [np.mean(np.abs(y - x[np.random.randint(0,m,n)]))
            for i in range(1000)])
    print "mean mae: ", random_mae.mean()
    random_spearman = np.array([spearmanr(x[np.random.randint(0,m,n)], y)
            for i in range(1000)])
    print "mean spearmanr: ", random_spearman.mean()

    print "poverty intensity"
    # poverty intensity
    # 'zero-inflated uniform'
    p = .634 if loc_tag == 'sen' else .496 # proportion of non zero in dataset
    x = np.random.binomial(1, p, m).astype(np.float)
    x[x==1] = np.random.random(x.sum())
    y = dhs.poverty_rate.values
    random_mae = np.array(
        [np.mean(np.abs(y - x[np.random.randint(0,m,n)]))
            for i in range(1000)])
    print "mean mae: ", random_mae.mean()
    random_spearman = np.array([spearmanr(x[np.random.randint(0,m,n)], y)
            for i in range(1000)])
    print "mean spearmanr: ", random_spearman.mean()

    # # classification
    # # --------------
    # all one predictor
    # y = (dhs.poverty_rate >= .5).astype(np.int)
    # p = np.mean(y)
    # n = len(y)
    # prec = np.zeros(10000)
    # sens = np.zeros(10000)
    # accr = np.zeros(10000)
    # for i in range(10000):
    #     y_hat = np.random.binomial(1, p, n)
    #     tp = np.sum(y[y_hat==1])
    #     fp = np.sum(y_hat==1) - tp
    #     prec[i] = tp / (tp + fp)
    #     sens[i] = tp / np.sum(y)
    #     accr[i] = np.mean((y_hat==1) == y)
    # print "precision: [mean: %.3f] [std: %.3f]" % (prec.mean(), prec.std())
    # print "sensitivity: [mean: %.3f] [std: %.3f]" % (sens.mean(), sens.std())
    # print "accuracy: [mean: %.3f] [std: %.3f]" % (accr.mean(), accr.std())


    ###################
    #      PD
    ###################
    print "population density baselines"
    X = sm.add_constant(dhs['pop_1km'])
    X['pop_1km'] = np.log(X['pop_1km'])
    # regression
    # ----------
    print "wealth"
    y = dhs['z_median']
    coefs, pvalues, scores, errors = fit_all(X, y, conf, 'wealth', 'pd')
    print "mean:", scores.groupby('p')['test_rmse'].mean()

    print "poverty intensity"
    y = dhs['poverty_rate']
    coefs, pvalues, scores, errors = fit_all(X, y, conf, 'poverty', 'pd')
    print "mean:", scores.groupby('p')['test_rmse'].mean()

    # classification
    # --------------
    # y = (dhs['poverty_rate'] >= .25).astype(np.int)
    # coefs, pvalues, scores, errors = fit_all(X, y, conf, 'clf', 'pd')
    # scores[['p','train_sens','test_sens','train_spec','test_spec',
    #         'train_prec','test_prec','train_accr','test_accr']].groupby('p').mean()

    ####################
    #    LAG
    ####################
    print "spatial lag baselines"
    X = pd.DataFrame({'const': np.ones(len(dhs))}, index=dhs.index)
    # regression
    # ----------
    print "wealth"
    y = dhs['z_median']
    coefs, pvalues, scores, errors = fit_all(X, y, conf, 'wealth', 'lag', include_lags=True)
    print "mean:", scores.groupby('p')['test_rmse'].mean()

    print "poverty intensity"
    y = dhs['poverty_rate']
    coefs, pvalues, scores, errors = fit_all(X, y, conf, 'poverty', 'lag', include_lags=True)
    print "mean:", scores.groupby('p')['test_rmse'].mean()

    # classification
    # --------------
    # y = (dhs['poverty_rate'] >= .25).astype(np.int)
    # coefs, pvalues, scores, errors = fit_all(X, y, conf, 'clf', 'lag', include_lags=True)


    ####################
    #   LAG + PD
    ####################
    print "population density + spatial lag baselines"
    X = sm.add_constant(dhs['pop_1km'])
    X['pop_1km'] = np.log(X['pop_1km'])
    # regression
    # ----------
    print "wealth"
    y = dhs['z_median']
    coefs, pvalues, scores, errors = fit_all(X, y, conf, 'wealth', 'pdlag', include_lags=True)
    print "mean:", scores.groupby('p')['test_rmse'].mean()

    print "poverty intensity"
    y = dhs['poverty_rate']
    coefs, pvalues, scores, errors = fit_all(X, y, conf, 'poverty', 'pdlag', include_lags=True)
    print "mean:", scores.groupby('p')['test_rmse'].mean()

    # classification
    # --------------
    # y = (dhs['poverty_rate'] >= .25).astype(np.int)
    # coefs, pvalues, scores, errors = fit_all(X, y, conf, 'clf', 'pdlag', include_lags=True)

    print "fin."

if __name__ == "__main__":
    loc_tag = sys.argv[1]
    go(loc_tag)

#
