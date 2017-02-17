from __future__ import division

import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from numpy.linalg.linalg import LinAlgError

import config
from latex import latex_table

def rgr(X_train, X_test, y_train, y_test):
    fit = sm.OLS(y_train, X_train).fit()
    y_pred = fit.predict(X_test)
    scores = {
        'aic': fit.aic,
        'bic': fit.bic,
        'r_sqrd': fit.rsquared,
        'r_sqrd_adj': fit.rsquared_adj,
        'train_rmse': np.sqrt(np.mean((y_train - fit.fittedvalues)**2)),
        'test_rmse': np.sqrt(np.mean((y_test - y_pred)**2)),
        'train_mad': np.median(np.abs(y_train - fit.fittedvalues)),
        'test_mad': np.median(np.abs(y_test - y_pred))
    }
    return fit.params, fit.pvalues, y_pred, scores

def negbin(X_train, X_test, y_train, y_test, offset_train, offset_test):
    fit = sm.GLM(y_train, X_train, offset=offset_train, family=sm.families.NegativeBinomial()).fit()
    y_pred = fit.predict(X_test, offset=offset_test)
    scores = {
        'aic': fit.aic,
        'bic': fit.bic,
        'train_rmse': np.sqrt(np.mean((y_train - fit.fittedvalues)**2)),
        'test_rmse': np.sqrt(np.mean((y_test - y_pred)**2)),
        'train_mad': np.median(np.abs(y_train - fit.fittedvalues)),
        'test_mad': np.median(np.abs(y_test - y_pred))
    }
    return fit.params, fit.pvalues, y_pred, scores


def clf(X_train, X_test, y_train, y_test):
    fit = sm.GLM(y_train, X_train, family=sm.families.Binomial()).fit()
    y_pred = fit.predict(X_test)
    train_tp = np.sum(y_train[fit.fittedvalues>.5])
    test_tp = np.sum(y_test[y_pred>.5])
    scores = {
        'aic': fit.aic,
        'bic': fit.bic,
        'train_prec': train_tp / np.sum(fit.fittedvalues>.5),
        'train_sens': train_tp / np.sum(y_train),
        'train_accr': np.mean((fit.fittedvalues>.5) == y_train),
        'train_spec': np.sum(y_train[fit.fittedvalues<=.5]==0) / np.sum(y_train==0),
        'test_prec': test_tp / np.sum(y_pred>.5),
        'test_sens': test_tp / np.sum(y_test),
        'test_accr': np.mean((y_pred>.5) == y_test),
        'test_spec': np.sum(y_test[y_pred<=.5]==0) / np.sum(y_test==0),
    }
    return fit.params, fit.pvalues, y_pred, scores

def iter_fit(X, y, p, train_ix, test_ix, model_func, lags=None):
    assert X.shape[0] == y.shape[0]
    n_iters = train_ix.shape[0]
    assert test_ix.shape[0] == n_iters
    if lags is not None:
        assert lags.shape[0] == X.shape[0]
        assert lags.shape[1] == n_iters

    n_predictors = X.shape[1] - int('offset' in X.columns)
    Y_pred = np.zeros(test_ix.shape)
    coefs = np.zeros((n_iters, n_predictors))
    pvalues = np.zeros((n_iters, n_predictors))
    scores = []
    errors = {}

    for i in range(n_iters):
        if lags is not None:
            lag = lags[i]
            assert lag.shape[0] == X.shape[0]
            X['lag'] = lag

        if 'offset' in X.columns:
            offset = X['offset']
            offset_train = offset[train_ix[i]].values
            offset_test = offset[test_ix[i]].values
            X_train = X.drop('offset', axis=1).ix[train_ix[i]].values
            X_test = X.drop('offset', axis=1).ix[test_ix[i]].values
        else:
            X_train = X.ix[train_ix[i]].values
            X_test = X.ix[test_ix[i]].values

        y_train = y.ix[train_ix[i]].values
        y_test = y.ix[test_ix[i]].values

        try:
            if 'offset' in X.columns:
                cfs, pvs, y_hat, scrs = model_func(X_train, X_test,
                    y_train, y_test, offset_train, offset_test)
            else:
                cfs, pvs, y_hat, scrs = model_func(
                    X_train, X_test, y_train, y_test)
            Y_pred[i] = y_hat
            coefs[i] = cfs
            pvalues[i] = pvs
            scores.append(scrs)
        except PerfectSeparationError:
            if errors.get('perfect_sep') is None:
                errors['perfect_sep'] = [i]
            else:
                errors['perfect_sep'].append(i)
        except LinAlgError:
            if errors.get('linalg') is None:
                errors['linalg'] = [i]
            else:
                errors['linalg'].append(i)

    cols = X.columns.tolist()
    if ('offset' in cols):
        cols.remove('offset')
    coefs = pd.DataFrame(coefs, columns=cols)
    coefs['p'] = p
    pvalues = pd.DataFrame(pvalues, columns=cols)
    pvalues['p'] = p
    scores = pd.DataFrame(scores)
    scores['p'] = p

    return Y_pred, coefs, pvalues, scores, errors


def fit_all(X, y, conf, model_tag, pred_tag, include_lags=False, data_tag='dhs'):
    coefs = []
    pvalues = []
    scores = []
    errors = []
    ps = range(5, 100, 5)

    if model_tag == 'clf':
        model_func = clf
    elif model_tag == 'negbin':
        assert 'offset' in X.columns
        model_func = negbin
    else:
        model_func = rgr

    for p in ps:
        print ".",
        train_ix = np.load(conf['train_ix_fn'](data_tag, p))
        test_ix = np.load(conf['test_ix_fn'](data_tag, p))
        lags = None
        if include_lags:
            lags = pd.DataFrame(
                np.load(conf['lags_fn'](data_tag, model_tag, p)).T, index=X.index)
            X['lag'] = 0.

        Y_pred, cfs, pvs, scr, ers = iter_fit(
            X, y, p, train_ix, test_ix, model_func, lags=lags)
        coefs.append(cfs)
        pvalues.append(pvs)
        scores.append(scr)
        np.save(conf['preds_fn'](data_tag, model_tag, pred_tag, p), Y_pred)
        errors.append(ers)
    print ""

    coefs = pd.concat(coefs)
    coefs.to_csv(conf['coefs_fn'](data_tag, model_tag, pred_tag))
    pvalues = pd.concat(pvalues)
    pvalues.to_csv(conf['pvalues_fn'](data_tag, model_tag, pred_tag))
    scores = pd.concat(scores).fillna(0)
    scores.to_csv(conf['scores_fn'](data_tag, model_tag, pred_tag))

    return coefs, pvalues, scores, errors
