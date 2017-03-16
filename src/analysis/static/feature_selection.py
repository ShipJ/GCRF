import pandas as pd
import statsmodels.formula.api as smf
from prettytable import PrettyTable
from src.config import config
import numpy as np
import sys

def normalise(df, adm):
    d = df.ix[:, 0]
    f = df.ix[:, 1:]

    f = f[np.abs(f - f.mean()) <= (3 * f.std())]
    f.insert(0, adm, d)
    df = f.dropna()

    f = df.ix[:, 1:]
    f = (f - f.mean()) / f.std()
    f.insert(0, adm, d)

    return f


def merge(a, b, adm):
    """
    Merge two dataframes on administrative level, then drop unnecessary columns ready for analysis
    :param a: dataframe
    :param b: dataframe
    :return:
    """
    merged = a.merge(b, on=adm)
    return merged.drop(adm, axis=1)


def get_country_data(country):
    PATH = config.get_dir()
    return pd.DataFrame(pd.read_csv(PATH+'/final/%s/master_2.0.csv' % country))


def output_model(country, model, adm, response):

    data = get_country_data(country)

    if model == 'Baseline':
        baseline_features = 'Log_pop_density'

        cdr_dhs = data[[adm,
                        baseline_features,
                        response]].dropna()

        cdr_dhs = normalise(cdr_dhs, adm)

        cdr = cdr_dhs.groupby(adm)[baseline_features].mean().reset_index()
        dhs = cdr_dhs.groupby(adm)[response].mean().reset_index()

        return merge(cdr, dhs, adm)

    elif model == 'CDR':
        cdr_dhs = data[[adm,
                        'Entropy', 'Med_degree', 'Degree_centrality',
                        'Introversion', 'G_residuals', 'Log_vol', 'Vol_pp',
                        response]].dropna()

        cdr_dhs = normalise(cdr_dhs, adm)

        cdr = cdr_dhs.groupby(adm)['Entropy', 'Med_degree', 'Degree_centrality',
                                   'Introversion','G_residuals','Log_vol', 'Vol_pp'].mean().reset_index()
        dhs = cdr_dhs.groupby(adm)[response].mean().reset_index()

        return merge(cdr, dhs, adm)

    elif model == 'Baseline+CDR':
        cdr_dhs = data[[adm,
                        'Log_pop_density',
                        'Entropy', 'Med_degree', 'Degree_centrality',
                        'Introversion', 'G_residuals', 'Log_vol', 'Vol_pp',
                        response]].dropna()

        cdr_dhs = normalise(cdr_dhs, adm)

        cdr = cdr_dhs.groupby(adm)['Log_pop_density',
                                 'Entropy', 'Med_degree', 'Degree_centrality',
                                 'Introversion','G_residuals', 'Log_vol', 'Vol_pp'].mean().reset_index()
        dhs = cdr_dhs.groupby(adm)[response].mean().reset_index()
        return merge(cdr, dhs, adm)

    elif model == 'Lag':
        lag_features = '%sSL' % response

        cdr_dhs = data[[adm,
                        lag_features,
                        response]].dropna()

        cdr_dhs = normalise(cdr_dhs, adm)

        cdr = cdr_dhs.groupby(adm)[lag_features].mean().reset_index()
        dhs = cdr_dhs.groupby(adm)[response].mean().reset_index()
        return merge(cdr, dhs, adm)

    elif model == 'Baseline+Lag':
        cdr_dhs = data[[adm,
                        'Log_pop_density',
                        '%sSL' % response,
                        response]].dropna()

        cdr_dhs = normalise(cdr_dhs, adm)

        cdr = cdr_dhs.groupby(adm)['Log_pop_density',
                                   '%sSL' % response].mean().reset_index()
        dhs = cdr_dhs.groupby(adm)[response].mean().reset_index()
        return merge(cdr, dhs, adm)

    elif model == 'All':
        cdr_dhs = data[[adm,
                        'Log_pop_density',
                        '%sSL' % response,
                        'Entropy',
                        'Introversion', 'G_residuals', 'Log_vol',
                        response]].dropna()

        cdr_dhs = normalise(cdr_dhs, adm)

        cdr = cdr_dhs.groupby(adm)['Log_pop_density',
                                   '%sSL' % response,
                                   'Entropy',
                                   'Introversion', 'G_residuals', 'Log_vol'].mean().reset_index()
        dhs = cdr_dhs.groupby(adm)[response].mean().reset_index()
        return merge(cdr, dhs, adm)


def forward_selected(data, response):
    """
    :param data: dataframe containing predictors X and response variable y
    :param response: name of response variable
    :return: fitted linear model with optimised R^2 value
    """

    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))

        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score

    formula = "{} ~ {} + 1".format(response, ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model

def stepwise_regression(country, response):
    models = config.get_headers(country, 'models')
    adms = config.get_headers(country, 'adm')

    model_table = PrettyTable()
    model_table.field_names = ['Model'] + adms

    for model in models:
        formulas, r2 = [], []

        for adm in adms:
            data = output_model(country, model, adm, response)

            selected = forward_selected(data, response)
            print adm, response, model, selected.summary()
            # np.savetxt('../../../reports/results/%s/statstables/%s_%s.txt' % (country, response, adm),
            #            [selected.summary().as_csv()], delimiter=' ', fmt='%s')

            formulas.append(selected.model.formula), r2.append(selected.rsquared_adj)

        model_table.add_row([model,
                             'Model: %s\nR^2-adj: %f\n' % (formulas[0], r2[0]),
                             'Model: %s\nR^2-adj: %f\n' % (formulas[1], r2[1]),
                             'Model: %s\nR^2-adj: %f\n' % (formulas[2], r2[2]),
                             'Model: %s\nR^2-adj: %f\n' % (formulas[3], r2[3])])

    return model_table


