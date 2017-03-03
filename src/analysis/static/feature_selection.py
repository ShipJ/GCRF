import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from prettytable import PrettyTable
from src.config import config


def get_data(country, model, adm):
    if country == 'civ':
        cdr = pd.DataFrame(pd.read_csv('../../../data/final/cdr.csv')).dropna()
        dhs = pd.DataFrame(pd.read_csv('../../../data/final/dhs.csv')).dropna()
        if model == 'Baseline':
            set1_sum = cdr.groupby(adm)['Pop_2010', 'Pop_2014', 'Area_km2'].sum().reset_index()
            set1_mean = cdr.groupby(adm)['Pop_density_2010', 'Pop_density_2014', 'Pop_1km'].mean().reset_index()
            set1 = set1_sum.merge(set1_mean, on=adm)
            set2 = dhs.groupby(adm)['HIVPosRate'].mean().reset_index()
            return set1, set2
        elif model == 'Baseline+CDR':
            set1_sum = cdr.groupby(adm)['Pop_2010', 'Pop_2014', 'Area_km2',
                                        'Vol', 'Vol_in', 'Vol_out', 'Dur', 'Dur_in', 'Dur_out'].sum().reset_index()
            set1_mean = cdr.groupby(adm)['Log_pop_density','Entropy', 'Introversion', 'Med_degree',
                                         'Degree_centrality', 'EigenvectorCentrality',
                                         'Vol_pp'].mean().reset_index()
            set1 = set1_sum.merge(set1_mean, on=adm)
            set2 = dhs.groupby(adm)['HIVPosRate'].sum().reset_index()
            return set1, set2
        elif model == 'CDR':
            set1_sum = cdr.groupby(adm)['Vol', 'Vol_in', 'Vol_out', 'Dur', 'Dur_in', 'Dur_out'].sum().reset_index()
            set1_mean = cdr.groupby(adm)['Entropy', 'Introversion', 'Med_degree', 'Degree_centrality',
                                         'EigenvectorCentrality', 'Vol_pp'].mean().reset_index()
            set1 = set1_sum.merge(set1_mean, on=adm)
            set2 = dhs.groupby(adm)['HIVPosRate'].sum().reset_index()
            return set1, set2

    elif country == 'sen':
        cdr = pd.DataFrame(pd.read_csv('../../../data/final/cdr.csv' % country)).dropna()
        dhs = pd.DataFrame(pd.read_csv('../../../data/final/dhs.csv' % country)).dropna()
        if model == 'Baseline':
            set1 = 1
            set2 = 1
            return set1, set2
        elif model == 'Baseline+CDR':
            set1 = 1
            set2 = 1
            return set1, set2
        elif model == 'CDR':
            set1 = 1
            set2 = 1
            return set1, set2
        else:
            print 'Model choice not recognised, try running again.'
    else:
        print 'Sorry, please type a genuine location code...'
        config.get_country()


def merge(a, b, adm):
    """
    Merge two dataframes on administrative level, then drop unnecessary columns ready for analysis
    :param a: dataframe
    :param b: dataframe
    :return:
    """
    merged = a.merge(b, on=adm)
    return merged.drop(adm, axis=1)


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
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model

def stepwise_regression(country):

    models = ['Baseline', 'Baseline+CDR', 'CDR']

    for model in models:

        adm_levels = ['Adm_1', 'Adm_2', 'Adm_3', 'Adm_4']
        model_table = PrettyTable()
        model_table.field_names = ['Model'] + adm_levels
        formulas, r2 = [], []

        for adm in adm_levels:

            set1, set2 = get_data(country, model, adm)
            data = merge(set1, set2, adm)
            selected = forward_selected(data, 'HIVPosRate')
            formulas.append(selected.model.formula), r2.append(selected.rsquared_adj)

        model_table.add_row([model,
                             'Model: %s\nR^2-adj: %f' % (formulas[0], r2[0]),
                             'Model: %s\nR^2-adj: %f' % (formulas[1], r2[1]),
                             'Model: %s\nR^2-adj: %f' % (formulas[2], r2[2]),
                             'Model: %s\nR^2-adj: %f' % (formulas[3], r2[3])])
        print model_table

