import pandas as pd
import statsmodels.formula.api as smf
from prettytable import PrettyTable
from src.config import config


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
    source = config.get_dir()
    if country == 'civ':
        cdr = pd.DataFrame(pd.read_csv(source+'/final/%s/cdr.csv' % country)).dropna()
        dhs = pd.DataFrame(pd.read_csv(source+'/final/%s/dhs.csv' % country)).dropna()
        return cdr, dhs
    elif country == 'sen':
        cdr = pd.DataFrame(pd.read_csv(source+'/final/%s/cdr_pop_1km.csv' % country)).dropna()
        dhs = pd.DataFrame(pd.read_csv(source+'/final/%s/dhs.csv' % country)).dropna()
        return cdr, dhs
    else:
        country = config.get_country()
        return get_country_data(country)


def output_model(country, model, adm, response):

    cdr, dhs = get_country_data(country)

    if model == 'Baseline':
        set1 = cdr.groupby(adm)['Log_density_2010'].mean().reset_index()
        set2 = dhs.groupby(adm)[response].mean().reset_index()
        return merge(set1, set2, adm)

    elif model == 'Baseline+CDR':
        set1_mean = cdr.groupby(adm)['Introversion', 'Log_density_2010',
                                     'Entropy', 'EigenvectorCentrality',
                                     'Residuals', 'Vol_pp'].mean().reset_index()
        set1_sum = cdr.groupby(adm)['Vol'].mean().reset_index()
        set1 = set1_sum.merge(set1_mean, on=adm)
        set2 = dhs.groupby(adm)[response].mean().reset_index()
        return merge(set1, set2, adm)

    elif model == 'CDR':
        set1_sum = cdr.groupby(adm)['Vol'].sum().reset_index()
        set1_mean = cdr.groupby(adm)['Entropy', 'EigenvectorCentrality',
                                     'Vol_pp', 'Residuals', 'Introversion'].mean().reset_index()
        set1 = set1_sum.merge(set1_mean, on=adm)
        set2 = dhs.groupby(adm)[response].mean().reset_index()
        return merge(set1, set2, adm)

    elif model == 'SpatialLag':
        set1 = dhs.groupby(adm)['SpatialLag%s' % response].mean().reset_index()
        set2 = dhs.groupby(adm)[response].mean().reset_index()
        return merge(set1, set2, adm)

    elif model == 'Baseline+Lag':
        spatial = dhs.groupby(adm)['SpatialLag%s' % response].mean().reset_index()
        log_dense = cdr.groupby(adm)['Log_density_2010'].mean().reset_index()
        set1 = spatial.merge(log_dense, on=adm)
        set2 = dhs.groupby(adm)[response].mean().reset_index()
        return merge(set1, set2, adm)

    elif model == 'All':
        spatial = dhs.groupby(adm)['SpatialLag%s' % response].mean().reset_index()
        log_dense = cdr.groupby(adm)['Log_density_2010'].mean().reset_index()
        cdr_mean = cdr.groupby(adm)['Residuals', 'Introversion',
                                     'Entropy', 'EigenvectorCentrality',
                                     'Vol_pp'].mean().reset_index()
        cdr_sum = cdr.groupby(adm)['Vol'].sum().reset_index()
        set1 = cdr_mean.merge(cdr_sum, on=adm).merge(log_dense, on=adm).merge(spatial, on=adm)
        set2 = dhs.groupby(adm)[response].mean().reset_index()
        return merge(set1, set2, adm)


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
    if country == 'civ':
        models = ['Baseline', 'Baseline+CDR', 'CDR', 'SpatialLag', 'Baseline+Lag', 'All']
    elif country == 'sen':
        models = ['Baseline', 'Baseline+CDR', 'CDR']
    else:
        models=[]

    model_table = PrettyTable()
    model_table.field_names = ['Model'] + ['Adm_1', 'Adm_2', 'Adm_3', 'Adm_4']

    for model in models:
        adm_levels = ['Adm_1', 'Adm_2', 'Adm_3', 'Adm_4']
        formulas, r2 = [], []

        for adm in adm_levels:
            data = output_model(country, model, adm, response)

            selected = forward_selected(data, response)
            formulas.append(selected.model.formula), r2.append(selected.rsquared_adj)

        model_table.add_row([model,
                             'Model: %s\nR^2-adj: %f\n' % (formulas[0], r2[0]),
                             'Model: %s\nR^2-adj: %f\n' % (formulas[1], r2[1]),
                             'Model: %s\nR^2-adj: %f\n' % (formulas[2], r2[2]),
                             'Model: %s\nR^2-adj: %f\n' % (formulas[3], r2[3])])

    print model_table



