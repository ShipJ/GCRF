

def get(loc_tag):
    if not loc_tag in ['sen', 'civ']:
        raise ValueError('loc_tag must be either "sen" or "civ".')
    data_dir = '/Volumes/Storage/chris/data/mobiflow/%s' % loc_tag
    utm = 30 if loc_tag == 'civ' else 28
    adm_geom_dir = '%s/geo/%s_adm' % (data_dir, loc_tag.upper())
    return {
        'data_dir': data_dir,
        'adm_geom_dir': adm_geom_dir,
        'dhs_fn': '%s/dhs_cluster_wealth.csv' % data_dir,
        'bts_xzero_fn': '%s/cdr_data/bts_xzero.csv' % data_dir,
        'adm_geom_fn': lambda i: '%s/%s_adm%d_utm%d.shp' % (adm_geom_dir, loc_tag, i, utm),
        'capital_geom_fn': '%s/capital_region.shp' % adm_geom_dir,
        'sqr_grid_fn': lambda size: '%s/geo/grid_points_%d.csv' % (data_dir, size),
        'hex_grid_fn': lambda size: '%s/geo/hex_grid_%d.csv' % (data_dir, size),
        'total_vol_fn': '%s/cdr_data/flows/total_vol.npy' % data_dir,
        'vol_fn': lambda data_tag,size: '%s/cdr_data/flows/%s%d_vol.npy' % (data_dir, data_tag, size),
        'noncapital_vol_fn': lambda data_tag,size: '%s/cdr_data/flows/%s%d_noncapital_vol.npy' % (data_dir, data_tag, size),
        'capital_vol_fn': lambda data_tag,size: '%s/cdr_data/flows/%s%d_capital_vol.npy' % (data_dir, data_tag, size),

        'dur_fn': lambda prefix: '%s/cdr_data/flows/%s_dur.npy' % (data_dir, prefix),
        'offset': 1 if loc_tag == 'civ' else -1,

        'bts_label_fn': lambda data_tag,size: '%s/cdr_data/bts_%s%d_labels.csv' % (data_dir,data_tag, size),
        'centroids_fn': lambda data_tag,size: '%s/cdr_data/%s%d_bts_centroids.csv' % (data_dir, data_tag, size),
        'noncapital_centroids_fn': lambda data_tag,size: '%s/cdr_data/%s%d_bts_noncapital_centroids.csv' % (data_dir, data_tag, size),
        'capital_centroids_fn': lambda data_tag,size: '%s/cdr_data/%s%d_bts_capital_centroids.csv' % (data_dir, data_tag, size),

        'rad_fn': lambda data_tag,size: '%s/cdr_data/flows/%s%d_radiation.csv' % (data_dir, data_tag, size),
        'noncapital_rad_fn': lambda data_tag,size: '%s/cdr_data/flows/%s%d_radiation_noncapital.csv' % (data_dir, data_tag, size),
        'capital_rad_fn': lambda data_tag,size: '%s/cdr_data/flows/%s%d_radiation_capital.csv' % (data_dir, data_tag, size),
        'flows_fn': lambda data_tag,size: '%s/cdr_data/flows/%s%d_flows.csv' % (data_dir, data_tag, size),
        'noncapital_flows_fn': lambda data_tag,size: '%s/cdr_data/flows/%s%d_flows_noncapital.csv' % (data_dir, data_tag, size),
        'capital_flows_fn': lambda data_tag,size: '%s/cdr_data/flows/%s%d_flows_capital.csv' % (data_dir, data_tag, size),

        'lm_params_fn': lambda data_tag,size: '%s/model_data/%s%d_lm_params.txt' % (data_dir, data_tag, size),
        'centroid_feats_fn': lambda data_tag,size: '%s/model_data/%s%d_centroid_features.csv' % (data_dir, data_tag, size),
        'noncapital_centroid_feats_fn': lambda data_tag,size: '%s/model_data/%s%d_noncapital_centroid_features.csv' % (data_dir, data_tag, size),
        'capital_centroid_feats_fn': lambda data_tag,size: '%s/model_data/%s%d_capital_centroid_features.csv' % (data_dir, data_tag, size),
        'dhs_feats_fn': lambda data_tag,size: '%s/model_data/%s%d_dhs_features.csv' % (data_dir, data_tag, size),
        'noncapital_dhs_feats_fn': lambda data_tag,size: '%s/model_data/%s%d_noncapital_dhs_features.csv' % (data_dir, data_tag, size),
        'capital_dhs_feats_fn': lambda data_tag,size: '%s/model_data/%s%d_capital_dhs_features.csv' % (data_dir, data_tag, size),

        'train_ix_fn': lambda data_tag, perc: '%s/model_data/%s_train_ix_%d.npy' % (data_dir, data_tag, perc),
        'test_ix_fn': lambda data_tag, perc: '%s/model_data/%s_test_ix_%d.npy' % (data_dir, data_tag, perc),
        'lags_fn': lambda data_tag, model_tag, perc: '%s/model_data/%s_lags_%s_%d.npy' % (data_dir, data_tag, model_tag, perc),

        'preds_fn': lambda data_tag, model_tag, pred_tag, perc: '%s/model_data/%s_%s_%s_preds_%d.npy' % (data_dir, data_tag, model_tag, pred_tag, perc),
        'coefs_fn': lambda data_tag, model_tag, pred_tag: '%s/model_data/%s_%s_%s_coefs.csv' % (data_dir, data_tag, model_tag, pred_tag),
        'pvalues_fn': lambda data_tag, model_tag, pred_tag: '%s/model_data/%s_%s_%s_pvalues.csv' % (data_dir, data_tag, model_tag, pred_tag),
        'scores_fn': lambda data_tag, model_tag, pred_tag: '%s/model_data/%s_%s_%s_scores.csv' % (data_dir, data_tag, model_tag, pred_tag),
    }











#
