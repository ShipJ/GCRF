library(RcppCNPy) # npyLoad
library(pscl) # zeroinf, hurdle
library(reshape2)
library(ncf) # lisa
library(plyr) # .

source('ggplot_blank_theme.R')


precision_at_k <- function (k, actual, predicted)  {
    sum(head(predicted, k) %in% head(actual, k))
}

dcg <- function(y) {
    sum(2^y/log(2:(length(y)+1)))
}

ndcg <- function(y) {
    dcg(y) / dcg(sort(y, decreasing=TRUE))
}

rmse <- function(x, y) sqrt(mean((x - y)^2))


plot_scores <- function(scores, ttl, random_baseline) {
    ggplot(scores[title==ttl], aes(train_size, value)) +
        geom_hline(yintercept=random_baseline, linetype='dashed', size=.7) +
        geom_line(aes(group=variable), size=.7) +
        geom_point(aes(shape=variable), size=2.5) +
        scale_shape_discrete(name='Model', labels=0:5) +
        xlab('Train proportion (%)') +
        ylab(ttl) +
        # facet_wrap(~title, scales='free_y') +
        theme_minimal() +
        theme(axis.title=element_text(size=16),
              axis.text=element_text(size=10),
              legend.title=element_text(size=16, face='plain'),
              legend.text=element_text(size=10))
        # theme(axis.title.y=element_blank())
}

map_errors <- function(df, trsize, model_name, geom, fname=NULL, capital_only=FALSE, city2_only=FALSE) {
    df[, var:=eval(parse(text=model_name))]
    if (capital_only) {
        tmp <- df[capital==TRUE]
    }
    else if (city2_only) {
        tmp <- df[city2==TRUE]
    }
    else {
        tmp <- df
    }
    plt <- ggplot(tmp[train_size==trsize]) +
        # geom_polygon(data=geom, aes(long,lat,group=group), fill=NA, colour='grey40', size=.2) +
        # geom_point(aes_string('x', 'y', colour=var)) +
        # scale_colour_gradientn(name='Residual\nError', colours=rainbow(3)) +
        geom_path(data=geom, aes(long,lat,group=group), colour='grey40', size=.2) +
        geom_point(aes(x, y, size=abs(var), colour=var<=0), alpha=.8) +
        # subset ensures negative cors are plotted on top
        geom_point(aes(x, y, size=abs(var), colour=var<0), alpha=.7, subset=.(var<=0)) +
        scale_size_continuous(limits=c(0, max(abs(df$var)))) +
        scale_colour_manual(labels=c('+ve','-ve'), values=c('#FC291C','#4D88CC'), na.value='grey40') +
        coord_equal() + blank_theme + theme(legend.title=element_blank())
    if (!is.null(fname)) {
        ggsave(sprintf('../figs/%s/%s.png', loc_tag, fname), plot=plt, width=10, height=7.5)
    }
    plt
}

data_dir <- '/Users/chris/data/mobiflow/data'
loc_tag <- 'sen'
df <- fread(sprintf('%s/%s/model_data/hex1000_dhs_features.csv', data_dir, loc_tag))

# df <- df[, c('vol_out_sum', 'vol_out_std', 'vol_out_ent', 'vol_norm', 'introv',
#             'vol_pagerank', 'vol_evc', 'grv_out_sum_res', 'grv_out_std_res',
#             'grv_out_ent_res', 'grv_pagerank_res', 'grv_evc_res',
#             'grv_smean_neg_in_res', 'grv_smean_neg_out_res',
#             'pop_1km', 'n', 'poor', 'poverty_rate', 'z_median', 'clust_id'), with=FALSE]

df[, ':='(clust_id=as.character(clust_id),
        pop_1km=scale(log(pop_1km)),
        vol_out_sum=scale(log(vol_out_sum)),
        vol_out_std=scale(log(vol_out_std)),
        vol_norm=scale(log(vol_norm)),
        vol_pagerank=scale(log(vol_pagerank)),
        vol_evc=scale(log(vol_evc)),
        grv_out_sum_res=scale(ifelse(grv_out_sum_res < -400, -400, grv_out_sum_res)),
        grv_out_std_res=scale(ifelse(grv_out_std_res < -20, -20, grv_out_std_res)),
        grv_evc_res=scale(ifelse(grv_evc_res< -.01,-.01,ifelse(grv_evc_res>.01,.01,grv_evc_res))),
        grv_smean_neg_in_res=scale(log(-grv_smean_neg_in_res)),
        grv_smean_neg_out_res=scale(log(-grv_smean_neg_out_res)))]

setkey(df, clust_id)

utm <- ifelse(loc_tag=='sen', 28, 30)
geom <- fread(sprintf('%s/%s/geo/%s_adm/%s_adm0_utm%d_geom.csv',
            data_dir, loc_tag, toupper(loc_tag), loc_tag, utm))
cap_geom <- fread(sprintf('%s/%s/geo/%s_adm/capital_region.csv',
            data_dir, loc_tag, toupper(loc_tag)))
city2_geom <- fread(sprintf('%s/%s/geo/%s_adm/city2_region.csv',
            data_dir, loc_tag, toupper(loc_tag)))


# ----------------
#     POVERTY RATE
# ----------------

print ('poverty rate')
r <- ifelse(loc_tag=='civ', 29000, 64000)
df[, ':='(lisa_mean_dist=0, lisa_cor=0, lisa_n_nbs=0, lisa_p_value=0)]
lis <- df[, lisa(x, y, poverty_rate, neigh=r)]
df[, ':='(lisa_mean_dist=lis$dmean, lisa_cor=lis$correlation,
            lisa_n_nbs=lis$n, lisa_p_value=lis$p)]

mean_scores <- list()
all_preds <- list()
all_errors <- list()

for (i in seq(50, 95, 5)) {
    print (paste("train size ", i))
    lags <- npyLoad(sprintf('%s/%s/model_data/dhs_lags_poverty_%d.npy', data_dir, loc_tag, i))
    train_ix <- npyLoad(sprintf('%s/%s/model_data/dhs_train_ix_%d.npy', data_dir, loc_tag, i), type='integer')
    test_ix <- npyLoad(sprintf('%s/%s/model_data/dhs_test_ix_%d.npy', data_dir, loc_tag, i), type='integer')

    scores <- list()
    preds <- list()
    errors <- list()

    for (j in 1:1000) {
        if (j %% 50 == 0) {
            print (j)
        }
        df[order(as.numeric(clust_id)), lag:=lags[j,]]
        X_train <- df[as.character(train_ix[j,])]
        X_test <- df[as.character(test_ix[j,])]

        hurd0 <- hurdle(poor ~ pop_1km + offset(log(n)), data=X_train, dist='poisson')
        hurd1 <- hurdle(poor ~ lag + offset(log(n)), data=X_train, dist='poisson')
        hurd2 <- hurdle(poor ~ pop_1km + lag + offset(log(n)), data=X_train, dist='poisson')
        hurd3 <- hurdle(poor ~ vol_out_sum + vol_out_std + vol_out_ent +
                        vol_norm + introv + vol_pagerank + vol_evc + grv_out_sum_res +
                        grv_out_std_res + grv_out_ent_res + grv_pagerank_res + grv_evc_res +
                        grv_smean_neg_in_res + grv_smean_neg_out_res + offset(log(n)),
                        data=X_train, dist='poisson')
        hurd4 <- hurdle(poor ~ pop_1km + vol_out_sum + vol_out_std + vol_out_ent +
                        vol_norm + introv + vol_pagerank + vol_evc + grv_out_sum_res +
                        grv_out_std_res + grv_out_ent_res + grv_pagerank_res + grv_evc_res +
                        grv_smean_neg_in_res + grv_smean_neg_out_res + offset(log(n)),
                        data=X_train, dist='poisson')
        hurd5 <- hurdle(poor ~ pop_1km + lag + vol_out_sum + vol_out_std + vol_out_ent +
                        vol_norm + introv + vol_pagerank + vol_evc + grv_out_sum_res +
                        grv_out_std_res + grv_out_ent_res + grv_pagerank_res + grv_evc_res +
                        grv_smean_neg_in_res + grv_smean_neg_out_res + offset(log(n)),
                        data=X_train, dist='poisson')

        prds <- X_test[, list(
                iter=j,
                clust_id,
                poverty_rate,
                hurd0=predict(hurd0, newdata=X_test)/n,
                hurd1=predict(hurd1, newdata=X_test)/n,
                hurd2=predict(hurd2, newdata=X_test)/n,
                hurd3=predict(hurd3, newdata=X_test)/n,
                hurd4=predict(hurd4, newdata=X_test)/n,
                hurd5=predict(hurd5, newdata=X_test)/n)]
        preds[[j]] <- prds

        errors[[j]] <- prds[, list(iter=j,
                                clust_id=clust_id,
                                hurd0=poverty_rate-hurd0,
                                hurd1=poverty_rate-hurd1,
                                hurd2=poverty_rate-hurd2,
                                hurd3=poverty_rate-hurd3,
                                hurd4=poverty_rate-hurd4,
                                hurd5=poverty_rate-hurd5)]

        ranked <- X_test[, clust_id[order(poverty_rate, decreasing=TRUE)]]

        scores[[j]] <- prds[, list(
                mae0=mean(abs(poverty_rate - hurd0)),
                mae1=mean(abs(poverty_rate - hurd1)),
                mae2=mean(abs(poverty_rate - hurd2)),
                mae3=mean(abs(poverty_rate - hurd3)),
                mae4=mean(abs(poverty_rate - hurd4)),
                mae5=mean(abs(poverty_rate - hurd5)),
                rmse0=rmse(poverty_rate, hurd0),
                rmse1=rmse(poverty_rate, hurd1),
                rmse2=rmse(poverty_rate, hurd2),
                rmse3=rmse(poverty_rate, hurd3),
                rmse4=rmse(poverty_rate, hurd4),
                rmse5=rmse(poverty_rate, hurd5),
                prec0=precision_at_k(10, ranked, clust_id[order(hurd0, decreasing=TRUE)]),
                prec1=precision_at_k(10, ranked, clust_id[order(hurd1, decreasing=TRUE)]),
                prec2=precision_at_k(10, ranked, clust_id[order(hurd2, decreasing=TRUE)]),
                prec3=precision_at_k(10, ranked, clust_id[order(hurd3, decreasing=TRUE)]),
                prec4=precision_at_k(10, ranked, clust_id[order(hurd4, decreasing=TRUE)]),
                prec5=precision_at_k(10, ranked, clust_id[order(hurd5, decreasing=TRUE)]),
                ndcg0=ndcg(poverty_rate[rev(order(hurd0))]),
                ndcg1=ndcg(poverty_rate[rev(order(hurd1))]),
                ndcg2=ndcg(poverty_rate[rev(order(hurd2))]),
                ndcg3=ndcg(poverty_rate[rev(order(hurd3))]),
                ndcg4=ndcg(poverty_rate[rev(order(hurd4))]),
                ndcg5=ndcg(poverty_rate[rev(order(hurd5))]),
                spearman0=cor(poverty_rate, hurd0, method='sp'),
                spearman1=cor(poverty_rate, hurd1, method='sp'),
                spearman2=cor(poverty_rate, hurd2, method='sp'),
                spearman3=cor(poverty_rate, hurd3, method='sp'),
                spearman4=cor(poverty_rate, hurd4, method='sp'),
                spearman5=cor(poverty_rate, hurd5, method='sp')
            )]
    }

    preds <- rbindlist(preds)
    preds[, train_size:=i]
    all_preds[[i]] <- preds

    errors <- rbindlist(errors)
    errors[, train_size:=i]
    all_errors[[i]] <- errors

    scores <- rbindlist(scores)
    mean_scores[[i]] <- scores[, list(
            train_size=i,
            mae0=mean(mae0), mae0_sd=sd(mae0),
            mae1=mean(mae1), mae1_sd=sd(mae1),
            mae2=mean(mae2), mae2_sd=sd(mae2),
            mae3=mean(mae3), mae3_sd=sd(mae3),
            mae4=mean(mae4), mae4_sd=sd(mae4),
            mae5=mean(mae5), mae5_sd=sd(mae5),
            rmse0=mean(rmse0), rmse0_sd=sd(rmse0),
            rmse1=mean(rmse1), rmse1_sd=sd(rmse1),
            rmse2=mean(rmse2), rmse2_sd=sd(rmse2),
            rmse3=mean(rmse3), rmse3_sd=sd(rmse3),
            rmse4=mean(rmse4), rmse4_sd=sd(rmse4),
            rmse5=mean(rmse5), rmse5_sd=sd(rmse5),
            prec0=mean(prec0), prec0_sd=sd(prec0),
            prec1=mean(prec1), prec1_sd=sd(prec1),
            prec2=mean(prec2), prec2_sd=sd(prec2),
            prec3=mean(prec3), prec3_sd=sd(prec3),
            prec4=mean(prec4), prec4_sd=sd(prec4),
            prec5=mean(prec5), prec5_sd=sd(prec5),
            ndcg0=mean(ndcg0), ndcg0_sd=sd(ndcg0),
            ndcg1=mean(ndcg1), ndcg1_sd=sd(ndcg1),
            ndcg2=mean(ndcg2), ndcg2_sd=sd(ndcg2),
            ndcg3=mean(ndcg3), ndcg3_sd=sd(ndcg3),
            ndcg4=mean(ndcg4), ndcg4_sd=sd(ndcg4),
            ndcg5=mean(ndcg5), ndcg5_sd=sd(ndcg5),
            spearman0=mean(spearman0), spearman0_sd=sd(spearman0),
            spearman1=mean(spearman1), spearman1_sd=sd(spearman1),
            spearman2=mean(spearman2), spearman2_sd=sd(spearman2),
            spearman3=mean(spearman3), spearman3_sd=sd(spearman3),
            spearman4=mean(spearman4), spearman4_sd=sd(spearman4),
            spearman5=mean(spearman5), spearman5_sd=sd(spearman5)
        )]
}

preds <- rbindlist(all_preds)
write.csv(preds, sprintf('%s/%s/model_data/poverty_hurdle_preds.csv', data_dir, loc_tag), row.names=FALSE)
# preds <- fread(sprintf('%s/%s/model_data/poverty_hurdle_preds.csv', data_dir, loc_tag))
mean_preds <- preds[, list(n=length(iter), poverty_rate=poverty_rate[1],
        hurd0=mean(hurd0), hurd1=mean(hurd1), hurd2=mean(hurd2),
        hurd3=mean(hurd3), hurd4=mean(hurd4), hurd5=mean(hurd5),
        hurd0_abs=mean(abs(hurd0)), hurd1_abs=mean(abs(hurd1)), hurd2_abs=mean(abs(hurd2)),
        hurd3_abs=mean(abs(hurd3)), hurd4_abs=mean(abs(hurd4)), hurd5_abs=mean(abs(hurd5)),
        hurd0_sd=sd(hurd0), hurd1_sd=sd(hurd1), hurd2_sd=sd(hurd2),
        hurd3_sd=sd(hurd3), hurd4_sd=sd(hurd4), hurd5_sd=sd(hurd5)
    ), by=list(train_size, clust_id)]

write.csv(mean_preds, sprintf('%s/%s/model_data/poverty_hurdle_mean_preds.csv', data_dir, loc_tag), row.names=FALSE)
mean_preds <- fread(sprintf('%s/%s/model_data/poverty_hurdle_mean_preds.csv', data_dir, loc_tag))

ranked <- df[, clust_id[order(poverty_rate, decreasing=TRUE)]]
mean_pred_scores <- rbindlist(lapply(seq(50,95,5), function(trsize) {
    rbindlist(lapply(0:5, function(i) {
        model <- paste('hurd', i, sep='')
        mean_preds[train_size==trsize, list(
                model=model, trsize=trsize,
                mae=mean(abs(poverty_rate - eval(parse(text=model)))),
                rmse=rmse(poverty_rate, eval(parse(text=model))),
                prec=precision_at_k(10, ranked, clust_id[order(eval(parse(text=model)), decreasing=TRUE)]),
                ndcg=ndcg(poverty_rate[order(eval(parse(text=model)), decreasing=TRUE)]),
                spearman=cor(poverty_rate, eval(parse(text=model)), method='s')
            )]
    }))
}))

ggplot(mean_pred_scores, aes(trsize, mae, colour=model)) + geom_line() + theme_minimal()

mean_pred_errors <- mean_preds[, list(
        hurd0=poverty_rate-hurd0, hurd1=poverty_rate-hurd1, hurd2=poverty_rate-hurd2,
        hurd3=poverty_rate-hurd3, hurd4=poverty_rate-hurd4, hurd5=poverty_rate-hurd5
    ), by=list(clust_id, train_size)]

errors <- rbindlist(all_errors)
write.csv(errors, sprintf('%s/%s/model_data/poverty_hurdle_errors.csv', data_dir, loc_tag), row.names=FALSE)
# errors <- fread(sprintf('%s/%s/model_data/poverty_hurdle_errors.csv', data_dir, loc_tag))

mean_errors <- errors[, list(n=length(iter),
        hurd0=mean(hurd0), hurd1=mean(hurd1), hurd2=mean(hurd2),
        hurd3=mean(hurd3), hurd4=mean(hurd4), hurd5=mean(hurd5),
        hurd0_abs=mean(abs(hurd0)), hurd1_abs=mean(abs(hurd1)), hurd2_abs=mean(abs(hurd2)),
        hurd3_abs=mean(abs(hurd3)), hurd4_abs=mean(abs(hurd4)), hurd5_abs=mean(abs(hurd5)),
        hurd0_sd=sd(hurd0), hurd1_sd=sd(hurd1), hurd2_sd=sd(hurd2),
        hurd3_sd=sd(hurd3), hurd4_sd=sd(hurd4), hurd5_sd=sd(hurd5)
    ), by=list(train_size, clust_id)]

write.csv(mean_errors, sprintf('%s/%s/model_data/poverty_hurdle_mean_errors.csv', data_dir, loc_tag), row.names=FALSE)
mean_errors <- fread(sprintf('%s/%s/model_data/poverty_hurdle_mean_errors.csv', data_dir, loc_tag))


mean_scores <- rbindlist(mean_scores)
write.csv(mean_scores, sprintf('%s/%s/model_data/poverty_hurdle_scores.csv', data_dir, loc_tag), row.names=FALSE)
mean_scores <- fread(sprintf('%s/%s/model_data/poverty_hurdle_scores.csv', data_dir, loc_tag))


mae_melt <- melt(mean_scores, id='train_size', measure=c('mae0','mae1','mae2','mae3','mae4','mae5'))
mae_melt[, title:='MAE']

rmse_melt <- melt(mean_scores, id='train_size', measure=c('rmse0','rmse1','rmse2','rmse3','rmse4','rmse5'))
rmse_melt[, title:='RMSE']

prec_melt <- melt(mean_scores, id='train_size', measure=c('prec0','prec1','prec2','prec3','prec4','prec5'))
prec_melt[, title:='Precision at 10']

ndcg_melt <- melt(mean_scores, id='train_size', measure=c('ndcg0','ndcg1','ndcg2','ndcg3','ndcg4','ndcg5'))
ndcg_melt[, title:='NDCG']

spearman_melt <- melt(mean_scores, id='train_size',
    measure=c('spearman0','spearman1','spearman2','spearman3','spearman4','spearman5'))
spearman_melt[, title:='Spearman\'s Rho']

scores <- rbindlist(list(mae_melt, rmse_melt, prec_melt, ndcg_melt, spearman_melt))


plt <- plot_scores(scores, 'MAE')
ggsave(sprintf('../figs/%s/poverty_hurdle_mae.png', loc_tag), plot=plt, width=6, height=4)
plt <- plot_scores(scores, 'RMSE')
ggsave(sprintf('../figs/%s/poverty_hurdle_rmse.png', loc_tag), plot=plt, width=6, height=4)
plt <- plot_scores(scores, 'Precision at 10')
ggsave(sprintf('../figs/%s/poverty_hurdle_prec10.png', loc_tag), plot=plt, width=6, height=4)
plt <- plot_scores(scores, 'NDCG')
ggsave(sprintf('../figs/%s/poverty_hurdle_ndcg.png', loc_tag), plot=plt, width=6, height=4)
plt <- plot_scores(scores, "Spearman's Rho")
ggsave(sprintf('../figs/%s/poverty_hurdle_spearman.png', loc_tag), plot=plt, width=6, height=4)


setkey(mean_errors, clust_id)
mean_errors <- mean_errors[df[, list(clust_id, x, y, capital, lisa_cor, lisa_n_nbs, lisa_p_value, mean_nbr_dist, n_nbrs_5km, n_nbrs_10km, n_nbrs_30km, n_nbrs_60km, n_nbrs_100km)]]
mean_errors[is.na(lisa_p_value), ':='(lisa_cor=0., lisa_p_value=-1)]

for (model in lapply(0:5, function(i) paste('hurd', i, sep=''))) {
    for (trsize in seq(50, 95, 5)) {
        plt <- map_errors(mean_errors, trsize, model, geom,
            sprintf('poverty_%s_%d_errors_map', model, trsize))
        plt <- map_errors(mean_errors, trsize, model, cap_geom,
            sprintf('poverty_%s_%d_errors_capital_map', model, trsize), capital_only=TRUE)
        # plt <- map_errors(mean_errors, trsize, model, city2_geom,
        #     sprintf('poverty_%s_%d_errors_city2_map', model, trsize), city2_only=TRUE)
    }
}

# mean_errors[, cor(hurd1, lisa_cor), by=train_size]
if (loc_tag=='civ') {
    mean_errors[, lisa_cor_cut:=cut(lisa_cor, c(-2,-.8,-.5,-.25,-.125,-.05,.05,.125,.25,.375,.5,.675,.75,1,6))]
} else {
    mean_errors[, lisa_cor_cut:=cut(lisa_cor, c(-2,-.5,seq(-.3,.8,.1),1,1.5,2,4))]
}

lisa_errors <- rbindlist(list(
    mean_errors[, list(model=0, lisa_mean=mean(lisa_cor),
            n=length(abs(hurd0)), m=mean(abs(hurd0)), lo=quantile(abs(hurd0),.25), hi=quantile(abs(hurd0),.75)),
        by=list(train_size, lisa_cor_cut)][order(lisa_mean)],
    mean_errors[, list(model=1, lisa_mean=mean(lisa_cor),
            n=length(abs(hurd1)), m=mean(abs(hurd1)), lo=quantile(abs(hurd1),.25), hi=quantile(abs(hurd1),.75)),
        by=list(train_size, lisa_cor_cut)][order(lisa_mean)],
    mean_errors[, list(model=2, lisa_mean=mean(lisa_cor),
            n=length(abs(hurd2)), m=mean(abs(hurd2)), lo=quantile(abs(hurd2),.25), hi=quantile(abs(hurd2),.75)),
        by=list(train_size, lisa_cor_cut)][order(lisa_mean)],
    mean_errors[, list(model=3, lisa_mean=mean(lisa_cor),
            n=length(abs(hurd3)), m=mean(abs(hurd3)), lo=quantile(abs(hurd3),.25), hi=quantile(abs(hurd3),.75)),
        by=list(train_size, lisa_cor_cut)][order(lisa_mean)],
    mean_errors[, list(model=4, lisa_mean=mean(lisa_cor),
            n=length(abs(hurd4)), m=mean(abs(hurd4)), lo=quantile(abs(hurd4),.25), hi=quantile(abs(hurd4),.75)),
        by=list(train_size, lisa_cor_cut)][order(lisa_mean)],
    mean_errors[, list(model=5, lisa_mean=mean(lisa_cor),
            n=length(abs(hurd5)), m=mean(abs(hurd5)), lo=quantile(abs(hurd5),.25), hi=quantile(abs(hurd5),.75)),
        by=list(train_size, lisa_cor_cut)][order(lisa_mean)]
    ))

lisa_errors[, ':='(has_pd=model %in% c(0,2,4,5), has_lag=model %in% c(1,5), has_cdr=model %in% 3:5)]

ggplot(lisa_errors[train_size==75], aes(lisa_mean, m, colour=as.factor(model))) +
    geom_line() +
    scale_colour_discrete(name='model') +
    theme_minimal()

ggsave(sprintf('../figs/%s/poverty_hurd_lisa_abs_errors.png', loc_tag), width=6, height=4)

plt <- ggplot(lisa_errors[train_size==75], aes(lisa_mean, m, group=as.factor(model))) +
    geom_line(aes(colour=has_cdr, linetype=has_pd)) +
    geom_point(aes(shape=has_lag), solid=FALSE) +
    xlab('LISA') + ylab('mean absolute error') +
    scale_colour_discrete(name='CDR') +
    scale_linetype(name='Pop') +
    scale_shape_discrete(name='lag', solid=FALSE) +
    theme_minimal()

ggsave(sprintf('../figs/%s/poverty_hurdle_lisa_abs_errors.png', loc_tag), plot=plt, height=6, width=10)

# --------------------
#  MEDIAN WEALTH INDEX
# --------------------

print ('median wealth index')
df[, ':='(lisa_mean_dist=0, lisa_cor=0, lisa_n_nbs=0, lisa_p_value=0)]
lis <- df[, lisa(x, y, z_median, neigh=r)]
df[, ':='(lisa_mean_dist=lis$dmean, lisa_cor=lis$correlation,
            lisa_n_nbs=lis$n, lisa_p_value=lis$p)]

mean_scores <- list()
all_preds <- list()
all_errors <- list()

for (i in seq(50, 95, 5)) {
    print (paste("train size ", i))
    lags <- npyLoad(sprintf('%s/%s/model_data/dhs_lags_wealth_%d.npy', data_dir, loc_tag, i))
    train_ix <- npyLoad(sprintf('%s/%s/model_data/dhs_train_ix_%d.npy', data_dir, loc_tag, i), type='integer')
    test_ix <- npyLoad(sprintf('%s/%s/model_data/dhs_test_ix_%d.npy', data_dir, loc_tag, i), type='integer')

    scores <- list()
    preds <- list()
    errors <- list()

    for (j in 1:1000) {
        if (j %% 50 == 0) {
            print (j)
        }
        df[order(as.numeric(clust_id)), lag:=lags[j,]]
        X_train <- df[as.character(train_ix[j,])]
        X_test <- df[as.character(test_ix[j,])]

        lm0 <- lm(z_median ~ pop_1km, data=X_train)
        lm1 <- lm(z_median ~ lag, data=X_train)
        lm2 <- lm(z_median ~ pop_1km + lag, data=X_train)
        lm3 <- lm(z_median ~ vol_out_sum + vol_out_std + vol_out_ent +
                    vol_norm + introv + vol_pagerank + vol_evc + grv_out_sum_res +
                    grv_out_std_res + grv_out_ent_res + grv_pagerank_res + grv_evc_res +
                    grv_smean_neg_in_res + grv_smean_neg_out_res,
                data=X_train)
        lm4 <- lm(z_median ~ pop_1km + vol_out_sum + vol_out_std + vol_out_ent +
                    vol_norm + introv + vol_pagerank + vol_evc + grv_out_sum_res +
                    grv_out_std_res + grv_out_ent_res + grv_pagerank_res + grv_evc_res +
                    grv_smean_neg_in_res + grv_smean_neg_out_res,
                data=X_train)
        lm5 <- lm(z_median ~ pop_1km + lag + vol_out_sum + vol_out_std + vol_out_ent +
                    vol_norm + introv + vol_pagerank + vol_evc + grv_out_sum_res +
                    grv_out_std_res + grv_out_ent_res + grv_pagerank_res + grv_evc_res +
                    grv_smean_neg_in_res + grv_smean_neg_out_res,
                data=X_train)

        prds <- X_test[, list(
                        iter=j,
                        clust_id,
                        z_median,
                        lm0=predict(lm0, newdata=X_test),
                        lm1=predict(lm1, newdata=X_test),
                        lm2=predict(lm2, newdata=X_test),
                        lm3=predict(lm3, newdata=X_test),
                        lm4=predict(lm4, newdata=X_test),
                        lm5=predict(lm5, newdata=X_test))]
        preds[[j]] <- prds

        errors[[j]] <- prds[, list(iter,
                                clust_id,
                                lm0=z_median-lm0,
                                lm1=z_median-lm1,
                                lm2=z_median-lm2,
                                lm3=z_median-lm3,
                                lm4=z_median-lm4,
                                lm5=z_median-lm5)]

        ranked <- X_test[, clust_id[order(z_median, decreasing=TRUE)]]

        scores[[j]] <- prds[, list(mae0=mean(abs(z_median - lm0)),
                          mae1=mean(abs(z_median - lm1)),
                          mae2=mean(abs(z_median - lm2)),
                          mae3=mean(abs(z_median - lm3)),
                          mae4=mean(abs(z_median - lm4)),
                          mae5=mean(abs(z_median - lm5)),
                          rmse0=rmse(z_median, lm0),
                          rmse1=rmse(z_median, lm1),
                          rmse2=rmse(z_median, lm2),
                          rmse3=rmse(z_median, lm3),
                          rmse4=rmse(z_median, lm4),
                          rmse5=rmse(z_median, lm5),
                          prec0=precision_at_k(10, ranked, clust_id[order(lm0, decreasing=TRUE)]),
                          prec1=precision_at_k(10, ranked, clust_id[order(lm1, decreasing=TRUE)]),
                          prec2=precision_at_k(10, ranked, clust_id[order(lm2, decreasing=TRUE)]),
                          prec3=precision_at_k(10, ranked, clust_id[order(lm3, decreasing=TRUE)]),
                          prec4=precision_at_k(10, ranked, clust_id[order(lm4, decreasing=TRUE)]),
                          prec5=precision_at_k(10, ranked, clust_id[order(lm5, decreasing=TRUE)]),
                          ndcg0=ndcg(z_median[rev(order(lm0))]),
                          ndcg1=ndcg(z_median[rev(order(lm1))]),
                          ndcg2=ndcg(z_median[rev(order(lm2))]),
                          ndcg3=ndcg(z_median[rev(order(lm3))]),
                          ndcg4=ndcg(z_median[rev(order(lm4))]),
                          ndcg5=ndcg(z_median[rev(order(lm5))]),
                          spearman0=cor(z_median, lm0, method='sp'),
                          spearman1=cor(z_median, lm1, method='sp'),
                          spearman2=cor(z_median, lm2, method='sp'),
                          spearman3=cor(z_median, lm3, method='sp'),
                          spearman4=cor(z_median, lm4, method='sp'),
                          spearman5=cor(z_median, lm5, method='sp')
                )]
    }

    preds <- rbindlist(preds)
    preds[, train_size:=i]
    all_preds[[i]] <- preds

    errors <- rbindlist(errors)
    errors[, train_size:=i]
    all_errors[[i]] <- errors

    scores <- rbindlist(scores)
    mean_scores[[i]] <- scores[, list(
                train_size=i,
                mae0=mean(mae0), mae0_sd=sd(mae0),
                mae1=mean(mae1), mae1_sd=sd(mae1),
                mae2=mean(mae2), mae2_sd=sd(mae2),
                mae3=mean(mae3), mae3_sd=sd(mae3),
                mae4=mean(mae4), mae4_sd=sd(mae4),
                mae5=mean(mae5), mae5_sd=sd(mae5),
                rmse0=mean(rmse0), rmse0_sd=sd(rmse0),
                rmse1=mean(rmse1), rmse1_sd=sd(rmse1),
                rmse2=mean(rmse2), rmse2_sd=sd(rmse2),
                rmse3=mean(rmse3), rmse3_sd=sd(rmse3),
                rmse4=mean(rmse4), rmse4_sd=sd(rmse4),
                rmse5=mean(rmse5), rmse5_sd=sd(rmse5),
                prec0=mean(prec0), prec0_sd=sd(prec0),
                prec1=mean(prec1), prec1_sd=sd(prec1),
                prec2=mean(prec2), prec2_sd=sd(prec2),
                prec3=mean(prec3), prec3_sd=sd(prec3),
                prec4=mean(prec4), prec4_sd=sd(prec4),
                prec5=mean(prec5), prec5_sd=sd(prec5),
                ndcg0=mean(ndcg0), ndcg0_sd=sd(ndcg0),
                ndcg1=mean(ndcg1), ndcg1_sd=sd(ndcg1),
                ndcg2=mean(ndcg2), ndcg2_sd=sd(ndcg2),
                ndcg3=mean(ndcg3), ndcg3_sd=sd(ndcg3),
                ndcg4=mean(ndcg4), ndcg4_sd=sd(ndcg4),
                ndcg5=mean(ndcg5), ndcg5_sd=sd(ndcg5),
                spearman0=mean(spearman0), spearman0_sd=sd(spearman0),
                spearman1=mean(spearman1), spearman1_sd=sd(spearman1),
                spearman2=mean(spearman2), spearman2_sd=sd(spearman2),
                spearman3=mean(spearman3), spearman3_sd=sd(spearman3),
                spearman4=mean(spearman4), spearman4_sd=sd(spearman4),
                spearman5=mean(spearman5), spearman5_sd=sd(spearman5)
        )]
}

preds <- rbindlist(all_preds)
write.csv(preds, sprintf('%s/%s/model_data/wealth_lm_preds.csv', data_dir, loc_tag), row.names=FALSE)
# preds <- fread(sprintf('%s/%s/model_data/wealth_lm_preds.csv', data_dir, loc_tag))
mean_preds <- preds[, list(n=length(iter), z_median=z_median[1],
        lm0=mean(lm0), lm1=mean(lm1), lm2=mean(lm2),
        lm3=mean(lm3), lm4=mean(lm4), lm5=mean(lm5),
        lm0_abs=mean(abs(lm0)), lm1_abs=mean(abs(lm1)), lm2_abs=mean(abs(lm2)),
        lm3_abs=mean(abs(lm3)), lm4_abs=mean(abs(lm4)), lm5_abs=mean(abs(lm5)),
        lm0_sd=sd(lm0), lm1_sd=sd(lm1), lm2_sd=sd(lm2),
        lm3_sd=sd(lm3), lm4_sd=sd(lm4), lm5_sd=sd(lm5)
    ), by=list(train_size, clust_id)]

write.csv(mean_preds, sprintf('%s/%s/model_data/wealth_lm_mean_preds.csv', data_dir, loc_tag), row.names=FALSE)
# mean_preds <- fread(sprintf('%s/%s/model_data/wealth_lm_mean_preds.csv', data_dir, loc_tag))

ranked <- df[, clust_id[order(z_median, decreasing=TRUE)]]
mean_pred_scores <- rbindlist(lapply(seq(50,95,5), function(trsize) {
    rbindlist(lapply(0:5, function(i) {
        model <- paste('lm', i, sep='')
        mean_preds[train_size==trsize, list(
                model=model, trsize=trsize,
                mae=mean(abs(z_median - eval(parse(text=model)))),
                rmse=rmse(z_median, eval(parse(text=model))),
                prec=precision_at_k(10, ranked, clust_id[order(eval(parse(text=model)), decreasing=TRUE)]),
                ndcg=ndcg(z_median[order(eval(parse(text=model)), decreasing=TRUE)]),
                spearman=cor(z_median, eval(parse(text=model)), method='s')
            )]
    }))
}))

ggplot(mean_pred_scores, aes(trsize, mae, colour=model)) + geom_line() + theme_minimal()
ggsave(sprintf('../figs/%s/wealth_lm_mean_pred_scores.png', loc_tag), width=6, height=4)

mean_errors <- mean_preds[, list(
        lm0=z_median-lm0, lm1=z_median-lm1, lm2=z_median-lm2,
        lm3=z_median-lm3, lm4=z_median-lm4, lm5=z_median-lm5
    ), by=list(clust_id, train_size)]


errors <- rbindlist(all_errors)
write.csv(errors, sprintf('%s/%s/model_data/wealth_lm_errors.csv', data_dir, loc_tag), row.names=FALSE)

mean_errors <- errors[, list(n=length(iter),
        lm0=mean(lm0), lm1=mean(lm1), lm2=mean(lm2),
        lm3=mean(lm3), lm4=mean(lm4), lm5=mean(lm5),
        lm0_abs=mean(abs(lm0)), lm1_abs=mean(abs(lm1)), lm2_abs=mean(abs(lm2)),
        lm3_abs=mean(abs(lm3)), lm4_abs=mean(abs(lm4)), lm5_abs=mean(abs(lm5)),
        lm0_sd=sd(lm0), lm1_sd=sd(lm1), lm2_sd=sd(lm2),
        lm3_sd=sd(lm3), lm4_sd=sd(lm4), lm5_sd=sd(lm5)
    ), by=list(train_size, clust_id)]

write.csv(mean_errors, sprintf('%s/%s/model_data/wealth_lm_mean_errors.csv', data_dir, loc_tag), row.names=FALSE)
mean_errors <- fread(sprintf('%s/%s/model_data/wealth_lm_mean_errors.csv', data_dir, loc_tag))


mean_scores <- rbindlist(mean_scores)
write.csv(mean_scores, sprintf('%s/%s/model_data/wealth_lm_scores.csv', data_dir, loc_tag), row.names=FALSE)
mean_scores <- fread(sprintf('%s/%s/model_data/wealth_lm_scores.csv', data_dir, loc_tag))

mae_melt <- melt(mean_scores, id='train_size', measure=c('mae0','mae1','mae2','mae3','mae4','mae5'))
mae_melt[, title:='MAE']

rmse_melt <- melt(mean_scores, id='train_size', measure=c('rmse0','rmse1','rmse2','rmse3','rmse4','rmse5'))
rmse_melt[, title:='RMSE']

prec_melt <- melt(mean_scores, id='train_size', measure=c('prec0','prec1','prec2','prec3','prec4','prec5'))
prec_melt[, title:='Precision at 10']

ndcg_melt <- melt(mean_scores, id='train_size', measure=c('ndcg0','ndcg1','ndcg2','ndcg3','ndcg4','ndcg5'))
ndcg_melt[, title:='NDCG']

spearman_melt <- melt(mean_scores, id='train_size',
    measure=c('spearman0','spearman1','spearman2','spearman3','spearman4','spearman5'))
spearman_melt[, title:='Spearman\'s Rho']

scores <- rbindlist(list(mae_melt, rmse_melt, prec_melt, ndcg_melt, spearman_melt))

plt <- plot_scores(scores, 'MAE')
ggsave(sprintf('../figs/%s/wealth_lm_mae.png', loc_tag), plot=plt, width=6, height=4)
plt <- plot_scores(scores, 'RMSE')
ggsave(sprintf('../figs/%s/wealth_lm_rmse.png', loc_tag), plot=plt, width=6, height=4)
plt <- plot_scores(scores, 'Precision at 10')
ggsave(sprintf('../figs/%s/wealth_lm_prec10.png', loc_tag), plot=plt, width=6, height=4)
plt <- plot_scores(scores, 'NDCG')
ggsave(sprintf('../figs/%s/wealth_lm_ndcg.png', loc_tag), plot=plt, width=6, height=4)
plt <- plot_scores(scores, "Spearman's Rho")
ggsave(sprintf('../figs/%s/wealth_lm_spearman.png', loc_tag), plot=plt, width=6, height=4)



setkey(mean_errors, clust_id)
mean_errors <- mean_errors[df[, list(clust_id, pop_1km, x, y, capital, lisa_cor, lisa_n_nbs, lisa_p_value)]]
mean_errors[is.na(lisa_p_value), ':='(lisa_cor=0., lisa_p_value=-1)]

for (model in lapply(0:5, function(i) paste('lm', i, sep=''))) {
    for (trsize in seq(50, 95, 5)) {
        plt <- map_errors(mean_errors, trsize, model, geom,
            sprintf('wealth_%s_%d_errors_map', model, trsize))
        plt <- map_errors(mean_errors, trsize, model, cap_geom,
            sprintf('wealth_%s_%d_errors_capital_map', model, trsize), capital_only=TRUE)
        plt <- map_errors(mean_errors, trsize, model, city2_geom,
            sprintf('wealth_%s_%d_errors_city2_map', model, trsize), city2_only=TRUE)
    }
}

if (loc_tag=='civ') {
    mean_errors[, lisa_cor_cut:=cut(lisa_cor, c(-3,-.5,-.25,-.125,-.05,.05,.125,.5,.675,.75,1,1.5,2,5))]
} else {
    mean_errors[, lisa_cor_cut:=cut(lisa_cor, c(-2,-.5,seq(-.3,.8,.1),1,1.5,2,4))]
}

lisa_errors <- rbindlist(list(
    mean_errors[, list(model=0, lisa_mean=mean(lisa_cor),
            n=length(abs(lm0)), m=mean(abs(lm0)), lo=quantile(abs(lm0),.25), hi=quantile(abs(lm0),.75)),
        by=list(train_size, lisa_cor_cut)][order(lisa_mean)],
    mean_errors[, list(model=1, lisa_mean=mean(lisa_cor),
            n=length(abs(lm1)), m=mean(abs(lm1)), lo=quantile(abs(lm1),.25), hi=quantile(abs(lm1),.75)),
        by=list(train_size, lisa_cor_cut)][order(lisa_mean)],
    mean_errors[, list(model=2, lisa_mean=mean(lisa_cor),
            n=length(abs(lm2)), m=mean(abs(lm2)), lo=quantile(abs(lm2),.25), hi=quantile(abs(lm2),.75)),
        by=list(train_size, lisa_cor_cut)][order(lisa_mean)],
    mean_errors[, list(model=3, lisa_mean=mean(lisa_cor),
            n=length(abs(lm3)), m=mean(abs(lm3)), lo=quantile(abs(lm3),.25), hi=quantile(abs(lm3),.75)),
        by=list(train_size, lisa_cor_cut)][order(lisa_mean)],
    mean_errors[, list(model=4, lisa_mean=mean(lisa_cor),
            n=length(abs(lm4)), m=mean(abs(lm4)), lo=quantile(abs(lm4),.25), hi=quantile(abs(lm4),.75)),
        by=list(train_size, lisa_cor_cut)][order(lisa_mean)],
    mean_errors[, list(model=5, lisa_mean=mean(lisa_cor),
            n=length(abs(lm5)), m=mean(abs(lm5)), lo=quantile(abs(lm5),.25), hi=quantile(abs(lm5),.75)),
        by=list(train_size, lisa_cor_cut)][order(lisa_mean)]
    ))

ggplot(lisa_errors[train_size==75], aes(lisa_mean, m, colour=as.factor(model))) +
    geom_point() +
    geom_smooth(method='lm', se=FALSE, linetype='dashed') +
    scale_colour_discrete(name='model') +
    theme_minimal()

ggsave(sprintf('../figs/%s/wealth_lm_lisa_abs_errors.png', loc_tag), width=6, height=4)


lisa_errors[, ':='(has_pd=model %in% c(0,2,4,5), has_lag=model %in% c(1,5), has_cdr=model %in% 3:5)]

plt <- ggplot(lisa_errors[train_size==95], aes(lisa_mean, m, group=as.factor(model))) +
    geom_line(aes(colour=has_cdr, linetype=has_pd)) +
    geom_point(aes(shape=has_lag), solid=FALSE) +
    xlab('LISA') + ylab('mean absolute error') +
    scale_colour_discrete(name='CDR') +
    scale_linetype(name='Pop') +
    scale_shape_discrete(name='lag', solid=FALSE) +
    theme_minimal()

ggsave(sprintf('../figs/%s/wealth_lm_lisa_abs_errors.png', loc_tag), plot=plt, height=6, width=10)

#
