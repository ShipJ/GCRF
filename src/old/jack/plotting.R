install.packages("/Users/JackShipway/Downloads/ncf", repos = NULL, type="source", dependencies = TRUE)

library(ncf) # correlog, lisa
library(plyr) # .()

#source('R/binned_scatter_plot.R')
#source('R/ggplot_blank_theme.R')

circle <- function(center=c(0,0), radius=1, npoints=100){
    tt <- seq(0,2*pi,length.out = npoints)
    xx <- center[1] + radius * cos(tt)
    yy <- center[2] + radius * sin(tt)
    return(data.table(x = xx, y = yy))
}

loc_tag <- 'sen'
# loc_tag <- 'civ'


# --------------
# DHS
# --------------
# sen <- fread(sprintf('/Users/JackShipway/Desktop/UCLProject/Python/Results/results.csv'))
# sen[, country:='Senegal']
civ <- fread(sprintf('/Users/JackShipway/Desktop/UCLProject/Python/Results/results.csv'))
civ[, country:="Côte d'Ivoire"]

# I modified this: dhs <- rbindlist(list(sen, civ)), to:
dhs <- rbindlist(list(civ))

# plot wealth distribution
plt <- ggplot(dhs) +
    geom_bar(aes(z_median, (..count..)/sum(..count..)), binwidth=.1, fill='#4D88CC') +
    xlab('wealth') + ylab('P(wealth)') +
    facet_wrap(~country) +
    theme_minimal() +
    theme(strip.text=element_text(size=20),
          axis.title=element_text(size=20),
          axis.text=element_text(size=14))
ggsave('figs/dhs/wealth_index_hists.png', plot=plt, width=10, height=4)

plt <- ggplot(dhs) +
    geom_bar(aes(poverty_rate, (..count..)/sum(..count..)), binwidth=.04, fill='#4D88CC') +
    xlab('poverty intensity') + ylab('P(poverty intensity)') +
    facet_wrap(~country) +
    theme_minimal() +
    theme(strip.text=element_text(size=20),
          axis.title=element_text(size=20),
          axis.text=element_text(size=14))
ggsave('figs/dhs/poverty_rate_hists.png', plot=plt, width=10, height=4)

# plot wealth vs population density
plt <- ggplot(dhs) + geom_point(aes(pop_1km, z_median, colour=type), size=3) +
            #, shape=ifelse(capital, 'Capital', 'Other')
            #, colour='#4D88CC') +
    scale_x_log10() +
    # scale_shape_manual(values=c(1,16)) +
    xlab('population density') + ylab('wealth') +
    facet_wrap(~country) +
    theme_minimal() +
    theme(legend.title=element_blank(),
          strip.text=element_text(size=20),
          axis.title=element_text(size=20),
          axis.text=element_text(size=14))
ggsave('figs/dhs/wealth_vs_pop_cols.png', plot=plt, width=10, height=4)

plt <- ggplot(dhs) + geom_point(aes(pop_1km, poverty_rate, colour=type), size=3) +
        # shape=ifelse(capital, 'Capital', 'Other'))) + #, colour='#4D88CC') +
    scale_x_log10() +
    # scale_shape_manual(values=c(1,16)) +
    xlab('population density') + ylab('poverty intensity') +
    facet_wrap(~country) +
    theme_minimal() +
    theme(legend.title=element_blank(),
          strip.text=element_text(size=20),
          axis.title=element_text(size=20),
          axis.text=element_text(size=14))
ggsave('figs/dhs/poverty_rate_vs_pop_cols.png', plot=plt, width=10, height=4)


# SPATIAL
utm <- ifelse(loc_tag=='sen', 28, 30)
geom <- fread(sprintf('%s/%s/geo/%s_adm/%s_adm0_utm%d_geom.csv',
            data_dir, loc_tag, toupper(loc_tag), loc_tag, utm))
cap_geom <- fread(sprintf('%s/%s/geo/%s_adm/capital_region.csv',
            data_dir, loc_tag, toupper(loc_tag)))
# map points
plt <- ggplot(dhs) +
    geom_path(data=geom, aes(long,lat,group=group), colour='grey40', size=.5) +
    geom_point(aes(x, y, colour=z_median), size=2.5, alpha=.8) +
    scale_colour_gradientn(name='Wealth', colours=rainbow(3),
        limits=dhs[,range(z_median)], breaks=dhs[,range(z_median)*.9],
        labels=c('poorest', 'richest')) +
    coord_equal() + blank_theme +
    theme(plot.margin=unit(c(1,1,1,1), 'mm'),
          legend.title=element_text(size=16, face='plain'),
          legend.text=element_text(size=10))
ggsave(sprintf('figs/%s/dhs/cluster_wealth.png',loc_tag), plot=plt, width=6, height=4)

plt <- ggplot(dhs[capital==TRUE]) +
    geom_polygon(data=cap_geom, aes(long,lat,group=group), fill=NA, colour='grey40', size=.2) +
    geom_point(aes(x, y, colour=z_median), size=3) +
    scale_colour_gradientn(name='Wealth', colours=rainbow(3),
        limits=dhs[,range(z_median)], breaks=dhs[,range(z_median)],
        labels=c('poorest', 'richest')) +
    coord_equal() + blank_theme
ggsave(sprintf('figs/%s/dhs/capital_cluster_wealth.png',loc_tag), plot=plt, width=10, height=7.5)

plt <- ggplot(dhs) +
    geom_polygon(data=geom, aes(long,lat,group=group), fill=NA, colour='grey40', size=.2) +
    geom_point(aes(x, y, colour=poverty_rate)) +
    scale_colour_gradientn(name='Poverty\nIntensity', colours=rev(rainbow(3)),
        limits=dhs[,range(poverty_rate)], breaks=dhs[,range(poverty_rate)],
        labels=c('lowest', 'highest')) +
    coord_equal() + blank_theme
ggsave(sprintf('figs/%s/dhs/cluster_poverty_intensity.png',loc_tag),
    plot=plt, width=10, height=7.5)

plt <- ggplot(dhs[capital==TRUE]) +
    geom_polygon(data=cap_geom, aes(long,lat,group=group), fill=NA, colour='grey40', size=.2) +
    geom_point(aes(x, y, colour=poverty_rate), size=3) +
    scale_colour_gradientn(name='Poverty\nIntensity', colours=rev(rainbow(3)),
        limits=dhs[,range(poverty_rate)], breaks=dhs[,range(poverty_rate)],
        labels=c('lowest', 'highest')) +
    coord_equal() + blank_theme
ggsave(sprintf('figs/%s/dhs/capital_cluster_poverty_intensity.png',loc_tag),
    plot=plt, width=10, height=7.5)


# correlograms
sen <- fread(sprintf('%s/sen/dhs_cluster_wealth.csv', data_dir))
sen_cgram <- sen[, correlog(x, y, z_median, increment=2000)]
sen <- data.table(country='Senegal', dist=sen_cgram$mean.of.class,
    cor=sen_cgram$correlation, n=sen_cgram$n, p=sen_cgram$p,
    x.intercept=sen_cgram$x.intercept)

civ <- fread(sprintf('%s/civ/dhs_cluster_wealth.csv', data_dir))
civ_cgram <- civ[, correlog(x, y, z_median, increment=2000)]
civ <- data.table(country="Côte d'Ivoire", dist=civ_cgram$mean.of.class,
    cor=civ_cgram$correlation, n=civ_cgram$n, p=civ_cgram$p,
    x.intercept=civ_cgram$x.intercept)

dhs_cgram <- rbindlist(list(sen, civ))

plt <- ggplot(dhs_cgram[dist<150000], aes(dist/1000, cor)) +
    # geom_point(data=cgram_sm_df[dist<100000&n>4], colour='#DD4B39', alpha=.6, size=1) +
    geom_line(colour='#4D88CC', size=1) +
    geom_vline(aes(xintercept=x.intercept/1000), colour='#4D88CC', linetype='dashed') +
    geom_text(aes(x=(x.intercept+12000)/1000, y=1.1,
        label=c(sprintf('%.0f km', x.intercept/1000))), size=4) +
    # geom_vline(xintercept=cgram_sm$x.intercept, colour='#DD4B39', linetype='dashed') +
    # geom_text(data=data.frame(x=cgram_sm$x.intercept+1000, y=1.1,
    #     label=c(sprintf('%.0f km',cgram_sm$x.intercept))), aes(x,y,label=label), size=4) +
    xlab('distance (km)') + ylab('Moran\'s I') +
    facet_wrap(~country) +
    theme_minimal() +
    theme(strip.text=element_text(size=20),
          axis.title=element_text(size=20),
          axis.text=element_text(size=14))
ggsave('figs/dhs/wealth_correlograms.png', plot=plt, width=10, height=4)

# plt <- ggplot(cgram_df[dist<200000], aes(dist, n)) + geom_line(colour='#4D88CC') +
#     ylab('p-value') + xlab('Moran\'s I') + theme_minimal()
# ggsave(sprintf('figs/%s/dhs/p_vs_correlogram_wealth.png',loc_tag), plot=plt, width=10, height=7.5)


cgram_lg <- dhs[, correlog(x, y, poverty_rate, increment=2000)]
# cgram_sm <- dhs[, correlog(x, y, poverty_rate, increment=100)]
cgram_lg_df <- data.table(dist=cgram_lg$mean.of.class,
    cor=cgram_lg$correlation, n=cgram_lg$n, p=cgram_lg$p)
# cgram_sm_df <- data.table(dist=cgram_sm$mean.of.class,
    cor=cgram_sm$correlation, n=cgram_sm$n, p=cgram_sm$p)

plt <- ggplot(cgram_lg_df[dist<150000], aes(dist/1000, cor)) +
    # geom_point(data=cgram_sm_df[dist<100000&n>4], colour='#DD4B39', alpha=.6, size=1) +
    geom_line(colour='#4D88CC', size=1) +
    geom_vline(xintercept=cgram_lg$x.intercept/1000, colour='#4D88CC', linetype='dashed') +
    geom_text(data=data.frame(x=(cgram_lg$x.intercept+12000)/1000, y=1.1,
        label=c(sprintf('%.0f km', cgram_lg$x.intercept/1000))), aes(x,y,label=label), size=4) +
    # geom_vline(xintercept=cgram_sm$x.intercept, colour='#DD4B39', linetype='dashed') +
    # geom_text(data=data.frame(x=cgram_sm$x.intercept+1000, y=1.1,
    #     label=c(sprintf('%.0f km',cgram_sm$x.intercept))), aes(x,y,label=label), size=4) +
    xlab('distance (km)') + ylab('Moran\'s I') + theme_minimal()
ggsave(sprintf('figs/%s/dhs/correlogram_poverty.png',loc_tag),
    plot=plt, width=10, height=7.5)


# LISA
dhs <- fread(sprintf('%s/%s/dhs_cluster_wealth.csv', data_dir, loc_tag))
utm <- ifelse(loc_tag=='sen', 28, 30)
geom <- fread(sprintf('%s/%s/geo/%s_adm/%s_adm0_utm%d_geom.csv',
            data_dir, loc_tag, toupper(loc_tag), loc_tag, utm))
cap_geom <- fread(sprintf('%s/%s/geo/%s_adm/capital_region.csv',
            data_dir, loc_tag, toupper(loc_tag)))
r <- ifelse(loc_tag=='civ', 29000, 64000)

lis <- dhs[, lisa(x, y, z_median, neigh=r)]
dhs <- cbind(dhs, data.table(mean_dist=lis$dmean, cor=lis$correlation,
                                n=lis$n, p=lis$p))
sig_level <- .05
sig_prop <- nrow(dhs[p<=sig_level]) / nrow(dhs)

exp_point1 <- ifelse(loc_tag=='civ', 265, 300)

plt <- ggplot(dhs) +
    geom_path(data=geom, aes(long,lat,group=group), colour='grey40', size=.5) +
    geom_path(data=circle(dhs[clust_id==exp_point1,c(x,y)], r), aes(x, y), size=.4, linetype='dashed') +
    geom_point(aes(x, y, size=abs(cor), colour=cor<=0, shape=p<sig_level)) +
    # subset insures negative cors are plotted on top
    geom_point(aes(x, y, size=abs(cor), colour=cor<0, shape=p<sig_level), subset=.(cor<=0)) +
    scale_size_continuous(limits=c(0,dhs[,max(cor)]), breaks=c(0,1,2,3), na.value=1.3) +
    scale_colour_manual(labels=c('+ve','-ve'), values=c('#FC291C','#4D88CC'), na.value='grey40') +
    scale_shape_manual(labels=c('p<.05','p>.05'), breaks=c(TRUE,FALSE),
        values=c(1,16), na.value=16) +
    coord_equal() + blank_theme +
    theme(plot.margin=unit(c(1,1,1,1), 'mm'),
          legend.title=element_blank(),
          legend.text=element_text(size=10))

ggsave(sprintf('figs/%s/dhs/lisa_wealth.png',loc_tag, r/1000), plot=plt, width=6, height=4)

exp_point2 <- ifelse(loc_tag=='civ', 265, 220)
plt <- ggplot(dhs[capital==TRUE]) +
    geom_path(data=cap_geom, aes(long,lat,group=group), colour='grey40', size=.2) +
    geom_point(aes(x, y, size=abs(cor), colour=cor<0)) +
    # geom_path(data=circle(dhs[clust_id==exp_point2,c(x,y)], r), aes(x, y), size=.2, linetype='dashed') +
    scale_size_continuous(limits=c(0,dhs[,max(cor)]), breaks=c(0,1,2,3)) +
    scale_colour_manual(labels=c('+ve','-ve'), values=c('#FC291C','#4D88CC')) +
    coord_equal() + blank_theme + theme(legend.title=element_blank())

ggsave(sprintf('figs/%s/dhs/capital_lisa_wealth.png',loc_tag, r/1000), plot=plt, width=10, height=7.5)



lis <- dhs[, lisa(x, y, poverty_rate, neigh=r)]
dhs[, ':='(mean_dist=lis$dmean, cor=lis$correlation, n=lis$n, p=lis$p)]
sig_level <- .05
sig_prop <- nrow(dhs[p<=sig_level]) / nrow(dhs)

exp_point1 <- ifelse(loc_tag=='civ', 265, 300)

plt <- ggplot(dhs) +
    geom_path(data=geom, aes(long,lat,group=group), colour='grey40', size=.2) +
    geom_path(data=circle(dhs[clust_id==exp_point1,c(x,y)], r), aes(x, y), size=.2, linetype='dashed') +
    geom_point(aes(x, y, size=abs(cor), colour=cor<=0, shape=p<sig_level)) +
    # subset insures negative cors are plotted on top
    geom_point(aes(x, y, size=abs(cor), colour=cor<0, shape=p<sig_level), subset=.(cor<=0)) +
    scale_size_continuous(limits=c(0,dhs[,max(cor)]), breaks=c(0,1,2,3), na.value=1.3) +
    scale_colour_manual(labels=c('+ve','-ve'), values=c('#FC291C','#4D88CC'), na.value='grey40') +
    scale_shape_manual(labels=c('Signif.','Not signif.'), breaks=c(TRUE,FALSE),
        values=c(1,16), na.value=16) +
    coord_equal() + blank_theme + theme(legend.title=element_blank())

ggsave(sprintf('figs/%s/dhs/lisa_poverty.png', loc_tag, r/1000), plot=plt, width=10, height=7.5)

exp_point2 <- ifelse(loc_tag=='civ', 265, 220)
plt <- ggplot(dhs[capital==TRUE]) +
    geom_path(data=cap_geom, aes(long,lat,group=group), colour='grey40', size=.2) +
    geom_point(aes(x, y, size=abs(cor), colour=cor<0)) +
    # geom_path(data=circle(dhs[clust_id==exp_point2,c(x,y)], r), aes(x, y), size=.2, linetype='dashed') +
    scale_size_continuous(limits=c(0,dhs[,max(cor)]), breaks=c(0,1,2,3)) +
    scale_colour_manual(labels=c('+ve','-ve'), values=c('#FC291C','#4D88CC')) +
    coord_equal() + blank_theme + theme(legend.title=element_blank())

ggsave(sprintf('figs/%s/dhs/capital_lisa_poverty.png',loc_tag, r/1000), plot=plt, width=10, height=7.5)

# ------
# MODELS
# ------


order_labels <- function(df, pred_labs) {
    df[, label:=factor(label, levels=pred_labs)]
    return (NULL)
}

combine_scores <- function(loc_tag, pred_tags, pred_labs) {
    rmse <- data.table(NULL)
    mad <- data.table(NULL)
    sens <- data.table(NULL)
    spec <- data.table(NULL)
    i <- 1
    for (pred_tag in pred_tags) {
        scr <- fread(sprintf('%s/%s/model_data/dhs_rgr_%s_scores.csv', data_dir, loc_tag, pred_tag))
        train_rmse <- scr[, list(
                label=pred_labs[i],
                stage='train',
                min_=min(train_rmse),
                quar1=quantile(train_rmse, .25),
                median_=median(train_rmse),
                quar3=quantile(train_rmse, .75),
                max_=max(train_rmse)),
            by=p]
        test_rmse <- scr[, list(
                label=pred_labs[i],
                stage='test',
                min_=min(test_rmse),
                quar1=quantile(test_rmse, .25),
                median_=median(test_rmse),
                quar3=quantile(test_rmse, .75),
                max_=max(test_rmse)),
            by=p]
        rmse <- rbindlist(list(rmse, train_rmse, test_rmse))
        rm(train_rmse); rm(test_rmse)

        train_mad <- scr[, list(
            label=pred_labs[i],
            stage='train',
            min_=min(train_mad),
            quar1=quantile(train_mad, .25),
            median_=median(train_mad),
            quar3=quantile(train_mad, .75),
            max_=max(train_mad)),
        by=p]
        test_mad <- scr[, list(
                label=pred_labs[i],
                stage='test',
                min_=min(test_mad),
                quar1=quantile(test_mad, .25),
                median_=median(test_mad),
                quar3=quantile(test_mad, .75),
                max_=max(test_mad)),
            by=p]
        mad <- rbindlist(list(mad, train_mad, test_mad))
        rm(train_mad); rm(test_mad)

        scr <- fread(sprintf('%s/%s/model_data/dhs_clf_%s_scores.csv', data_dir, loc_tag, pred_tag))
        train_sens <- scr[, list(
                label=pred_labs[i],
                stage='train',
                min_=min(train_sens),
                quar1=quantile(train_sens, .25),
                median_=median(train_sens),
                quar3=quantile(train_sens, .75),
                max_=max(train_sens)),
            by=p]
        test_sens <- scr[, list(
                label=pred_labs[i],
                stage='test',
                min_=min(test_sens),
                quar1=quantile(test_sens, .25),
                median_=median(test_sens),
                quar3=quantile(test_sens, .75),
                max_=max(test_sens)),
            by=p]
        sens <- rbindlist(list(sens, train_sens, test_sens))
        rm(train_sens); rm(test_sens)

        train_spec <- scr[, list(
                label=pred_labs[i],
                stage='train',
                min_=min(train_spec),
                quar1=quantile(train_spec, .25),
                median_=median(train_spec),
                quar3=quantile(train_spec, .75),
                max_=max(train_spec)),
            by=p]
        test_spec <- scr[, list(
                label=pred_labs[i],
                stage='test',
                min_=min(test_spec),
                quar1=quantile(test_spec, .25),
                median_=median(test_spec),
                quar3=quantile(test_spec, .75),
                max_=max(test_spec)),
            by=p]
        spec <- rbindlist(list(spec, train_spec, test_spec))
        rm(train_spec); rm(test_spec)
        i <- i + 1
    }

    order_labels(rmse, pred_labs)
    order_labels(mad, pred_labs)
    order_labels(sens, pred_labs)
    order_labels(spec, pred_labs)

    list(rmse, mad, sens, spec)
}


plot_train_test <- function(df, y_lab, pred_tag, score_tag) {
    plt <- ggplot(df, aes(p, median_, fill=stage)) +
        geom_ribbon(aes(ymin=quar1, ymax=quar3), alpha=.3) +
        geom_line(aes(colour=stage)) +
        # geom_line(aes(y=min_, colour=stage), linetype='dotted') +
        # geom_line(aes(y=max_, colour=stage), linetype='dotted') +
        scale_colour_manual(labels=c('Test', 'Train'), values=c('#FC291C','#4D88CC')) +
        scale_fill_manual(labels=c('Test', 'Train'), values=c('#FC291C','#4D88CC')) +
        scale_x_continuous(minor_breaks=seq(0, 100, 5)) +
        xlab('% training') + ylab(y_lab) +
        facet_wrap(~label) +
        theme_minimal() + theme(legend.title=element_blank())
    ggsave(sprintf('figs/%s/%s_%s.png', loc_tag, pred_tag, tolower(substr(y_lab, 1, 4))), plot=plt, width=12, height=5)
    plt
}

plot_scores <- function(df, y_lab, stage_='test') {
    plt <- ggplot(sens[stage==stage_], aes(p, median_, colour=label)) +
        geom_line() + xlab('% Training') + ylab(y_lab) +
        scale_colour_discrete('Model') +
        theme_minimal() + theme(legend.position='top')
    ggsave(sprintf('figs/%s/%s_%s.png', loc_tag, tolower(substr(y_lab, 1, 4)), stage_),
        plot=plt, width=10, height=7.5)
    plt
}

pred_tags <- c('pd', 'lag', 'pdlag')
pred_labs <- c('PD', 'Lag', 'PD and Lag')
bline_scrs <- combine_scores('sen', pred_tags, pred_labs)

plot_train_test(bline_scrs[[1]], 'RMSE', 'rmse')
plot_train_test(bline_scrs[[2]], 'MAD', 'mad')
plot_train_test(bline_scrs[[3]], 'Sensitivity', 'sens')
plot_train_test(bline_scrs[[4]], 'Specificity', 'spec')

pred_tags <- c('cdr', 'cdrpd', 'cdrpdlag')
pred_labs <- c('CDR', 'CDR and PD', 'CDR, PD and Lag')
cdr_scrs <- combine_scores('sen', pred_tags, pred_labs)
plot_train_test(cdr_scrs[[2]], 'MAD', 'cdr', 'mad')
plot_train_test(cdr_scrs[[3]], 'Sensitivity', 'cdr', 'sens')
plot_train_test(cdr_scrs[[4]], 'Specificity', 'cdr', 'spec')

scr_labs <- c('RMSE', 'MAD', 'Sensitivity', 'Specificity')
for (i in 1:4) {
    plot_train_test(cdr_scrs[[i]], scr_labs[i], 'cdr')
    plot_scores(rbindlist(list(bline_scrs[[i]], cdr_scrs[[i]])), scr_labs[i], 'test')
}
# ----------------
# FLOWS
# ----------------
flows <- fread(sprintf('%s/%s/cdr_data/flows/bts0_flows.csv', data_dir, loc_tag))
flows_internal <- flows[a==b]
flows <- flows[a!=b]
# random sample
ix <- sample(1:nrow(flows), 10000, replace=FALSE)

# plot pairwise vol distribution
plt <- ggplot(flows[ix]) + geom_bar(aes(x=vol, y=(..count..)/sum(..count..)), fill='grey40') +
    scale_x_log10() +
    xlab('v') + ylab('P(v)') + theme_minimal()
ggsave(sprintf('figs/%s/bts_volume_hist.png', loc_tag), plot=plt, width=10, height=7.5)

# plot distance distribution
plt <- ggplot(flows[ix]) + geom_bar(aes(x=dist, y=(..count..)/sum(..count..)), fill='grey40') +
    xlab('d') + ylab('P(d)') + theme_minimal()
ggsave(sprintf('figs/%s/bts_distance_hist.png',loc_tag), plot=plt, width=10, height=7.5)

# plot volumn vs distance
vd <- bin_data(flows, 5000, c('dist','vol'), logx=TRUE)
plt <- plot_bin_data_points(vd, c('distance','volume'))
ggsave(sprintf('figs/%s/bts_dist_vs_vol.png', loc_tag), plot=plt, width=10, height=7.5)








