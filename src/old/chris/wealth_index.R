 # install.packages('rgdal', type = "source", configure.args=c('--with-proj-include=/usr/local/include','--with-proj-lib=/usr/local/lib'))

library(data.table)
library(foreign)
library(ggplot2)
library(rgdal)
# library(devtools)

# source_url('https://gist.githubusercontent.com/chri55c/ae3181def8012166076a/raw/c55b95eae837944fdd7ca016df7b246e368fd922/ggplot_blank_theme.R')

# recode manual from https://dhsprogram.com/publications/publication-DHSG4-DHS-Questionnaires-and-Manuals.cfm
get_wealth_index <- function(hr_file, gr_path, ge_file, out_file, zone) {
    dhs <- read.spss(hr_file, to.data.frame=TRUE)
    dhs <- data.table(dhs[, c('HHID','HV001','HV005','HV024','HV025','HV270','HV271')])
    setnames(dhs, c('household_id','clust_id','sample_weight',
                    'region','type','wealth_quintile','wealth'))
    dhs[, poor:=ifelse(wealth_quintile=='Poorest', 1, 0)]
    clust.sp <- readOGR(ge_path, ge_file)
    clust.sp <- subset(clust.sp, LONGNUM != 0 & LATNUM != 0)
    if (zone == 28) {
        clust.sp <- spTransform(clust.sp,
            CRS("+proj=utm +zone=28 +a=6378249.2 +b=6356515 +units=m +no_defs"))
    } else if (zone == 30){
        clust.sp <- spTransform(clust.sp,
            CRS("+proj=utm +zone=30 +ellps=clrk80 +towgs84=-124.76,53,466.79,0,0,0,0 +units=m +no_defs"))
    }
    clust_geom <- data.table(cbind(clust.sp$DHSCLUST, coordinates(clust.sp)))
    setnames(clust_geom, c('clust_id', 'x', 'y'))
    setkey(clust_geom, clust_id)
    clust <- dhs[, list(
            type=type[1],
            n=length(household_id),
            poor=sum(poor),
            min=min(wealth),
            mean=mean(wealth),
            median=median(wealth),
            max=max(wealth),
            sd=sd(wealth)),
        keyby=clust_id]
    clust[, ':='(poverty_rate=poor/n,
                 range=max-min,
                 z_median=(median-mean(median)) / sd(median))]
    clust <- clust[clust_geom]
    write.csv(clust, out_file, row.names=FALSE, quote=FALSE)
    clust
}

# ggplot(dhs[order(wealth_index), list(wealth_quintile, wealth_index)], aes(1:nrow(dhs), wealth_index, colour=wealth_quintile)) + geom_point()

# # stretch this out horizontally
# ggplot(dhs, aes(reorder(factor(cluster_id), wealth_index, median), wealth_index)) + geom_boxplot(outlier.size=.5) + guides(fill=FALSE) + theme(axis.ticks=element_blank(), axis.text=element_blank()) + xlab('cluster') + ylab('wealth index')


# clust_buf <- readOGR('sen_clust_buff/','sen_clust_buff')
# clust_buf <- fortify(clust_buf, region='clust_id')

# source('~/gists/proj4_strings.txt')
# adm0 <- readOGR('SEN_adm/','SEN_adm0')
# adm0 <- spTransform(adm0, CRS(epsg31028))
# adm0 <- fortify(adm0)

# ggplot(clust) +
#     geom_polygon(data=adm0, aes(long, lat, group=group), fill='grey80') +
#     geom_map(aes(map_id=clust_id, fill=mean.q), map=clust_buf) +
#     expand_limits(x=clust_buf$long, y=clust_buf$lat) +
#     coord_equal() + scale_fill_brewer('Wealth', palette='Spectral') + blank_theme

#
