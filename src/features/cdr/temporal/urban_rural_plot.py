# Jack Shipway October 2016 UCL GCRF Research Project
#
# This file
#

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import sys
import scipy

MONTHS = ['dec', 'jan', 'feb', 'mar', 'apr']

if __name__ == "__main__":

    antennae = np.setdiff1d(range(1237), [0, 573, 749, 1061, 1200, 1205, 1208, 1213])

    rainfall = pd.DataFrame(pd.read_csv("IvoryCoastData/Geography/ICRainfall11-12.csv"), index=None)

    # ''' Get data by: total '''
    # total_morn = np.load("IvoryCoastData/CDR/AggregateData/total_morn.npy")
    # total_aft = np.load("IvoryCoastData/CDR/AggregateData/total_aft.npy")
    # total_eve = np.load("IvoryCoastData/CDR/AggregateData/total_eve.npy")
    # total_night = np.load("IvoryCoastData/CDR/AggregateData/total_night.npy")
    total_w = np.load("IvoryCoastData/CDR/AggregateData/total_working.npy")
    total_nw = np.load("IvoryCoastData/CDR/AggregateData/total_nworking.npy")

    # Work out which nodes 'switch on', do any switch off?..
    total_urban_w = total_w[:, :, 0:140]
    total_rural_w = total_w[:, :, 0:140]
    avg_std1 = np.std(total_urban_w, axis=2)
    avg_std2 = np.std(total_rural_w, axis=2)
    a = np.where(avg_std1[0] < 50)
    b = np.where(avg_std2[0] < 50)
    urban_rural = pd.DataFrame(pd.read_csv('IvoryCoastData/CDR/antennaUrbanRural.csv', sep=','))

    urban = urban_rural[:][urban_rural['Class'] == 'U']
    urban_id = urban['AntennaID']
    rural = urban_rural[:][urban_rural['Class'] == 'R']
    rural_id = rural['AntennaID']

    # Urban versus rural antennae working hours
    total_w_urban = total_w[:, np.setdiff1d([urban_id - 1], np.append([-1], a[0])), :]
    total_w_rural = total_w[:, np.setdiff1d([rural_id - 1], b[0]), :]

    # Urban versus rural antennae non-working hours]
    total_nw_urban = total_nw[:, np.setdiff1d([urban_id - 1], [-1, 0, 573, 749, 1061, 1200, 1205, 1208, 1213]), :]
    total_nw_rural = total_nw[:, rural_id - 1, :]

    mean_w_urban = []
    mean_w_rural =[]
    mean_nw_urban = []
    mean_nw_rural = []
    
    std_w_urban = []
    std_w_rural = []
    std_nw_urban = []
    std_nw_rural = []

    for i in range(140):
        mean_w_u = scipy.stats.gmean(np.log(total_w_urban[0, :, i][total_w_urban[0, :, i] > 0] + 1))
        mean_w_urban.append(mean_w_u)

        mean_w_r = scipy.stats.gmean(np.log(total_w_rural[0, :, i][total_w_rural[0, :, i] > 0] + 1))
        mean_w_rural.append(mean_w_r)

        mean_nw_u = scipy.stats.gmean(np.log(total_nw_urban[0, :, i][total_nw_urban[0, :, i] > 0] + 1))
        mean_nw_urban.append(mean_nw_u)

        mean_nw_r = scipy.stats.gmean(np.log(total_nw_rural[0, :, i][total_nw_rural[0, :, i] > 0] + 1))
        mean_nw_rural.append(mean_nw_r)

        std_w_u = np.std(np.log((total_w_urban[0, :, i][total_w_urban[0, :, i] > 0]) + 1))
        std_w_urban.append(std_w_u)

        std_w_r = np.std(np.log(total_w_rural[0, :, i][total_w_rural[0, :, i] > 0] + 1))
        std_w_rural.append(std_w_r)

        std_nw_u = np.std(np.log((total_nw_urban[0, :, i][total_nw_urban[0, :, i] > 0]) + 1))
        std_nw_urban.append(std_nw_u)

        std_nw_r = np.std(np.log(total_nw_rural[0, :, i][total_nw_rural[0, :, i] > 0] + 1))
        std_nw_rural.append(std_nw_r)

    print np.mean(std_w_urban[0:135]), np.mean(std_nw_urban[0:135])
    print np.mean(std_w_rural[0:135]), np.mean(std_nw_rural[0:135])

    # plt.figure(1, facecolor='white')
    #
    # plt.subplot(1, 2, 1)
    # plt.errorbar(range(140), mean_w_urban, yerr=std_w_urban, label='Urban Output: Working Hours')
    # plt.errorbar(range(140), mean_w_rural, yerr=std_w_rural, label='Rural Output: Working Hours')
    # plt.bar(range(0, 140, 28), rainfall['rainfall'] * 0.05, color='blue', width=28, alpha=0.1, label='Rainfall')
    # plt.xlim(0, 140)
    # plt.ylim(0, 10)
    # plt.legend(loc='lower right')
    # plt.grid()
    #
    # plt.subplot(1, 2, 2)
    #
    # plt.errorbar(range(140), mean_nw_rural, yerr=std_nw_rural, label='Rural Output: Non-working Hours')
    # plt.errorbar(range(140), mean_nw_urban, yerr=std_nw_urban, label='Urban Output: Non-working Hours')
    # plt.bar(range(0, 140, 28), rainfall['rainfall'] * 0.05, color='blue', width=28, alpha=0.1, label='Rainfall')
    # plt.xlim(0, 140)
    # plt.ylim(0, 10)
    # plt.legend(loc='lower right')
    # plt.grid()
    # plt.show()
    
    
    # ''' by month '''
    # dec_mean_urban_w = np.mean(mean_w_urban[0:28])
    # dec_std_urban_w = np.mean(std_w_urban[0:28])
    # dec_mean_urban_nw = np.mean(mean_nw_urban[0:28])
    # dec_std_urban_nw = np.mean(std_nw_urban[0:28])
    # dec_mean_rural_w = np.mean(mean_w_rural[0:28])
    # dec_std_rural_w = np.mean(std_w_rural[0:28])
    # dec_mean_rural_nw = np.mean(mean_nw_rural[0:28])
    # dec_std_rural_nw = np.mean(std_nw_rural[0:28])
    #
    # jan_mean_urban_w = np.mean(mean_w_urban[28:56])
    # jan_std_urban_w = np.mean(std_w_urban[28:56])
    # jan_mean_urban_nw = np.mean(mean_nw_urban[28:56])
    # jan_std_urban_nw = np.mean(std_nw_urban[28:56])
    # jan_mean_rural_w = np.mean(mean_w_rural[28:56])
    # jan_std_rural_w = np.mean(std_w_rural[28:56])
    # jan_mean_rural_nw = np.mean(mean_nw_rural[28:56])
    # jan_std_rural_nw = np.mean(std_nw_rural[28:56])
    #
    # feb_mean_urban_w = np.mean(mean_w_urban[56:84])
    # feb_std_urban_w = np.mean(std_w_urban[56:84])
    # feb_mean_urban_nw = np.mean(mean_nw_urban[56:84])
    # feb_std_urban_nw = np.mean(std_nw_urban[56:84])
    # feb_mean_rural_w = np.mean(mean_w_rural[56:84])
    # feb_std_rural_w = np.mean(std_w_rural[56:84])
    # feb_mean_rural_nw = np.mean(mean_nw_rural[56:84])
    # feb_std_rural_nw = np.mean(std_nw_rural[56:84])
    #
    # mar_mean_urban_w = np.mean(mean_w_urban[84:112])
    # mar_std_urban_w = np.mean(std_w_urban[84:112])
    # mar_mean_urban_nw = np.mean(mean_nw_urban[84:112])
    # mar_std_urban_nw = np.mean(std_nw_urban[84:112])
    # mar_mean_rural_w = np.mean(mean_w_rural[84:112])
    # mar_std_rural_w = np.mean(std_w_rural[84:112])
    # mar_mean_rural_nw = np.mean(mean_nw_rural[84:112])
    # mar_std_rural_nw = np.mean(std_nw_rural[84:112])
    #
    # apr_mean_urban_w = np.mean(mean_w_urban[112:136])
    # apr_std_urban_w = np.mean(std_w_urban[112:136])
    # apr_mean_urban_nw = np.mean(mean_nw_urban[112:136])
    # apr_std_urban_nw = np.mean(std_nw_urban[112:136])
    # apr_mean_rural_w = np.mean(mean_w_rural[112:136])
    # apr_std_rural_w = np.mean(std_w_rural[112:136])
    # apr_mean_rural_nw = np.mean(mean_nw_rural[112:136])
    # apr_std_rural_nw = np.mean(std_nw_rural[112:136])
    #
    # monthly_urban_w_mean = [dec_mean_urban_w, jan_mean_urban_w, feb_mean_urban_w, mar_mean_urban_w, apr_mean_urban_w]
    # monthly_urban_nw_mean = [dec_mean_urban_nw, jan_mean_urban_nw, feb_mean_urban_nw, mar_mean_urban_nw, apr_mean_urban_nw]
    # monthly_rural_w_mean = [dec_mean_rural_w, jan_mean_rural_w, feb_mean_rural_w, mar_mean_rural_w, apr_mean_rural_w]
    # monthly_rural_nw_mean = [dec_mean_rural_nw, jan_mean_rural_nw, feb_mean_rural_nw, mar_mean_rural_nw, apr_mean_rural_nw]
    #
    # monthly_urban_w_std = [dec_std_urban_w, jan_std_urban_w, feb_std_urban_w, mar_std_urban_w, apr_std_urban_w]
    # monthly_urban_nw_std = [dec_std_urban_nw, jan_std_urban_nw, feb_std_urban_nw, mar_std_urban_nw, apr_std_urban_nw]
    # monthly_rural_w_std = [dec_std_rural_w, jan_std_rural_w, feb_std_rural_w, mar_std_rural_w, apr_std_rural_w]
    # monthly_rural_nw_std = [dec_std_rural_nw, jan_std_rural_nw, feb_std_rural_nw, mar_std_rural_nw, apr_std_rural_nw]
    #
    # # plt.figure(facecolor='white')
    # # plt.errorbar(range(1, 11, 2), monthly_urban_w_mean, yerr=monthly_urban_w_std, label='Urban: Working Hours')
    # # plt.errorbar(range(1, 11, 2), monthly_urban_nw_mean, yerr=monthly_urban_nw_std, label='Urban: Non-working Hours')
    # # plt.errorbar(range(1, 11, 2), monthly_rural_w_mean, yerr=monthly_rural_w_std, label='Rural: Working Hours')
    # # plt.errorbar(range(1, 11, 2), monthly_rural_nw_mean, yerr=monthly_rural_nw_std, label='Rural: Non-working Hours')
    # # plt.bar(range(0,10,2), rainfall['rainfall']*0.08, color='blue', width=2, alpha=0.1, label='Rainfall')
    # # plt.grid()
    # # plt.xlim(0, 10)
    # # plt.legend(loc='upper left')
    # # plt.show()
    #
    #
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    