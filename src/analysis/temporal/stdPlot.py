# Jack Shipway October 2016 UCL GCRF Research Project
#
# This file
#

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import sys

MONTHS = ['dec', 'jan', 'feb', 'mar', 'apr']

if __name__ == "__main__":

    ''' Get data by: total '''
    total_morn = np.load("IvoryCoastData/CDR/AggregateData/total_morn.npy")
    total_aft = np.load("IvoryCoastData/CDR/AggregateData/total_aft.npy")
    total_eve = np.load("IvoryCoastData/CDR/AggregateData/total_eve.npy")
    total_night = np.load("IvoryCoastData/CDR/AggregateData/total_night.npy")
    total_w = np.load("IvoryCoastData/CDR/AggregateData/total_working.npy")
    total_nw = np.load("IvoryCoastData/CDR/AggregateData/total_nworking.npy")

    ''' Get data by: month '''
    dec_morn = np.load("IvoryCoastData/CDR/AggregateData/dec_morn.npy")
    jan_morn = np.load("IvoryCoastData/CDR/AggregateData/jan_morn.npy")
    feb_morn = np.load("IvoryCoastData/CDR/AggregateData/feb_morn.npy")
    mar_morn = np.load("IvoryCoastData/CDR/AggregateData/mar_morn.npy")
    apr_morn = np.load("IvoryCoastData/CDR/AggregateData/apr_morn.npy")

    dec_aft = np.load("IvoryCoastData/CDR/AggregateData/dec_aft.npy")
    jan_aft = np.load("IvoryCoastData/CDR/AggregateData/jan_aft.npy")
    feb_aft = np.load("IvoryCoastData/CDR/AggregateData/feb_aft.npy")
    mar_aft = np.load("IvoryCoastData/CDR/AggregateData/mar_aft.npy")
    apr_aft = np.load("IvoryCoastData/CDR/AggregateData/apr_aft.npy")

    dec_eve = np.load("IvoryCoastData/CDR/AggregateData/dec_eve.npy")
    jan_eve = np.load("IvoryCoastData/CDR/AggregateData/jan_eve.npy")
    feb_eve = np.load("IvoryCoastData/CDR/AggregateData/feb_eve.npy")
    mar_eve = np.load("IvoryCoastData/CDR/AggregateData/mar_eve.npy")
    apr_eve = np.load("IvoryCoastData/CDR/AggregateData/apr_eve.npy")

    dec_night = np.load("IvoryCoastData/CDR/AggregateData/dec_night.npy")
    jan_night = np.load("IvoryCoastData/CDR/AggregateData/jan_night.npy")
    feb_night = np.load("IvoryCoastData/CDR/AggregateData/feb_night.npy")
    mar_night = np.load("IvoryCoastData/CDR/AggregateData/mar_night.npy")
    apr_night = np.load("IvoryCoastData/CDR/AggregateData/apr_night.npy")
    
    dec_w = np.load("IvoryCoastData/CDR/AggregateData/dec_working.npy")
    jan_w = np.load("IvoryCoastData/CDR/AggregateData/jan_working.npy")
    feb_w = np.load("IvoryCoastData/CDR/AggregateData/feb_working.npy")
    mar_w = np.load("IvoryCoastData/CDR/AggregateData/mar_working.npy")
    apr_w = np.load("IvoryCoastData/CDR/AggregateData/apr_working.npy")

    dec_nw = np.load("IvoryCoastData/CDR/AggregateData/dec_nworking.npy")
    jan_nw = np.load("IvoryCoastData/CDR/AggregateData/jan_nworking.npy")
    feb_nw = np.load("IvoryCoastData/CDR/AggregateData/feb_nworking.npy")
    mar_nw = np.load("IvoryCoastData/CDR/AggregateData/mar_nworking.npy")
    apr_nw = np.load("IvoryCoastData/CDR/AggregateData/apr_nworking.npy")

    ''' Mean Outputs '''
    dec_morn_mean = np.mean(dec_morn, axis=2)
    dec_morn_mean = np.where(dec_morn_mean[0][0:1237] > 0, np.log(dec_morn_mean[0][0:1237]), dec_morn_mean[0][0:1237])
    dec_aft_mean = np.mean(dec_aft[0][0:1237])
    dec_eve_mean = np.mean(dec_eve[0][0:1237])
    dec_night_mean = np.mean(dec_night[0][0:1237])
    dec_w_mean = np.mean(dec_w[0][0:1237])
    dec_nw_mean = np.mean(dec_nw[0][0:1237])

    jan_morn_mean = np.mean(jan_morn, axis=2)
    jan_morn_mean = np.where(jan_morn_mean[0][0:1237] > 0, np.log(jan_morn_mean[0][0:1237]), jan_morn_mean[0][0:1237])
    jan_aft_mean = np.mean(jan_aft[0][0:1237])
    jan_eve_mean = np.mean(jan_eve[0][0:1237])
    jan_night_mean = np.mean(jan_night[0][0:1237])
    jan_w_mean = np.mean(jan_w[0][0:1237])
    jan_nw_mean = np.mean(jan_nw[0][0:1237])

    feb_morn_mean = np.mean(feb_morn, axis=2)
    feb_morn_mean = np.where(feb_morn_mean[0][0:1237] > 0, np.log(feb_morn_mean[0][0:1237]), feb_morn_mean[0][0:1237])
    feb_aft_mean = np.mean(feb_aft[0][0:1237])
    feb_eve_mean = np.mean(feb_eve[0][0:1237])
    feb_night_mean = np.mean(feb_night[0][0:1237])
    feb_w_mean = np.mean(feb_w[0][0:1237])
    feb_nw_mean = np.mean(feb_nw[0][0:1237])

    mar_morn_mean = np.mean(mar_morn, axis=2)
    mar_morn_mean = np.where(mar_morn_mean[0][0:1237] > 0, np.log(mar_morn_mean[0][0:1237]), mar_morn_mean[0][0:1237])
    mar_aft_mean = np.mean(mar_aft[0][0:1237])
    mar_eve_mean = np.mean(mar_eve[0][0:1237])
    mar_night_mean = np.mean(mar_night[0][0:1237])
    mar_w_mean = np.mean(mar_w[0][0:1237])
    mar_nw_mean = np.mean(mar_nw[0][0:1237])

    apr_morn_mean = np.mean(apr_morn, axis=2)
    apr_morn_mean = np.where(apr_morn_mean[0][0:1237] > 0, np.log(apr_morn_mean[0][0:1237]), apr_morn_mean[0][0:1237])
    apr_aft_mean = np.mean(apr_aft[0][0:1237])
    apr_eve_mean = np.mean(apr_eve[0][0:1237])
    apr_night_mean = np.mean(apr_night[0][0:1237])
    apr_w_mean = np.mean(apr_w[0][0:1237])
    apr_nw_mean = np.mean(apr_nw[0][0:1237])
    
    ''' Output Volatility '''

    dec_morn_std = np.std(dec_morn_mean)
    dec_morn_mean = np.mean(dec_morn_mean)
    dec_aft_std = np.std(dec_aft[0][0:1237])
    dec_eve_std = np.std(dec_eve[0][0:1237])
    dec_night_std = np.std(dec_night[0][0:1237])
    dec_w_std = np.std(dec_w[0][0:1237])
    dec_nw_std = np.std(dec_nw[0][0:1237])

    jan_morn_std = np.std(jan_morn_mean)
    jan_morn_mean = np.mean(jan_morn_mean)
    jan_aft_std = np.std(jan_aft[0][0:1237])
    jan_eve_std = np.std(jan_eve[0][0:1237])
    jan_night_std = np.std(jan_night[0][0:1237])
    jan_w_std = np.std(jan_w[0][0:1237])
    jan_nw_std = np.std(jan_nw[0][0:1237])

    feb_morn_std = np.std(feb_morn_mean)
    feb_morn_mean = np.mean(feb_morn_mean)
    feb_aft_std = np.std(feb_aft[0][0:1237])
    feb_eve_std = np.std(feb_eve[0][0:1237])
    feb_night_std = np.std(feb_night[0][0:1237])
    feb_w_std = np.std(feb_w[0][0:1237])
    feb_nw_std = np.std(feb_nw[0][0:1237])

    mar_morn_std = np.std(mar_morn_mean)
    mar_morn_mean = np.mean(mar_morn_mean)
    mar_aft_std = np.std(mar_aft[0][0:1237])
    mar_eve_std = np.std(mar_eve[0][0:1237])
    mar_night_std = np.std(mar_night[0][0:1237])
    mar_w_std = np.std(mar_w[0][0:1237])
    mar_nw_std = np.std(mar_nw[0][0:1237])

    apr_morn_std = np.std(apr_morn_mean)
    apr_morn_mean = np.mean(apr_morn_mean)
    apr_aft_std = np.std(apr_aft[0][0:1237])
    apr_eve_std = np.std(apr_eve[0][0:1237])
    apr_night_std = np.std(apr_night[0][0:1237])
    apr_w_std = np.std(apr_w[0][0:1237])
    apr_nw_std = np.std(apr_nw[0][0:1237])

    mean_morn = [dec_morn_mean, jan_morn_mean, feb_morn_mean, mar_morn_mean, apr_morn_mean]
    mean_aft = [dec_aft_mean, jan_aft_mean, feb_aft_mean, mar_aft_mean, apr_aft_mean]
    mean_eve = [dec_eve_mean, jan_eve_mean, feb_eve_mean, mar_eve_mean, apr_eve_mean]
    mean_night = [dec_night_mean, jan_night_mean, feb_night_mean, mar_night_mean, apr_night_mean]
    mean_w = [dec_w_mean, jan_w_mean, feb_w_mean, mar_w_mean, apr_w_mean]
    mean_nw = [dec_nw_mean, jan_nw_mean, feb_nw_mean, mar_nw_mean, apr_nw_mean]

    volatility_morn = [dec_morn_std, jan_morn_std, feb_morn_std, mar_morn_std, apr_morn_std]
    volatility_aft = [dec_aft_std, jan_aft_std, feb_aft_std, mar_aft_std, apr_aft_std]
    volatility_eve = [dec_eve_std, jan_eve_std, feb_eve_std, mar_eve_std, apr_eve_std]
    volatility_night = [dec_night_std, jan_night_std, feb_night_std, mar_night_std, apr_night_std]
    volatility_w = [dec_w_std, jan_w_std, feb_w_std, mar_w_std, apr_w_std]
    volatility_nw = [dec_nw_std, jan_nw_std, feb_nw_std, mar_nw_std, apr_nw_std]

    months = ['D', 'J', 'F', 'M', 'A']
    rainfall = pd.DataFrame(pd.read_csv("IvoryCoastData/Geography/ICRainfall11-12.csv"), index=None)

    # plt.bar(range(5), volatility_morn, label='Morn')
    # plt.plot(range(5), volatility_aft, label='Aft')
    # plt.plot(range(5), volatility_eve, label='Eve')
    # plt.plot(range(5), volatility_night, label='Night')
    # plt.bar(range(5), rainfall['rainfall'])
    # plt.ylim(0, 1500)
    # plt.legend()
    # plt.grid()
    # plt.show()

    plt.errorbar(range(5), mean_morn, yerr=volatility_morn)
    # plt.bar(range(5), rainfall['rainfall'], color='blue')
    plt.legend()
    plt.grid()
    plt.show()




    # plt.subplot(1, 2, 1)
    # mean_working_dec = sorted(np.mean(dec_working, axis=2)[0])
    # fit = stats.norm.pdf(mean_working_dec[0:1237], np.mean(mean_working_dec[0:1237]), np.std(mean_working_dec[0:1237]))
    # # plt.hist(sorted(mean_working_dec[0:1237]), 40, normed=True, color='red')
    # plt.plot(mean_working_dec[0:1237], fit, '-', label='Dec')
    #
    # mean_working_jan = sorted(np.mean(jan_working, axis=2)[0])
    # fit = stats.norm.pdf(mean_working_jan[0:1237], np.mean(mean_working_jan[0:1237]), np.std(mean_working_jan[0:1237]))
    # # plt.hist(sorted(mean_working_jan[0:1237]), 40, normed=True, color='red')
    # plt.plot(mean_working_jan[0:1237], fit, '-', label='Jan')
    #
    # mean_working_feb = sorted(np.mean(feb_working, axis=2)[0])
    # fit = stats.norm.pdf(mean_working_feb[0:1237], np.mean(mean_working_feb[0:1237]), np.std(mean_working_feb[0:1237]))
    # # plt.hist(sorted(mean_working_feb[0:1237]), 40, normed=True, color='red')
    # plt.plot(mean_working_feb[0:1237], fit, '-', label='Feb')
    #
    # mean_working_mar = sorted(np.mean(mar_working, axis=2)[0])
    # fit = stats.norm.pdf(mean_working_mar[0:1237], np.mean(mean_working_mar[0:1237]), np.std(mean_working_mar[0:1237]))
    # # plt.hist(sorted(mean_working_mar[0:1237]), 40, normed=True, color='red')
    # plt.plot(mean_working_mar[0:1237], fit, '-', label='Mar')
    #
    # mean_working_apr = sorted(np.mean(apr_working, axis=2)[0])
    # fit = stats.norm.pdf(mean_working_apr[0:1237], np.mean(mean_working_apr[0:1237]), np.std(mean_working_apr[0:1237]))
    # # plt.hist(sorted(mean_working_apr[0:1237]), 40, normed=True, color='red')
    # plt.plot(mean_working_apr[0:1237], fit, '-', label='Apr')
    # plt.grid()
    # plt.legend()





    # # ''' Plotting whole period working/non working'''
    # # plt.subplot(2,2,1)
    # # mean_working = sorted(np.mean(total_working, axis=2)[0])
    # # fit = stats.halfnorm.pdf(mean_working[0:1237], np.mean(mean_working[0:1237]), np.std(mean_working[0:1237]))
    # # plt.hist(sorted(mean_working[0:1237]), 40, normed=True, color='red')
    # # plt.plot(mean_working[0:1237], fit, '-', label='Working Hours')
    # # plt.legend()
    # #
    # # plt.subplot(2, 2, 2)
    # mean_output_w = sorted(np.mean(total_w, axis=2)[0])
    # volatility_w = sorted(np.std(total_w, axis=2)[0])
    #
    # # fit = stats.norm.pdf(volatility_w[0:1237], np.mean(volatility_w[0:1237]), np.std(volatility_w[0:1237]))
    # # plt.hist(sorted(volatility_w[0:1237]), 40, normed=True, color='red')
    # # plt.plot(volatility_w[0:1237], fit, '-',  label='Working Hours')
    # # plt.legend()
    #
    # mean_output_nw = sorted(np.mean(total_nw, axis=2)[0])
    # volatility_nw = sorted(np.std(total_nw, axis=2)[0])
    #
    # fit = stats.norm.pdf(volatility_nw[0:1237], np.mean(volatility_nw[0:1237]), np.std(volatility_nw[0:1237]))
    # plt.plot(volatility_nw[0:1237], fit, '-', label='Non-working Hours')
    # plt.hist(sorted(volatility_nw[0:1237]), 8, normed=True, color='blue', alpha=0.5)
    # plt.axvline(x=198, ymin=0, ymax=0.0030)
    # plt.legend()
    # plt.show()


    
    
    
    
    
    
    
    
    
    #
    # plt.subplot(2, 2, 4)
    # volatility_nw = sorted(np.std(total_nw, axis=2)[0])
    # mean_output_nw = sorted(np.mean(total_nw, axis=2)[0])
    # fit = stats.norm.pdf(std_nworking[0:1237], np.mean(std_nworking[0:1237]), np.std(std_nworking[0:1237]))
    # plt.plot(std_nworking[0:1237], fit, '-', label='Non-working Hours')
    # plt.hist(sorted(std_nworking[0:1237]), 8, normed=True, color='blue', alpha=0.5)
    # plt.legend()
    # plt.show()

    




    # plt.scatter(range(1237), volatility_w[0:1237], label='working', facecolor='red')
    # plt.scatter(range(1237), volatility_nw[0:1237], label='non-working', facecolor='blue')
    # plt.legend()
    # plt.grid()
    # plt.show()
    
    
    
    # 
    # ''' Splitting into individual months '''
    # dec_working = np.load("IvoryCoastData/CDR/AggregateData/dec_working.npy")
    # jan_working = np.load("IvoryCoastData/CDR/AggregateData/jan_working.npy")
    # feb_working = np.load("IvoryCoastData/CDR/AggregateData/feb_working.npy")
    # mar_working = np.load("IvoryCoastData/CDR/AggregateData/mar_working.npy")
    # apr_working = np.load("IvoryCoastData/CDR/AggregateData/apr_working.npy")
    # 
    # rainfall = pd.DataFrame(pd.read_csv("IvoryCoastData/Geography/ICRainfall11-12.csv"), index=None)
    # 
    # plt.subplot(1,2,1)
    # mean_working_dec = sorted(np.mean(dec_working, axis=2)[0])
    # fit = stats.norm.pdf(mean_working_dec[0:1237], np.mean(mean_working_dec[0:1237]), np.std(mean_working_dec[0:1237]))
    # # plt.hist(sorted(mean_working_dec[0:1237]), 40, normed=True, color='red')
    # plt.plot(mean_working_dec[0:1237], fit, '-', label='Dec')
    # 
    # mean_working_jan = sorted(np.mean(jan_working, axis=2)[0])
    # fit = stats.norm.pdf(mean_working_jan[0:1237], np.mean(mean_working_jan[0:1237]), np.std(mean_working_jan[0:1237]))
    # # plt.hist(sorted(mean_working_jan[0:1237]), 40, normed=True, color='red')
    # plt.plot(mean_working_jan[0:1237], fit, '-', label='Jan')
    # 
    # mean_working_feb = sorted(np.mean(feb_working, axis=2)[0])
    # fit = stats.norm.pdf(mean_working_feb[0:1237], np.mean(mean_working_feb[0:1237]), np.std(mean_working_feb[0:1237]))
    # # plt.hist(sorted(mean_working_feb[0:1237]), 40, normed=True, color='red')
    # plt.plot(mean_working_feb[0:1237], fit, '-', label='Feb')
    # 
    # mean_working_mar = sorted(np.mean(mar_working, axis=2)[0])
    # fit = stats.norm.pdf(mean_working_mar[0:1237], np.mean(mean_working_mar[0:1237]), np.std(mean_working_mar[0:1237]))
    # # plt.hist(sorted(mean_working_mar[0:1237]), 40, normed=True, color='red')
    # plt.plot(mean_working_mar[0:1237], fit, '-', label='Mar')
    # 
    # mean_working_apr = sorted(np.mean(apr_working, axis=2)[0])
    # fit = stats.norm.pdf(mean_working_apr[0:1237], np.mean(mean_working_apr[0:1237]), np.std(mean_working_apr[0:1237]))
    # # plt.hist(sorted(mean_working_apr[0:1237]), 40, normed=True, color='red')
    # plt.plot(mean_working_apr[0:1237], fit, '-', label='Apr')
    # plt.grid()
    # plt.legend()
    # 
    # plt.subplot(1, 2, 2)
    # rain = rainfall['rainfall']
    # plt.bar(range(5), rain, align='center')
    # labels = ['dec', 'jan', 'feb', 'mar', 'apr']
    # plt.xticks(range(5), labels)
    # 
    # plt.grid()
    # plt.legend()
    # plt.show()
    # 
    # 



















    # # Working hours model
    # mean_working = np.mean(total_working, axis=2)[0]
    # for i in range(1238):
    #     if mean_working[i] > 0:
    #         mean_working[i] = np.log(mean_working[i])
    # data_working = sorted(mean_working[0:1237])
    # fit_working = stats.norm.pdf(data_working, np.mean(data_working), np.std(data_working))
    # plt.plot(data_working, fit_working, '-o', label='Working Hours')
    #
    # # # Non-working hours
    # # mean_nworking = np.mean(total_nworking, axis=2)
    # # # z_nworking = (mean_nworking[0][0:1237] - np.mean(mean_nworking[0][0:1237])) / np.std(mean_nworking[0][0:1237])
    # # data_nworking = sorted(mean_nworking[0][0:1237])
    # # fit_nworking = stats.norm.pdf(data_nworking, np.mean(data_nworking), np.std(data_nworking))
    # # plt.plot(data_nworking, fit_nworking, '-o', label='Non-working Hours')
    #
    # plt.grid()
    # plt.legend()
    # plt.show()

    # for month in MONTHS:
    #     # Working hours
    #     month_working = np.load("IvoryCoastData/CDR/AggregateData/%s_working.npy" % month)
    #     mean_month_working = np.mean(month_working, axis=2)
    #     for i in range(1238):
    #         if mean_month_working[0][i] != 0:
    #             mean_month_working[0][i] = np.log(mean_month_working[0][i])
    #     data_month_working = sorted(mean_month_working[0][0:1237])
    #     fit_month_working = stats.norm.pdf(data_month_working, np.mean(data_month_working), np.std(data_month_working))
    #     plt.plot(data_month_working, fit_month_working, '-', label='Working Hours: %s' % month)
    #
    #     # Non-working hours
    #     month_nworking = np.load("IvoryCoastData/CDR/AggregateData/%s_nworking.npy" % month)
    #     mean_month_nworking = np.mean(month_nworking, axis=2)
    #     for i in range(1238):
    #         if mean_month_nworking[0][i] != 0:
    #             mean_month_nworking[0][i] = np.log(mean_month_nworking[0][i])
    #     data_month_nworking = sorted(mean_month_nworking[0][0:1237])
    #     fit_month_nworking = stats.norm.pdf(data_month_nworking, np.mean(data_month_nworking), np.std(data_month_nworking))
    #     plt.plot(data_month_nworking, fit_month_nworking, '-', label='Non-working Hours: %s' % month)
    # #
    # plt.grid()
    # plt.legend()
    # plt.show()
