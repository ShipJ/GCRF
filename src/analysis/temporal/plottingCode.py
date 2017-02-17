import pandas as pd
import numpy as np
import arrow as ar
import time
import sys
import math
import matplotlib.pyplot as plt

# Hours of the day within each phase
MORN = range(5, 12)
AFT = range(12, 17)
EVE = range(17, 21)
NIGHT = [int(math.fmod(i, 24)) for i in range(21, 30)]
PHASES = [MORN, AFT, EVE, NIGHT]

# Start timer
t0 = time.time()

if __name__ == "__main__":


    total_morn = np.load("IvoryCoastData/CDR/AggregateData/total_morn.npy")

    monthly_avg_morn = np.mean(total_morn, axis=2)
    monthly_std_morn = np.std(total_morn, axis=2)
    index1 = np.argsort(monthly_avg_morn[0])
    index2 = np.argsort(monthly_avg_morn[1])
    index3 = np.argsort(monthly_avg_morn[2])
    index4 = np.argsort(monthly_avg_morn[3])
    monthly_avg_morn[0] = monthly_avg_morn[0][index1]
    monthly_std_morn[0] = monthly_std_morn[0][index1]
    monthly_avg_morn[1] = monthly_avg_morn[1][index1]
    monthly_std_morn[1] = monthly_std_morn[1][index1]
    monthly_avg_morn[2] = monthly_avg_morn[2][index3]
    monthly_std_morn[2] = monthly_std_morn[2][index3]
    monthly_avg_morn[3] = monthly_avg_morn[3][index4]
    monthly_std_morn[3] = monthly_std_morn[3][index4]

    total_aft = np.load("IvoryCoastData/CDR/AggregateData/total_aft.npy")
    monthly_avg_aft = np.mean(total_aft, axis=2)
    monthly_std_aft = np.std(total_aft, axis=2)
    index1 = np.argsort(monthly_avg_aft[0])
    index2 = np.argsort(monthly_avg_aft[1])
    index3 = np.argsort(monthly_avg_aft[2])
    index4 = np.argsort(monthly_avg_aft[3])
    monthly_avg_aft[0] = monthly_avg_aft[0][index1]
    monthly_std_aft[0] = monthly_std_aft[0][index1]
    monthly_avg_aft[1] = monthly_avg_aft[1][index1]
    monthly_std_aft[1] = monthly_std_aft[1][index1]
    monthly_avg_aft[2] = monthly_avg_aft[2][index3]
    monthly_std_aft[2] = monthly_std_aft[2][index3]
    monthly_avg_aft[3] = monthly_avg_aft[3][index4]
    monthly_std_aft[3] = monthly_std_aft[3][index4]


    total_eve = np.load("IvoryCoastData/CDR/AggregateData/total_eve.npy")
    monthly_avg_eve = np.mean(total_eve, axis=2)
    monthly_std_eve = np.std(total_eve, axis=2)
    index1 = np.argsort(monthly_avg_eve[0])
    index2 = np.argsort(monthly_avg_eve[1])
    index3 = np.argsort(monthly_avg_eve[2])
    index4 = np.argsort(monthly_avg_eve[3])
    monthly_avg_eve[0] = monthly_avg_eve[0][index1]
    monthly_std_eve[0] = monthly_std_eve[0][index1]
    monthly_avg_eve[1] = monthly_avg_eve[1][index1]
    monthly_std_eve[1] = monthly_std_eve[1][index1]
    monthly_avg_eve[2] = monthly_avg_eve[2][index3]
    monthly_std_eve[2] = monthly_std_eve[2][index3]
    monthly_avg_eve[3] = monthly_avg_eve[3][index4]
    monthly_std_eve[3] = monthly_std_eve[3][index4]

    total_night = np.load("IvoryCoastData/CDR/AggregateData/total_night.npy")
    monthly_avg_night = np.mean(total_night, axis=2)
    monthly_std_night = np.std(total_night, axis=2)
    index1 = np.argsort(monthly_avg_night[0])
    index2 = np.argsort(monthly_avg_night[1])
    index3 = np.argsort(monthly_avg_night[2])
    index4 = np.argsort(monthly_avg_night[3])
    monthly_avg_night[0] = monthly_avg_night[0][index1]
    monthly_std_night[0] = monthly_std_night[0][index1]
    monthly_avg_night[1] = monthly_avg_night[1][index1]
    monthly_std_night[1] = monthly_std_night[1][index1]
    monthly_avg_night[2] = monthly_avg_night[2][index3]
    monthly_std_night[2] = monthly_std_night[2][index3]
    monthly_avg_night[3] = monthly_avg_night[3][index4]
    monthly_std_night[3] = monthly_std_night[3][index4]


    plt.figure(1, facecolor='white')
    plt.subplot(2,1,1)
    plt.plot(range(1237), monthly_avg_morn[0][0:1237], '-', label='morn', linewidth=1)
    plt.plot(range(1237), monthly_avg_aft[0][0:1237], '-', label='aft', linewidth=1)
    plt.plot(range(1237), monthly_avg_eve[0][0:1237], '-', label='eve', linewidth=1)
    plt.plot(range(1237), monthly_avg_night[0][0:1237], '-', label='night', linewidth=1)
    plt.xlim(0, 1237)
    plt.ylim(0, 12000)
    plt.xlabel("Node ID", fontsize=16)
    plt.ylabel("Outgoing Call Volume", fontsize=16)
    plt.title("Outgoing call volume for each node - 5 months", fontsize=16)
    plt.legend()
    plt.grid()

    plt.subplot(2,1,2)
    plt.plot(range(1237), monthly_avg_morn[1][0:1237], '-', label='morn', linewidth=1)
    plt.plot(range(1237), monthly_avg_aft[1][0:1237], '-', label='aft', linewidth=1)
    plt.plot(range(1237), monthly_avg_eve[1][0:1237], '-', label='eve', linewidth=1)
    plt.plot(range(1237), monthly_avg_night[1][0:1237], '-', label='night', linewidth=1)
    plt.xlim(0, 1237)
    plt.ylim(0, 12000)
    plt.xlabel("Node ID", fontsize=16)
    plt.ylabel("Incoming Call Volume", fontsize=16)
    plt.title("Incoming call volume for each node - 5 months", fontsize=16)
    plt.legend()
    plt.grid()

    plt.show()






    # plt.subplot(2,2,1)
    # plt.plot(range(1237), monthly_avg_morn[0][0:1237], '-', label='morn', linewidth=1)
    # plt.plot(range(1237), monthly_avg_aft[0][0:1237], '-', label='aft', linewidth=1)
    # plt.plot(range(1237), monthly_avg_eve[0][0:1237], '-', label='eve', linewidth=1)
    # plt.plot(range(1237), monthly_avg_night[0][0:1237], '-', label='night', linewidth=1)
    # plt.xlim(0, 1237)
    # plt.ylim(0, 12000)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Outgoing Call Volume", fontsize=16)
    # plt.title("Outgoing call volume for each node - 5 months", fontsize=16)
    # plt.legend()
    # plt.grid()
    #
    #
    #
    # prophet_bday_morn = total_morn[:, :, 62]
    # index1 = np.argsort(prophet_bday_morn[0])
    # prophet_bday_aft = total_aft[:, :, 62]
    # index2 = np.argsort(prophet_bday_aft[0])
    # prophet_bday_eve = total_eve[:, :, 62]
    # index3 = np.argsort(prophet_bday_eve[0])
    # prophet_bday_night = total_night[:, :, 62]
    # index4 = np.argsort(prophet_bday_night[0])
    # prophet_bday_morn[0] = prophet_bday_morn[0][index1]
    # prophet_bday_aft[0] = prophet_bday_aft[0][index2]
    # prophet_bday_eve[0] = prophet_bday_eve[0][index3]
    # prophet_bday_night[0] = prophet_bday_night[0][index4]
    #
    #
    # plt.subplot(2,2,2)
    # plt.plot(range(1237), prophet_bday_morn[0][0:1237], '-', label='morn', linewidth=1)
    # plt.plot(range(1237), prophet_bday_aft[0][0:1237], '-', label='aft', linewidth=1)
    # plt.plot(range(1237), prophet_bday_eve[0][0:1237], '-', label='eve', linewidth=1)
    # plt.plot(range(1237), prophet_bday_night[0][0:1237], '-', label='night', linewidth=1)
    # plt.xlim(0, 1237)
    # plt.ylim(0, 12000)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Outgoing Call Volume", fontsize=16)
    # plt.title("Outgoing call volume for each node - Prophet's Birthday", fontsize=16)
    # plt.legend()
    # plt.grid()
    #
    # new_year_morn = total_morn[:, :, 27]
    # index1 = np.argsort(new_year_morn[0])
    # new_year_aft = total_aft[:, :, 27]
    # index2 = np.argsort(new_year_aft[0])
    # new_year_eve = total_eve[:, :, 27]
    # index3 = np.argsort(new_year_eve[0])
    # new_year_night = total_night[:, :, 27]
    # index4 = np.argsort(new_year_night[0])
    # new_year_morn[0] = new_year_morn[0][index1]
    # new_year_aft[0] = new_year_aft[0][index2]
    # new_year_eve[0] = new_year_eve[0][index3]
    # new_year_night[0] = new_year_night[0][index4]
    #
    # plt.subplot(2,2,3)
    # plt.plot(range(1237), new_year_morn[0][0:1237], '-', label='morn', linewidth=1)
    # plt.plot(range(1237), new_year_aft[0][0:1237], '-', label='aft', linewidth=1)
    # plt.plot(range(1237), new_year_eve[0][0:1237], '-', label='eve', linewidth=1)
    # plt.plot(range(1237), new_year_night[0][0:1237], '-', label='night', linewidth=1)
    # plt.xlim(0, 1237)
    # plt.ylim(0, 12000)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Outgoing Call Volume", fontsize=16)
    # plt.title("Outgoing call volume for each node - New Year's Day", fontsize=16)
    # plt.legend()
    # plt.grid()
    #
    #
    #
    # easter_morn = total_morn[:, :, 126]
    # index1 = np.argsort(easter_morn[0])
    # easter_aft = total_aft[:, :, 126]
    # index2 = np.argsort(easter_aft[0])
    # easter_eve = total_eve[:, :, 126]
    # index3 = np.argsort(easter_eve[0])
    # easter_night = total_night[:, :, 126]
    # index4 = np.argsort(easter_night[0])
    # easter_morn[0] = easter_morn[0][index1]
    # easter_aft[0] = easter_aft[0][index2]
    # easter_eve[0] = easter_eve[0][index3]
    # easter_night[0] = easter_night[0][index4]
    #
    # plt.subplot(2,2,4)
    # plt.plot(range(1237), easter_morn[0][0:1237], '-', label='morn', linewidth=1)
    # plt.plot(range(1237), easter_aft[0][0:1237], '-', label='aft', linewidth=1)
    # plt.plot(range(1237), easter_eve[0][0:1237], '-', label='eve', linewidth=1)
    # plt.plot(range(1237), easter_night[0][0:1237], '-', label='night', linewidth=1)
    # plt.xlim(0, 1237)
    # plt.ylim(0, 12000)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Outgoing Call Volume", fontsize=16)
    # plt.title("Outgoing call volume for each node - Easter Day", fontsize=16)
    # plt.legend()
    # plt.grid()






    plt.show()





























    # ''' MORNING '''
    # mon_morn = np.load("IvoryCoastData/CDR/AggregateData/mon_morn.npy")
    # mon_morn_avg = np.mean(mon_morn, axis=2)
    # mon_morn_std = np.std(mon_morn, axis=2)
    # index1 = np.argsort(mon_morn_avg[0])
    # # index2 = np.argsort(mon_morn_avg[1])
    # # index3 = np.argsort(mon_morn_avg[2])
    # # index4 = np.argsort(mon_morn_avg[3])
    # mon_morn_avg[0] = mon_morn_avg[0][index1]
    # mon_morn_std[0] = mon_morn_std[0][index1]
    # # mon_morn_std[1] = mon_morn_std[1][index2]
    # # mon_morn_std[1] = mon_morn_std[1][index2]
    # # mon_morn_std[2] = mon_morn_std[2][index3]
    # # mon_morn_std[2] = mon_morn_std[2][index3]
    # # mon_morn_std[3] = mon_morn_std[3][index4]
    # # mon_morn_std[3] = mon_morn_std[3][index4]
    #
    # tue_morn = np.load("IvoryCoastData/CDR/AggregateData/tue_morn.npy")
    # tue_morn_avg = np.mean(tue_morn, axis=2)
    # tue_morn_std = np.std(tue_morn, axis=2)
    # index1 = np.argsort(tue_morn_avg[0])
    # tue_morn_avg[0] = tue_morn_avg[0][index1]
    # tue_morn_std[0] = tue_morn_std[0][index1]
    #
    # wed_morn = np.load("IvoryCoastData/CDR/AggregateData/wed_morn.npy")
    # wed_morn_avg = np.mean(wed_morn, axis=2)
    # wed_morn_std = np.std(wed_morn, axis=2)
    # index1 = np.argsort(wed_morn_avg[0])
    # wed_morn_avg[0] = wed_morn_avg[0][index1]
    # wed_morn_std[0] = wed_morn_std[0][index1]
    #
    # thu_morn = np.load("IvoryCoastData/CDR/AggregateData/thu_morn.npy")
    # thu_morn_avg = np.mean(thu_morn, axis=2)
    # thu_morn_std = np.std(thu_morn, axis=2)
    # index1 = np.argsort(thu_morn_avg[0])
    # thu_morn_avg[0] = thu_morn_avg[0][index1]
    # thu_morn_std[0] = thu_morn_std[0][index1]
    #
    # fri_morn = np.load("IvoryCoastData/CDR/AggregateData/fri_morn.npy")
    # fri_morn_avg = np.mean(fri_morn, axis=2)
    # fri_morn_std = np.std(fri_morn, axis=2)
    # index1 = np.argsort(fri_morn_avg[0])
    # fri_morn_avg[0] = fri_morn_avg[0][index1]
    # fri_morn_std[0] = fri_morn_std[0][index1]
    #
    # sat_morn = np.load("IvoryCoastData/CDR/AggregateData/sat_morn.npy")
    # sat_morn_avg = np.mean(sat_morn, axis=2)
    # sat_morn_std = np.std(sat_morn, axis=2)
    # index1 = np.argsort(sat_morn_avg[0])
    # sat_morn_avg[0] = sat_morn_avg[0][index1]
    # sat_morn_std[0] = sat_morn_std[0][index1]
    #
    # sun_morn = np.load("IvoryCoastData/CDR/AggregateData/sun_morn.npy")
    # sun_morn_avg = np.mean(sun_morn, axis=2)
    # sun_morn_std = np.std(sun_morn, axis=2)
    # index1 = np.argsort(sun_morn_avg[0])
    # sun_morn_avg[0] = sun_morn_avg[0][index1]
    # sun_morn_std[0] = sun_morn_std[0][index1]
    #
    #
    #
    # plt.figure(1, facecolor='white')
    # plt.subplot(2,2,1)
    # plt.plot(range(1237), mon_morn_avg[0][0:1237], '-', label='Monday', linewidth=0.8)
    # plt.plot(range(1237), tue_morn_avg[0][0:1237], '-', label='Tuesday', linewidth=0.8)
    # plt.plot(range(1237), wed_morn_avg[0][0:1237], '-', label='Wednesday', linewidth=0.8)
    # plt.plot(range(1237), thu_morn_avg[0][0:1237], '-', label='Thursday', linewidth=0.8)
    # plt.plot(range(1237), fri_morn_avg[0][0:1237], '-', label='Friday', linewidth=0.8)
    # plt.plot(range(1237), sat_morn_avg[0][0:1237], '-', label='Saturday', linewidth=0.8)
    # plt.plot(range(1237), sun_morn_avg[0][0:1237], '-', label='Sunday', linewidth=0.8)
    # plt.xlim(0, 1237)
    # plt.ylim(0, 10000)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Outgoing Call Volume (Morning)", fontsize=16)
    # plt.title("Outgoing call volume: Days of the Week - Morning", fontsize=16)
    # plt.legend(loc='upper left')
    # plt.grid()
    #
    #
    # ''' NIGHT '''
    # mon_night = np.load("IvoryCoastData/CDR/AggregateData/mon_night.npy")
    # mon_night_avg = np.mean(mon_night, axis=2)
    # mon_night_std = np.std(mon_night, axis=2)
    # index1 = np.argsort(mon_night_avg[0])
    # # index2 = np.argsort(mon_morn_avg[1])
    # # index3 = np.argsort(mon_morn_avg[2])
    # # index4 = np.argsort(mon_morn_avg[3])
    # mon_night_avg[0] = mon_night_avg[0][index1]
    # mon_night_std[0] = mon_night_std[0][index1]
    # # mon_morn_std[1] = mon_morn_std[1][index2]
    # # mon_morn_std[1] = mon_morn_std[1][index2]
    # # mon_morn_std[2] = mon_morn_std[2][index3]
    # # mon_morn_std[2] = mon_morn_std[2][index3]
    # # mon_morn_std[3] = mon_morn_std[3][index4]
    # # mon_morn_std[3] = mon_morn_std[3][index4]
    #
    # tue_night = np.load("IvoryCoastData/CDR/AggregateData/tue_night.npy")
    # tue_night_avg = np.mean(tue_night, axis=2)
    # tue_night_std = np.std(tue_night, axis=2)
    # index1 = np.argsort(tue_night_avg[0])
    # tue_night_avg[0] = tue_night_avg[0][index1]
    # tue_night_std[0] = tue_night_std[0][index1]
    #
    # wed_night = np.load("IvoryCoastData/CDR/AggregateData/wed_night.npy")
    # wed_night_avg = np.mean(wed_night, axis=2)
    # wed_night_std = np.std(wed_night, axis=2)
    # index1 = np.argsort(wed_night_avg[0])
    # wed_night_avg[0] = wed_night_avg[0][index1]
    # wed_night_std[0] = wed_night_std[0][index1]
    #
    # thu_night = np.load("IvoryCoastData/CDR/AggregateData/thu_night.npy")
    # thu_night_avg = np.mean(thu_night, axis=2)
    # thu_night_std = np.std(thu_night, axis=2)
    # index1 = np.argsort(thu_night_avg[0])
    # thu_night_avg[0] = thu_night_avg[0][index1]
    # thu_night_std[0] = thu_night_std[0][index1]
    #
    # fri_night = np.load("IvoryCoastData/CDR/AggregateData/fri_night.npy")
    # fri_night_avg = np.mean(fri_night, axis=2)
    # fri_night_std = np.std(fri_night, axis=2)
    # index1 = np.argsort(fri_night_avg[0])
    # fri_night_avg[0] = fri_night_avg[0][index1]
    # fri_night_std[0] = fri_night_std[0][index1]
    #
    # sat_night = np.load("IvoryCoastData/CDR/AggregateData/sat_night.npy")
    # sat_night_avg = np.mean(sat_night, axis=2)
    # sat_night_std = np.std(sat_night, axis=2)
    # index1 = np.argsort(sat_night_avg[0])
    # sat_night_avg[0] = sat_night_avg[0][index1]
    # sat_night_std[0] = sat_night_std[0][index1]
    #
    # sun_night = np.load("IvoryCoastData/CDR/AggregateData/sun_night.npy")
    # sun_night_avg = np.mean(sun_night, axis=2)
    # sun_night_std = np.std(sun_night, axis=2)
    # index1 = np.argsort(sun_night_avg[0])
    # sun_night_avg[0] = sun_night_avg[0][index1]
    # sun_night_std[0] = sun_night_std[0][index1]
    #
    #
    # plt.subplot(2,2,4)
    # plt.plot(range(1237), mon_night_avg[0][0:1237], '-', label='Monday', linewidth=0.8)
    # plt.plot(range(1237), tue_night_avg[0][0:1237], '-', label='Tuesday', linewidth=0.8)
    # plt.plot(range(1237), wed_night_avg[0][0:1237], '-', label='Wednesday', linewidth=0.8)
    # plt.plot(range(1237), thu_night_avg[0][0:1237], '-', label='Thursday', linewidth=0.8)
    # plt.plot(range(1237), fri_night_avg[0][0:1237], '-', label='Friday', linewidth=0.8)
    # plt.plot(range(1237), sat_night_avg[0][0:1237], '-', label='Saturday', linewidth=0.8)
    # plt.plot(range(1237), sun_night_avg[0][0:1237], '-', label='Sunday', linewidth=0.8)
    # plt.xlim(0, 1237)
    # plt.ylim(0, 10000)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Outgoing Call Volume (Night)", fontsize=16)
    # plt.title("Outgoing call volume: Days of the Week - Night", fontsize=16)
    # plt.legend(loc='upper left')
    # plt.grid()
    #
    #
    # ''' AFTERNOON '''
    # mon_aft = np.load("IvoryCoastData/CDR/AggregateData/mon_aft.npy")
    # mon_aft_avg = np.mean(mon_aft, axis=2)
    # mon_aft_std = np.std(mon_aft, axis=2)
    # index1 = np.argsort(mon_aft_avg[0])
    # # index2 = np.argsort(mon_morn_avg[1])
    # # index3 = np.argsort(mon_morn_avg[2])
    # # index4 = np.argsort(mon_morn_avg[3])
    # mon_aft_avg[0] = mon_aft_avg[0][index1]
    # mon_aft_std[0] = mon_aft_std[0][index1]
    # # mon_morn_std[1] = mon_morn_std[1][index2]
    # # mon_morn_std[1] = mon_morn_std[1][index2]
    # # mon_morn_std[2] = mon_morn_std[2][index3]
    # # mon_morn_std[2] = mon_morn_std[2][index3]
    # # mon_morn_std[3] = mon_morn_std[3][index4]
    # # mon_morn_std[3] = mon_morn_std[3][index4]
    #
    # tue_aft = np.load("IvoryCoastData/CDR/AggregateData/tue_aft.npy")
    # tue_aft_avg = np.mean(tue_aft, axis=2)
    # tue_aft_std = np.std(tue_aft, axis=2)
    # index1 = np.argsort(tue_aft_avg[0])
    # tue_aft_avg[0] = tue_aft_avg[0][index1]
    # tue_aft_std[0] = tue_aft_std[0][index1]
    #
    # wed_aft = np.load("IvoryCoastData/CDR/AggregateData/wed_aft.npy")
    # wed_aft_avg = np.mean(wed_aft, axis=2)
    # wed_aft_std = np.std(wed_aft, axis=2)
    # index1 = np.argsort(wed_aft_avg[0])
    # wed_aft_avg[0] = wed_aft_avg[0][index1]
    # wed_aft_std[0] = wed_aft_std[0][index1]
    #
    # thu_aft = np.load("IvoryCoastData/CDR/AggregateData/thu_aft.npy")
    # thu_aft_avg = np.mean(thu_aft, axis=2)
    # thu_aft_std = np.std(thu_aft, axis=2)
    # index1 = np.argsort(thu_aft_avg[0])
    # thu_aft_avg[0] = thu_aft_avg[0][index1]
    # thu_aft_std[0] = thu_aft_std[0][index1]
    #
    # fri_aft = np.load("IvoryCoastData/CDR/AggregateData/fri_aft.npy")
    # fri_aft_avg = np.mean(fri_aft, axis=2)
    # fri_aft_std = np.std(fri_aft, axis=2)
    # index1 = np.argsort(fri_aft_avg[0])
    # fri_aft_avg[0] = fri_aft_avg[0][index1]
    # fri_aft_std[0] = fri_aft_std[0][index1]
    #
    # sat_aft = np.load("IvoryCoastData/CDR/AggregateData/sat_aft.npy")
    # sat_aft_avg = np.mean(sat_aft, axis=2)
    # sat_aft_std = np.std(sat_aft, axis=2)
    # index1 = np.argsort(sat_aft_avg[0])
    # sat_aft_avg[0] = sat_aft_avg[0][index1]
    # sat_aft_std[0] = sat_aft_std[0][index1]
    #
    # sun_aft = np.load("IvoryCoastData/CDR/AggregateData/sun_aft.npy")
    # sun_aft_avg = np.mean(sun_aft, axis=2)
    # sun_aft_std = np.std(sun_aft, axis=2)
    # index1 = np.argsort(sun_aft_avg[0])
    # sun_aft_avg[0] = sun_aft_avg[0][index1]
    # sun_aft_std[0] = sun_aft_std[0][index1]
    #
    #
    # plt.subplot(2,2,2)
    # plt.plot(range(1237), mon_aft_avg[0][0:1237], '-', label='Monday', linewidth=0.8)
    # plt.plot(range(1237), tue_aft_avg[0][0:1237], '-', label='Tuesday', linewidth=0.8)
    # plt.plot(range(1237), wed_aft_avg[0][0:1237], '-', label='Wednesday', linewidth=0.8)
    # plt.plot(range(1237), thu_aft_avg[0][0:1237], '-', label='Thursday', linewidth=0.8)
    # plt.plot(range(1237), fri_aft_avg[0][0:1237], '-', label='Friday', linewidth=0.8)
    # plt.plot(range(1237), sat_aft_avg[0][0:1237], '-', label='Saturday', linewidth=0.8)
    # plt.plot(range(1237), sun_aft_avg[0][0:1237], '-', label='Sunday', linewidth=0.8)
    # plt.xlim(0, 1237)
    # plt.ylim(0, 10000)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Outgoing Call Volume (Afternoon)", fontsize=16)
    # plt.title("Outgoing call volume: Days of the Week - Afternoon", fontsize=16)
    # plt.legend(loc='upper left')
    # plt.grid()
    #
    # ''' EVENING '''
    # mon_eve = np.load("IvoryCoastData/CDR/AggregateData/mon_eve.npy")
    # mon_eve_avg = np.mean(mon_eve, axis=2)
    # mon_eve_std = np.std(mon_eve, axis=2)
    # index1 = np.argsort(mon_eve_avg[0])
    # # index2 = np.argsort(mon_morn_avg[1])
    # # index3 = np.argsort(mon_morn_avg[2])
    # # index4 = np.argsort(mon_morn_avg[3])
    # mon_eve_avg[0] = mon_eve_avg[0][index1]
    # mon_eve_std[0] = mon_eve_std[0][index1]
    # # mon_morn_std[1] = mon_morn_std[1][index2]
    # # mon_morn_std[1] = mon_morn_std[1][index2]
    # # mon_morn_std[2] = mon_morn_std[2][index3]
    # # mon_morn_std[2] = mon_morn_std[2][index3]
    # # mon_morn_std[3] = mon_morn_std[3][index4]
    # # mon_morn_std[3] = mon_morn_std[3][index4]
    #
    # tue_eve = np.load("IvoryCoastData/CDR/AggregateData/tue_eve.npy")
    # tue_eve_avg = np.mean(tue_eve, axis=2)
    # tue_eve_std = np.std(tue_eve, axis=2)
    # index1 = np.argsort(tue_eve_avg[0])
    # tue_eve_avg[0] = tue_eve_avg[0][index1]
    # tue_eve_std[0] = tue_eve_std[0][index1]
    #
    # wed_eve = np.load("IvoryCoastData/CDR/AggregateData/wed_eve.npy")
    # wed_eve_avg = np.mean(wed_eve, axis=2)
    # wed_eve_std = np.std(wed_eve, axis=2)
    # index1 = np.argsort(wed_eve_avg[0])
    # wed_eve_avg[0] = wed_eve_avg[0][index1]
    # wed_eve_std[0] = wed_eve_std[0][index1]
    #
    # thu_eve = np.load("IvoryCoastData/CDR/AggregateData/thu_eve.npy")
    # thu_eve_avg = np.mean(thu_eve, axis=2)
    # thu_eve_std = np.std(thu_eve, axis=2)
    # index1 = np.argsort(thu_eve_avg[0])
    # thu_eve_avg[0] = thu_eve_avg[0][index1]
    # thu_eve_std[0] = thu_eve_std[0][index1]
    #
    # fri_eve = np.load("IvoryCoastData/CDR/AggregateData/fri_eve.npy")
    # fri_eve_avg = np.mean(fri_eve, axis=2)
    # fri_eve_std = np.std(fri_eve, axis=2)
    # index1 = np.argsort(fri_eve_avg[0])
    # fri_eve_avg[0] = fri_eve_avg[0][index1]
    # fri_eve_std[0] = fri_eve_std[0][index1]
    #
    # sat_eve = np.load("IvoryCoastData/CDR/AggregateData/sat_eve.npy")
    # sat_eve_avg = np.mean(sat_eve, axis=2)
    # sat_eve_std = np.std(sat_eve, axis=2)
    # index1 = np.argsort(sat_eve_avg[0])
    # sat_eve_avg[0] = sat_eve_avg[0][index1]
    # sat_eve_std[0] = sat_eve_std[0][index1]
    #
    # sun_eve = np.load("IvoryCoastData/CDR/AggregateData/sun_eve.npy")
    # sun_eve_avg = np.mean(sun_eve, axis=2)
    # sun_eve_std = np.std(sun_eve, axis=2)
    # index1 = np.argsort(sun_eve_avg[0])
    # sun_eve_avg[0] = sun_eve_avg[0][index1]
    # sun_eve_std[0] = sun_eve_std[0][index1]
    #
    #
    # plt.subplot(2,2,3)
    # plt.plot(range(1237), mon_eve_avg[0][0:1237], '-', label='Monday', linewidth=0.8)
    # plt.plot(range(1237), tue_eve_avg[0][0:1237], '-', label='Tuesday', linewidth=0.8)
    # plt.plot(range(1237), wed_eve_avg[0][0:1237], '-', label='Wednesday', linewidth=0.8)
    # plt.plot(range(1237), thu_eve_avg[0][0:1237], '-', label='Thursday', linewidth=0.8)
    # plt.plot(range(1237), fri_eve_avg[0][0:1237], '-', label='Friday', linewidth=0.8)
    # plt.plot(range(1237), sat_eve_avg[0][0:1237], '-', label='Saturday', linewidth=0.8)
    # plt.plot(range(1237), sun_eve_avg[0][0:1237], '-', label='Sunday', linewidth=0.8)
    # plt.xlim(0, 1237)
    # plt.ylim(0, 10000)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Outgoing Call Volume (Evening)", fontsize=16)
    # plt.title("Outgoing call volume: Days of the Week - Evening", fontsize=16)
    # plt.legend(loc='upper left')
    # plt.grid()
    #
    # plt.show()
    #
    #

    #
    # ''' Plotting of Results '''
    #
    # # take average over time axis, want a 4 x 1238 array for each
    # monthly_avg_morn = np.multiply(np.mean(total_morn, axis=2), (8/7.0))
    # monthly_std_morn = np.multiply(np.std(total_morn, axis=2), (8/7.0))
    # # monthly_avg_morn[0] /= float(np.max(monthly_avg_morn[0]))
    # # monthly_std_morn[0] /= float(np.max(monthly_std_morn[0]))
    # index1 = np.argsort(monthly_avg_morn[0])
    # index2 = np.argsort(monthly_avg_morn[1])
    # index3 = np.argsort(monthly_avg_morn[2])
    # index4 = np.argsort(monthly_avg_morn[3])
    # monthly_avg_morn[0] = monthly_avg_morn[0][index1]
    # monthly_std_morn[0] = monthly_std_morn[0][index1]
    # monthly_avg_morn[1] = monthly_avg_morn[1][index2]
    # monthly_std_morn[1] = monthly_std_morn[1][index2]
    # monthly_avg_morn[2] = monthly_avg_morn[2][index3]
    # monthly_std_morn[2] = monthly_std_morn[2][index3]
    # monthly_avg_morn[3] = monthly_avg_morn[3][index4]
    # monthly_std_morn[3] = monthly_std_morn[3][index4]
    #
    # # take average over time axis, want a 4 x 1238 array for each
    # monthly_avg_aft = np.multiply(np.mean(total_aft, axis=2), 1.6)
    # monthly_std_aft = np.multiply(np.std(total_aft, axis=2), 1.6)
    # # monthly_avg_morn[0] /= float(np.max(monthly_avg_morn[0]))
    # # monthly_std_morn[0] /= float(np.max(monthly_std_morn[0]))
    # index1 = np.argsort(monthly_avg_aft[0])
    # index2 = np.argsort(monthly_avg_aft[1])
    # index3 = np.argsort(monthly_avg_aft[2])
    # index4 = np.argsort(monthly_avg_aft[3])
    # monthly_avg_aft[0] = monthly_avg_aft[0][index1]
    # monthly_std_aft[0] = monthly_std_aft[0][index1]
    # monthly_avg_aft[1] = monthly_avg_aft[1][index2]
    # monthly_std_aft[1] = monthly_std_aft[1][index2]
    # monthly_avg_aft[2] = monthly_avg_aft[2][index3]
    # monthly_std_aft[2] = monthly_std_aft[2][index3]
    # monthly_avg_aft[3] = monthly_avg_aft[3][index4]
    # monthly_std_aft[3] = monthly_std_aft[3][index4]
    #
    #
    # # take average over time axis, want a 4 x 1238 array for each
    # monthly_avg_eve = np.multiply(np.mean(total_eve, axis=2), 2)
    # monthly_std_eve = np.multiply(np.std(total_eve, axis=2), 2)
    # # monthly_avg_morn[0] /= float(np.max(monthly_avg_morn[0]))
    # # monthly_std_morn[0] /= float(np.max(monthly_std_morn[0]))
    # index1 = np.argsort(monthly_avg_eve[0])
    # index2 = np.argsort(monthly_avg_eve[1])
    # index3 = np.argsort(monthly_avg_eve[2])
    # index4 = np.argsort(monthly_avg_eve[3])
    # monthly_avg_eve[0] = monthly_avg_eve[0][index1]
    # monthly_std_eve[0] = monthly_std_eve[0][index1]
    # monthly_avg_eve[1] = monthly_avg_eve[1][index2]
    # monthly_std_eve[1] = monthly_std_eve[1][index2]
    # monthly_avg_eve[2] = monthly_avg_eve[2][index3]
    # monthly_std_eve[2] = monthly_std_eve[2][index3]
    # monthly_avg_eve[3] = monthly_avg_eve[3][index4]
    # monthly_std_eve[3] = monthly_std_eve[3][index4]
    #
    # monthly_avg_night = np.mean(total_night, axis=2)
    # monthly_std_night = np.std(total_night, axis=2)
    # # monthly_avg_night[0] /= float(np.max(monthly_avg_night[0]))
    # # monthly_std_night[0] /= float(np.max(monthly_std_night[0]))
    # index1 = np.argsort(monthly_avg_night[0])
    # index2 = np.argsort(monthly_avg_night[1])
    # index3 = np.argsort(monthly_avg_night[2])
    # index4 = np.argsort(monthly_avg_night[3])
    # monthly_avg_night[0] = monthly_avg_night[0][index1]
    # monthly_std_night[0] = monthly_std_night[0][index1]
    # monthly_avg_night[1] = monthly_avg_night[1][index2]
    # monthly_std_night[1] = monthly_std_night[1][index2]
    # monthly_avg_night[2] = monthly_avg_night[2][index3]
    # monthly_std_night[2] = monthly_std_night[2][index3]
    # monthly_avg_night[3] = monthly_avg_night[3][index4]
    # monthly_std_night[3] = monthly_std_night[3][index4]


    # # Plot of node vs total outgoing volume
    # plt.plot(range(1237), monthly_avg_morn[0][0:1237], '-', label='morn', linewidth=0.1)
    # plt.plot(range(1237), monthly_avg_aft[0][0:1237], '-', label='aft', linewidth=0.2)
    # plt.plot(range(1237), monthly_avg_eve[0][0:1237], '-', label='eve', linewidth=0.5)
    # plt.plot(range(1237), monthly_avg_night[0][0:1237], '-', label='night', linewidth=1)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Outgoing Call Volume", fontsize=16)
    # plt.title("Outgoing call volume for each node - 5 months", fontsize=16)
    # plt.legend()
    # plt.grid()
    # plt.show()
    #
    # plt.figure(1, facecolor='white')
    #
    # plt.subplot(2,2,1)
    # plt.errorbar(range(1237), monthly_avg_morn[0][0:1237], yerr=monthly_std_morn[0][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlim(0, 1237)
    # plt.ylim(-2000, 12000)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Outgoing Call Volume", fontsize=16)
    # plt.title("Outgoing call volume: Morning", fontsize=16)
    # plt.grid()
    #
    # plt.subplot(2,2,2)
    # plt.errorbar(range(1237), monthly_avg_aft[0][0:1237], yerr=monthly_std_aft[0][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Outgoing Call Volume", fontsize=16)
    # plt.title("Outgoing call volume: Afternoon", fontsize=16)
    # plt.xlim(0, 1237)
    # plt.ylim(-2000, 12000)
    # plt.grid()
    #
    # plt.subplot(2,2,3)
    # plt.errorbar(range(1237), monthly_avg_eve[0][0:1237], yerr=monthly_std_eve[0][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Outgoing Call Volume", fontsize=16)
    # plt.title("Outgoing call volume: Evening", fontsize=16)
    # plt.xlim(0, 1237)
    # plt.ylim(-2000, 12000)
    # plt.grid()
    #
    # plt.subplot(2,2,4)
    # plt.errorbar(range(1237), monthly_avg_night[0][0:1237], yerr=monthly_std_night[0][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Volume of Outgoing Calls", fontsize=16)
    # plt.title("Outgoing Call Volume: Night", fontsize=16)
    # plt.xlim(0, 1237)
    # plt.ylim(-2000, 12000)
    # plt.grid()
    #
    # plt.show()
    #
    #
    #
    #
    #
    #
    # ''' Plot 2 '''
    # plt.figure(1, facecolor='white')
    #
    # plt.subplot(2,2,1)
    # plt.errorbar(range(1237), monthly_avg_morn[1][0:1237], yerr=monthly_std_morn[1][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlim(0, 1237)
    # plt.ylim(-2000, 12000)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Incoming Call Volume", fontsize=16)
    # plt.title("Incoming call volume: Morning", fontsize=16)
    # plt.grid()
    #
    # plt.subplot(2,2,2)
    # plt.errorbar(range(1237), monthly_avg_aft[1][0:1237], yerr=monthly_std_aft[1][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Incoming Call Volume", fontsize=16)
    # plt.title("Incoming call volume: Afternoon", fontsize=16)
    # plt.xlim(0, 1237)
    # plt.ylim(-2000, 12000)
    # plt.grid()
    #
    # plt.subplot(2,2,3)
    # plt.errorbar(range(1237), monthly_avg_eve[1][0:1237], yerr=monthly_std_eve[1][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Incoming Call Volume", fontsize=16)
    # plt.title("Incoming call volume: Evening", fontsize=16)
    # plt.xlim(0, 1237)
    # plt.ylim(-2000, 12000)
    # plt.grid()
    #
    # plt.subplot(2,2,4)
    # plt.errorbar(range(1237), monthly_avg_night[1][0:1237], yerr=monthly_std_night[1][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Incoming Call Volume", fontsize=16)
    # plt.title("Incoming Call Volume: Night", fontsize=16)
    # plt.xlim(0, 1237)
    # plt.ylim(-2000, 12000)
    # plt.grid()
    #
    # plt.show()

    # ''' Plot 3 '''
    #
    # plt.figure(1, facecolor='white')
    #
    # plt.subplot(2,2,1)
    # plt.errorbar(range(1237), monthly_avg_morn[2][0:1237], yerr=monthly_std_morn[2][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlim(0, 1237)
    # plt.ylim(-300000, 2000000)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Outgoing Call Duration (seconds)", fontsize=16)
    # plt.title("Outgoing Call Duration: Morning", fontsize=16)
    # plt.grid()
    #
    # plt.subplot(2,2,2)
    # plt.errorbar(range(1237), monthly_avg_aft[2][0:1237], yerr=monthly_std_aft[2][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Outgoing Call Duration (seconds)", fontsize=16)
    # plt.title("Outgoing Call Duration: Afternoon", fontsize=16)
    # plt.xlim(0, 1237)
    # plt.ylim(-300000, 2000000)
    # plt.grid()
    #
    # plt.subplot(2,2,3)
    # plt.errorbar(range(1237), monthly_avg_eve[2][0:1237], yerr=monthly_std_eve[2][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Outgoing Call Duration (seconds)", fontsize=16)
    # plt.title("Outgoing Call Duration: Evening", fontsize=16)
    # plt.xlim(0, 1237)
    # plt.ylim(-300000, 2000000)
    # plt.grid()
    #
    # plt.subplot(2,2,4)
    # plt.errorbar(range(1237), monthly_avg_night[2][0:1237], yerr=monthly_std_night[2][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Outgoing Call Duration (seconds)", fontsize=16)
    # plt.title("Outgoing Call Duration: Night", fontsize=16)
    # plt.xlim(0, 1237)
    # plt.ylim(-300000, 2000000)
    # plt.grid()
    #
    # plt.show()
    #
    # ''' Plot 4 '''
    #
    # plt.figure(1, facecolor='white')
    #
    # plt.subplot(2,2,1)
    # plt.errorbar(range(1237), monthly_avg_morn[3][0:1237], yerr=monthly_std_morn[3][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlim(0, 1237)
    # plt.ylim(-300000, 2000000)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Incoming Call Duration", fontsize=16)
    # plt.title("Incoming Call Duration: Morning", fontsize=16)
    # plt.grid()
    #
    # plt.subplot(2,2,2)
    # plt.errorbar(range(1237), monthly_avg_aft[3][0:1237], yerr=monthly_std_aft[3][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Incoming Call Duration", fontsize=16)
    # plt.title("Incoming Call Duration: Afternoon", fontsize=16)
    # plt.xlim(0, 1237)
    # plt.ylim(-300000, 2000000)
    # plt.grid()
    #
    # plt.subplot(2,2,3)
    # plt.errorbar(range(1237), monthly_avg_eve[3][0:1237], yerr=monthly_std_eve[3][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Incoming Call Duration", fontsize=16)
    # plt.title("Incoming Call Duration: Evening", fontsize=16)
    # plt.xlim(0, 1237)
    # plt.ylim(-300000, 2000000)
    # plt.grid()
    #
    # plt.subplot(2,2,4)
    # plt.errorbar(range(1237), monthly_avg_night[3][0:1237], yerr=monthly_std_night[3][0:1237], fmt='--', elinewidth=0.1)
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Volume of Outgoing Calls", fontsize=16)
    # plt.title("Outgoing Call Volume: Night", fontsize=16)
    # plt.xlim(0, 1237)
    # plt.ylim(-300000, 2000000)
    # plt.grid()
    #
    # plt.show()



    #
    # # Plot of node vs total incoming volume
    # plt.plot(range(1237), monthly_avg_morn[1][0:1237], '-', label='morn')
    # plt.plot(range(1237), monthly_avg_aft[1][0:1237], '-', label='aft')
    # plt.plot(range(1237), monthly_avg_eve[1][0:1237], '-', label='eve')
    # plt.plot(range(1237), monthly_avg_night[1][0:1237], '-', label='night')
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Volume of Incoming Calls", fontsize=16)
    # plt.title("Incoming call volume for each node - 5 months", fontsize=16)
    # plt.legend()
    # plt.grid()
    # plt.show()
    #
    # # Plot of node vs total outgoing duration
    # plt.plot(range(1237), monthly_avg_morn[2][0:1237], '-', label='morn')
    # plt.plot(range(1237), monthly_avg_aft[2][0:1237], '-', label='aft')
    # plt.plot(range(1237), monthly_avg_eve[2][0:1237], '-', label='eve')
    # plt.plot(range(1237), monthly_avg_night[2][0:1237], '-', label='night')
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Duration of Outgoing Calls (seconds)", fontsize=16)
    # plt.title("Outgoing call duration for each node - 5 months", fontsize=16)
    # plt.legend()
    # plt.grid()
    # plt.show()
    #
    # # Plot of node vs total incoming duration
    # plt.plot(range(1237), monthly_avg_morn[3][0:1237], '-', label='morn')
    # plt.plot(range(1237), monthly_avg_aft[3][0:1237], '-', label='aft')
    # plt.plot(range(1237), monthly_avg_eve[3][0:1237], '-', label='eve')
    # plt.plot(range(1237), monthly_avg_night[3][0:1237], '-', label='night')
    # plt.xlabel("Node ID", fontsize=16)
    # plt.ylabel("Duration of Incoming Calls (seconds)", fontsize=16)
    # plt.title("Incoming call duration for each node - 5 months", fontsize=16)
    # plt.legend()
    # plt.grid()
    # plt.show()


    #
    # plt.errorbar(range(1237), monthly_avg_night[0][0:1237], yerr=monthly_std_night[0][0:1237], fmt='--')
    # plt.grid()
    # plt.show()
