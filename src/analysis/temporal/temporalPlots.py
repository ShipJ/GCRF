# Jack Shipway October 2016 UCL GCRF Research Project
#
# This file
#

import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

if __name__ == "__main__":
    total_w = np.load("IvoryCoastData/CDR/AggregateData/total_working.npy")
    total_nw = np.load("IvoryCoastData/CDR/AggregateData/total_nworking.npy")
    rainfall = pd.DataFrame(pd.read_csv("IvoryCoastData/Geography/ICRainfall11-12.csv"), index=None)

    output_mean_rural_w = []
    output_std_rural_w = []
    output_mean_urban_w = []
    output_std_urban_w = []
    output_mean_rural_nw = []
    output_std_rural_nw = []
    output_mean_urban_nw = []
    output_std_urban_nw = []
    for i in range(140):
        data_w = np.where(total_w[0, 0:1237, i] > 0, np.log(total_w[0, 0:1237, i]), total_w[0, 0:1237, i])
        rural_w = sorted(data_w)[0:600]
        urban_w = sorted(data_w)[600:]
        output_mean_rural_w.append(np.mean(rural_w))
        output_std_rural_w.append(np.std(rural_w))
        output_mean_urban_w.append(np.mean(urban_w))
        output_std_urban_w.append(np.std(urban_w))
        
        data_nw = np.where(total_nw[0, 0:1237, i] > 0, np.log(total_nw[0, 0:1237, i]), total_nw[0, 0:1237, i])
        rural_nw = sorted(data_nw)[0:600]
        urban_nw = sorted(data_nw)[600:]
        output_mean_rural_nw.append(np.mean(rural_nw))
        output_std_rural_nw.append(np.std(rural_nw))
        output_mean_urban_nw.append(np.mean(urban_nw))
        output_std_urban_nw.append(np.std(urban_nw))

    plt.errorbar(range(140), output_mean_rural_w, yerr=output_std_rural_w, alpha=0.5, color='green')
    plt.errorbar(range(140), output_mean_rural_nw, yerr=output_std_rural_nw, color='red')
    plt.errorbar(range(140), output_mean_urban_w, yerr=output_std_urban_w, alpha=0.5, color='blue')
    plt.errorbar(range(140), output_mean_urban_nw, yerr=output_std_urban_nw, color='orange')
    # plt.bar(range(14, 154, 28), rainfall['rainfall'] / max(rainfall['rainfall']), color='blue', width=10, alpha=0.5)
    plt.grid()
    plt.show()

