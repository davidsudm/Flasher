#! /anaconda3/envs/Flasher/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import find_intersection as inter

proton_scaling = 1.5

version = 'v5'
colors = ['#1f77b4', '#ff7f0e', '#2ac02c', '#d62728', '#9467bd', '#17becf']

path_proton = '/Users/lonewolf/Desktop/test_proton/new_script/'

proton_all = 'proton_rate_tel_0.dat'
proton_3 = 'proton_rate_tel_3.dat'
proton_4 = 'proton_rate_tel_4.dat'
proton_5 = 'proton_rate_tel_5.dat'
proton_6 = 'proton_rate_tel_6.dat'

proton_rate_file = [path_proton + proton_all,
                    path_proton + proton_3,
                    path_proton + proton_4,
                    path_proton + proton_5,
                    path_proton + proton_6]

proton_thresholds = []
proton_rates = []

config_labels = ['Ave. per telescope (from array)', 'Telescope 3', 'Telescope 4', 'Telescope 5', 'Telescope 6']

# LOADING PROTON DATA (for 5 different trigger telescope array config.)
for i, file in enumerate(proton_rate_file):
    with open(file, 'r') as f:
        temp_threshold, temp_rates = np.loadtxt(file, delimiter=' ', usecols=(0, 1), unpack=True)
        proton_thresholds.append(temp_threshold)
        proton_rates.append(temp_rates)

proton_thresholds = np.array(proton_thresholds)
proton_rates = np.array(proton_rates)

# Arranging by increasing thresholds an array value
# Protons
for i in range(len(proton_rate_file)):
    ind = np.argsort(proton_thresholds[i])
    proton_thresholds[i] = (proton_thresholds[i])[ind]
    proton_rates[i] = (proton_rates[i])[ind]

ave_rate = np.zeros_like(proton_rates[0])
ave_threshold = np.zeros_like(proton_rates[0])
for i in range(len(proton_rate_file)):
    if i is not 0:
        ave_rate += proton_rates[i]
        ave_threshold += proton_thresholds[i]
ave_rate /= i
ave_threshold /= i

# Only protons rates
fig, ax = plt.subplots()
for i in range(len(proton_rate_file)):
    ax.semilogy(proton_thresholds[i], proton_scaling * proton_rates[i], label=config_labels[i], color=colors[i])
ax.semilogy(ave_threshold, proton_scaling * ave_rate, label='average rate from 3, 4, 5, 6', color='black', linestyle='dashed')
ax.legend(fontsize=10, loc=1)
ax.set_ylabel('Rate [Hz]')
ax.set_xlabel('Threshold [mV]')
plt.savefig('/Users/lonewolf/Desktop/test_proton/proton_rate_{}.pdf'.format(version))
plt.show()
