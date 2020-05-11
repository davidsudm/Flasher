#! /anaconda3/envs/Flasher/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import find_intersection as inter

proton_scaling = 1.5

version = 'v4_telescope_all'
colors = ['#1f77b4', '#ff7f0e', '#2ac02c', '#d62728', '#9467bd', '#17becf']


path_proton = '/Users/lonewolf/CTA/server_data/mc_rate/protons/'
proton_tel_all_f = path_proton + 'pmt_pixel-lst_window/proton_rate_v4_29229536_all_imposed.dat' # v4 telescope ON 3,4,5,6
proton_tel_3_f = path_proton + 'pmt_pixel-lst_window/proton_rate_v4_29228009_3.dat' # v4 telescope ON 3
proton_tel_4_f = path_proton + 'pmt_pixel-lst_window/proton_rate_v4_28956509_tel_3_bis.dat' # v4 telescope ON 4
proton_tel_5_f = path_proton + 'pmt_pixel-lst_window/proton_rate_v4_28952991_tel_5.dat' # v4 telescope ON 5
proton_tel_6_f = path_proton + 'pmt_pixel-lst_window/proton_rate_v4_28953627_tel_6.dat' # v4 telescope ON 6

config_labels = ['All telescopes imposed', 'Telescope 3', 'Telescope 3 old', 'Telescope 5', 'Telescope 6']
proton_files = [proton_tel_all_f, proton_tel_3_f, proton_tel_4_f, proton_tel_5_f, proton_tel_6_f]


proton_thresholds = []
proton_rates = []
proton_triggered = []
proton_total = []

# LOADING PROTON DATA (for 5 different trigger telescope array config.)
for i, file in enumerate(proton_files):
    with open(file, 'r') as f:
        temp_threshold, temp_rates, temp_triggered, temp_total = np.loadtxt(file, delimiter=' ', usecols=(0, 1, 2, 3), unpack=True)
        proton_thresholds.append(temp_threshold)
        proton_rates.append(temp_rates)
        proton_triggered.append(temp_triggered)
        proton_total.append(temp_total)

proton_thresholds = np.array(proton_thresholds)
proton_rates = np.array(proton_rates)
proton_triggered = np.array(proton_triggered)
proton_total = np.array(proton_total)

# Arranging by increasing thresholds an array value
# Protons
for i in range(len(proton_files)):
    ind = np.argsort(proton_thresholds[i])
    proton_thresholds[i] = (proton_thresholds[i])[ind]
    proton_rates[i] = (proton_rates[i])[ind]
    proton_triggered[i] = (proton_triggered[i])[ind]
    proton_total[i] = (proton_total[i])[ind]

# Proton average of the 4 telescope
n_points = len(proton_thresholds[0])
matrix = np.zeros((4, n_points))
for i, row in enumerate(matrix):
    matrix[i, :] = proton_rates[i+1]

average_rate = np.mean(matrix, axis=0)


# Only protons rates
fig, ax = plt.subplots()
for i in range(len(config_labels)):
    ax.semilogy(proton_thresholds[i], proton_rates[i], label=config_labels[i], color=colors[i])
ax.semilogy(proton_thresholds[0], average_rate, label='average of all telescope', color=colors[5])
ax.legend(fontsize=8, loc=1)
ax.set_ylabel('Rate [Hz]')
ax.set_xlabel('Threshold [mV]')
plt.savefig('/Users/lonewolf/CTA/scripts/proton_rates_{}_telescopes_log.pdf'.format(version))
plt.show()

fig, ax = plt.subplots()
for i in range(len(config_labels)):
    ax.plot(proton_thresholds[i], proton_rates[i], label=config_labels[i], color=colors[i])
ax.plot(proton_thresholds[0], average_rate, label='average of all telescope', color=colors[5])
ax.legend(fontsize=8, loc=1)
ax.set_ylabel('Rate [Hz]')
ax.set_xlabel('Threshold [mV]')
plt.savefig('/Users/lonewolf/CTA/scripts/proton_rates_{}_telescopes_lin.pdf'.format(version))
plt.show()

