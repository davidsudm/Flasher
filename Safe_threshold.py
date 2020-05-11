#! /anaconda3/envs/Flasher/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import find_intersection as inter

proton_scaling = 1.5
Nbins = 50
fadc = 1e9
T_event = Nbins / fadc

#version = 'v2'
#version = 'v3_telescope_on'
#version = 'v3_telescope_off'
version = 'v4_telescope_ON_3456'
colors = ['#1f77b4', '#ff7f0e', '#2ac02c', '#d62728', '#9467bd']

# This is where I store the file on my personal computer
path_proton = '/Users/lonewolf/CTA/server_data/mc_rate/protons/'

proton_pmt_lst_window_f = path_proton + 'pmt_pixel-lst_window/proton_rate_v2_26478354.dat'
proton_pmt_sst1m_window_f = path_proton + 'pmt_pixel-sst1m_window/proton_rate_v2_26478642.dat'
proton_sipm_lst_window_f = path_proton + 'sipm_pixel-lst_window/proton_rate_v2_26479175_26581900.dat'
proton_sipm_sst1m_window_f = path_proton + 'sipm_pixel-sst1m_window/proton_rate_v2_26479176.dat'

# This is where I store the file on my personal computer
path_nsb = '/Users/lonewolf/CTA/server_data/mc_rate/nsb/'

nsbx1_pmt_lst_window_f = path_nsb + 'pmt_pixel-lst_window/trigger_rate_1xNSB_0.25539GHz_jobid_25927697.dat'
nsbx2_pmt_lst_window_f = path_nsb + 'pmt_pixel-lst_window/trigger_rate_2xNSB_0.25539GHz_jobid_25927698.dat'
#nsbx2_pmt_lst_window_f = '/Users/lonewolf/Desktop/pmt_pixel_lst_filter_nsbx2-1663547.dat'
nsbx1_pmt_sst1m_window_f = path_nsb + 'pmt_pixel-sst1m_window/trigger_rate_1xNSB_0.21341GHz_jobid_25936823.dat'
nsbx2_pmt_sst1m_window_f = path_nsb + 'pmt_pixel-sst1m_window/trigger_rate_2xNSB_0.21341GHz_jobid_25936824.dat'
nsbx1_sipm_lst_window_f = path_nsb + 'sipm_pixel-lst_window/trigger_rate_1xNSB_0.386GHz_jobid_25044321.dat'
nsbx2_sipm_lst_window_f = path_nsb + 'sipm_pixel-lst_window/trigger_rate_2xNSB_0.386GHz_jobid_25044323.dat'
nsbx1_sipm_sst1m_window_f = path_nsb + 'sipm_pixel-sst1m_window/trigger_rate_1xNSB_0.1076GHz_jobid_25044326.dat'
nsbx2_sipm_sst1m_window_f = path_nsb + 'sipm_pixel-sst1m_window/trigger_rate_2xNSB_0.1076GHz_jobid_25044329.dat'

config_labels = ['PMT pixel, LST window', 'PMT pixel, SST1M window', 'SiPM pixel, LST window', 'SiPM pixel, SST1M window']
proton_files = [proton_pmt_lst_window_f, proton_pmt_sst1m_window_f, proton_sipm_lst_window_f, proton_sipm_sst1m_window_f]

nsb_names = ['PMT pixel, LST window', 'PMT pixel, SST1M window', 'SiPM pixel, LST window', 'SiPM pixel, SST1M window']
nsbx1_files = [nsbx1_pmt_lst_window_f, nsbx1_pmt_sst1m_window_f, nsbx1_sipm_lst_window_f, nsbx1_sipm_sst1m_window_f]
nsbx2_files = [nsbx2_pmt_lst_window_f, nsbx2_pmt_sst1m_window_f, nsbx2_sipm_lst_window_f, nsbx2_sipm_sst1m_window_f]

proton_thresholds = []
proton_rates = []
proton_triggered = []
proton_total = []

nsbx1_thresholds = []
nsbx1_rates = []
nsbx1_triggered = []
nsbx1_total = []

nsbx2_thresholds = []
nsbx2_rates = []
nsbx2_triggered = []
nsbx2_total = []

# LOADING PROTON DATA PMT PIXELS and SiPM PIXELS (both windows)
for i, file in enumerate(proton_files):
    with open(file, 'r') as f:
        temp_threshold, temp_rates, temp_triggered, temp_total = np.loadtxt(file, delimiter=' ', usecols=(0, 1, 2, 3), unpack=True)
        proton_thresholds.append(temp_threshold)
        proton_rates.append(temp_rates)
        proton_triggered.append(temp_triggered)
        proton_total.append(temp_total)

# LOADING NSBx1 DATA PMT PIXELS (both windows)
for i, file in enumerate(nsbx1_files):
    with open(file, 'r') as f:
        temp_threshold, temp_triggered, temp_total = np.loadtxt(file, delimiter=' ', usecols=(2, 1, 0), unpack=True)
        nsbx1_thresholds.append(temp_threshold)
        nsbx1_triggered.append(temp_triggered)
        nsbx1_total.append(temp_total)
        nsbx1_rates.append((np.array(temp_triggered) / np.array(temp_total)) * (1 / T_event))

# LOADING NSBx2 DATA PMT PIXELS (both windows)
for i, file in enumerate(nsbx2_files):
    with open(file, 'r') as f:
        temp_threshold, temp_triggered, temp_total = np.loadtxt(file, delimiter=' ', usecols=(2, 1, 0), unpack=True)
        #temp_threshold, temp_triggered, temp_total = np.loadtxt(file, delimiter=' ', usecols=(0, 1, 2), unpack=True)
        nsbx2_thresholds.append(temp_threshold)
        nsbx2_triggered.append(temp_triggered)
        nsbx2_total.append(temp_total)
        nsbx2_rates.append((np.array(temp_triggered) / np.array(temp_total)) * (1/T_event))

proton_thresholds = np.array(proton_thresholds)
proton_rates = np.array(proton_rates)
proton_triggered = np.array(proton_triggered)
proton_total = np.array(proton_total)

nsbx1_thresholds = np.array(nsbx1_thresholds)
nsbx1_rates = np.array(nsbx1_rates)
nsbx1_triggered = np.array(nsbx1_triggered)
nsbx1_total = np.array(nsbx1_total)

nsbx2_thresholds = np.array(nsbx2_thresholds)
nsbx2_rates = np.array(nsbx2_rates)
nsbx2_triggered = np.array(nsbx2_triggered)
nsbx2_total = np.array(nsbx2_total)

# Arranging by increasing thresholds an array value
# Protons
for i in range(len(proton_files)):
    ind = np.argsort(proton_thresholds[i])
    proton_thresholds[i] = (proton_thresholds[i])[ind]
    proton_rates[i] = (proton_rates[i])[ind]
    proton_triggered[i] = (proton_triggered[i])[ind]
    proton_total[i] = (proton_total[i])[ind]

# NSBx1
for i in range(len(nsbx1_files)):
    ind = np.argsort(nsbx1_thresholds[i])
    nsbx1_thresholds[i] = (nsbx1_thresholds[i])[ind]
    nsbx1_triggered[i] = (nsbx1_triggered[i])[ind]
    nsbx1_total[i] = (nsbx1_total[i])[ind]
    nsbx1_rates[i] = (nsbx1_rates[i])[ind]

# NSBx2
for i in range(len(nsbx2_files)):
    ind = np.argsort(nsbx2_thresholds[i])
    nsbx2_thresholds[i] = (nsbx2_thresholds[i])[ind]
    nsbx2_triggered[i] = (nsbx2_triggered[i])[ind]
    nsbx2_total[i] = (nsbx2_total[i])[ind]
    nsbx2_rates[i] = (nsbx2_rates[i])[ind]

# Finding intersections
x_int = []
y_int = []

for i in range(len(proton_files)):
    x_temp, y_temp = inter.intersection(nsbx2_thresholds[i], nsbx2_rates[i],
                                        proton_thresholds[i], proton_rates[i])
    print('{} --> threshold = {} , rate = {}'.format(config_labels[i], x_temp, y_temp))
    x_int.append(x_temp)
    y_int.append(y_temp)

# Plotting results :

# Safe threshold final for each configuration
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
x_pos = 0.98
y_pos = 1.07
ax = [ax1, ax2, ax3, ax4]
for i, ax in enumerate(ax):
    ax.semilogy(nsbx2_thresholds[i], nsbx2_rates[i], color=colors[i], label='NSB x 2', linestyle='dashed')
    ax.semilogy(proton_thresholds[i], proton_rates[i], color=colors[i], label='proton x 1.5')
    ax.plot(x_int[i], y_int[i], color='black', marker='*', linestyle='None', alpha=0.5)
    #ax.axvline(x=x_int[i], linestyle='dotted', color='gray')
    #ax.axhline(y=y_int[i], linestyle='dotted', color='gray')
    ax.text(x_pos, y_pos, config_labels[i], transform=ax.transAxes, fontsize=8, va='center_baseline', ha='right', bbox=props)
    ax.set_xlim(400, 2000)
    ax.set_xticks(range(400, 2001, 100), minor=True)
    ax.legend(fontsize=8, loc=1)
ax1.set_ylabel('Rate [Hz]')
ax3.set_ylabel('Rate [Hz]')
ax3.set_xlabel('Threshold [mV]')
ax4.set_xlabel('Threshold [mV]')
plt.savefig('/Users/lonewolf/CTA/scripts/safe_threshold_subplots_{}.pdf'.format(version))
plt.show()

# Only protons rates
fig, ax = plt.subplots()
for i in range(len(config_labels)):
    ax.semilogy(proton_thresholds[i], proton_rates[i], label=config_labels[i], color=colors[i])
ax.legend(fontsize=8, loc=1)
ax.set_ylabel('Rate [Hz]')
ax.set_xlabel('Threshold [mV]')
plt.savefig('/Users/lonewolf/CTA/scripts/proton_rates_{}.pdf'.format(version))
plt.show()



# Proton events :  Triggered events and total events
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
for i in range(len(config_labels)):
    ax1.plot(proton_thresholds[i], proton_triggered[i], label=config_labels[i], color=colors[i])
    ax2.plot(proton_thresholds[i], proton_total[i], label=config_labels[i], color=colors[i])

ax1.legend(fontsize=8, loc=1)
ax1.set_ylabel('Triggered Events')
ax2.set_xlabel('Threshold [mV]')
ax2.set_ylabel('Total Events')
plt.savefig('/Users/lonewolf/CTA/scripts/events_{}.pdf'.format(version))
plt.show()

# Proton trigger probability :
fig, ax = plt.subplots()
for i in range(len(config_labels)):
    ax.plot(proton_thresholds[i], 100*proton_triggered[i]/proton_total[i], label=config_labels[i], color=colors[i])
ax.legend(fontsize=8, loc=1)
ax.set_xlabel('Threshold [mV]')
ax.set_ylabel('Trigger probability [%]')
plt.savefig('/Users/lonewolf/CTA/scripts/trig_prob_{}.pdf'.format(version))
plt.show()
