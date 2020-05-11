#! /anaconda3/envs/Flasher/bin/python

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.lines as lines
import find_intersection as inter

Nbins = 55
freq_adc = np.array([1e9, 1e9, 0.5e9])

T_event = Nbins / freq_adc
print('f_ADC / n_ADC bins = ', freq_adc/Nbins)
print('n_ADC bins / f_ADC  = ', Nbins/freq_adc)

nsb_path = '/Users/lonewolf/CTA/sshfs_disks/baobab/data'

nsb_config_label = ['NSBx2 rate : config 1', 'NSBx2 rate : config 2', 'NSBx2 rate : config 3']

debug_files1 = ['simtel_NSB_rate_config_1/simtel/trigger_rate_proton.dat',
                'simtel_NSB_rate_config_2/simtel/trigger_rate_proton.dat',
                'simtel_NSB_rate_config_3/simtel/trigger_rate_proton.dat']

debug_files2 = ['trigger_rate_2xNSB_0.386GHz_jobid_31827668.dat',
               'trigger_rate_2xNSB_0.386GHz_jobid_31831800.dat',
               'trigger_rate_2xNSB_0.386GHz_jobid_31832043.dat']

debug_files3 = ['trigger_rate_1xNSB_0.386GHz_jobid_31831787.dat',
               'trigger_rate_1xNSB_0.386GHz_jobid_31832304.dat']

debug_files4 = ['simtel_NSB_rate_config_1/simtel/trigger_rate_2xNSB_0.386GHz_jobid_31836519.dat',
                'simtel_NSB_rate_config_2/simtel/trigger_rate_2xNSB_0.386GHz_jobid_31836533.dat',
                'simtel_NSB_rate_config_3/simtel/trigger_rate_2xNSB_0.386GHz_jobid_31836534.dat']

debug_files5 = ['simtel_NSB_rate_config_1/simtel/trigger_rate_2xNSB_0.386GHz_jobid_31836641.dat',
                'simtel_NSB_rate_config_2/simtel/trigger_rate_2xNSB_0.386GHz_jobid_31836667.dat']
#                'simtel_NSB_rate_config_3/simtel/trigger_rate_2xNSB_0.386GHz_jobid_31836669.dat']

factor = 1.5
proton_path = '/Users/lonewolf/CTA/sshfs_disks/baobab/data'

proton_config_label = ['proton rate : config 1', 'proton rate : config 2', 'proton rate : config 3']

debug_files = debug_files4

label_names = []


fig_rate, ax0 = plt.subplots()
fig_triggered, ax1 = plt.subplots()
fig_event, ax2 = plt.subplots()
fig_prob, ax3 = plt.subplots()

# LOADING NSBx2 DATA PMT PIXELS (both windows)
for i, file in enumerate(debug_files):

    file = os.path.join(nsb_path, debug_files[i])
    with open(file, 'r') as f:
        threshold, triggered, events = np.loadtxt(file, delimiter=' ', usecols=(0, 1, 2), unpack=True)

    threshold = np.array(threshold)
    triggered = np.array(triggered)
    events = np.array(events)

    ind = np.argsort(threshold)
    threshold = threshold[ind]
    triggered = triggered[ind]
    events = events[ind]

    nsb_rate = (triggered / events) * (1 / T_event[i])
    nsb_rate = np.array(nsb_rate)

    ax0.semilogy(threshold, nsb_rate, label=nsb_config_label[i])
    #ax0.semilogy(threshold, nsb_rate, label=nsb_config_label[i], linestyle='None', marker='o')
    #ax0.plot(threshold, nsb_rate, label=nsb_config_label[i])
    ax1.plot(threshold, triggered, label=nsb_config_label[i])
    ax2.plot(threshold, events, label=nsb_config_label[i])
    ax3.plot(threshold, triggered/events, label=nsb_config_label[i])

    ax0.set_xlabel('Threshold [LSB]')
    ax0.set_ylabel('Rate [Hz]')
    ax1.set_xlabel('Threshold [LSB]')
    ax1.set_ylabel('Triggered events')
    ax2.set_xlabel('Threshold [LSB]')
    ax2.set_ylabel('Total events')
    ax3.set_xlabel('Threshold [LSB]')
    ax3.set_ylabel('Trigger probability')

    #ax0.legend()
    ax1.legend()
    ax2.legend()
    ax3.legend()

debug_files1 = ['simtel_NSB_rate_config_1/simtel/trigger_rate_proton.dat',
                'simtel_NSB_rate_config_2/simtel/trigger_rate_proton.dat',
                'simtel_NSB_rate_config_3/simtel/trigger_rate_proton.dat']

debug_files = debug_files1
# LOADING NSBx2 DATA PMT PIXELS (both windows)
for i, file in enumerate(debug_files):

    file = os.path.join(proton_path, debug_files[i])
    with open(file, 'r') as f:
        threshold, rate = np.loadtxt(file, delimiter=' ', usecols=(0, 1), unpack=True)

    threshold = np.array(threshold)
    rate = np.array(rate)

    ind = np.argsort(threshold)
    threshold = threshold[ind]
    rate = rate[ind]
    ax0.semilogy(threshold, factor * rate, label=proton_config_label[i])

    ax0.legend()



pdf = PdfPages("/Users/lonewolf/Desktop/nsb_rates.pdf")

fig_rate.tight_layout()
fig_event.tight_layout()
fig_triggered.tight_layout()
fig_prob.tight_layout()

pdf.savefig(fig_event)
pdf.savefig(fig_triggered)
pdf.savefig(fig_prob)
pdf.savefig(fig_rate)

pdf.close()