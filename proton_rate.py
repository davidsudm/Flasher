#! /anaconda3/envs/Flasher/bin/python

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.lines as lines
import find_intersection as inter

factor = 1.5
proton_path = '/Users/lonewolf/CTA/sshfs_disks/baobab/data'

nsb_config_label = ['proton rate : config 1', 'proton rate : config 2', 'proton rate : config 2']

debug_files1 = ['simtel_NSB_rate_config_1/simtel/trigger_rate_proton.dat',
                'simtel_NSB_rate_config_2/simtel/trigger_rate_proton.dat',
                'simtel_NSB_rate_config_3/simtel/trigger_rate_proton.dat']

debug_files = debug_files1


fig_rate, ax0 = plt.subplots()

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
    ax0.semilogy(threshold, factor * rate, label=nsb_config_label[i])
    ax0.set_xlabel('Threshold [LSB]')
    ax0.set_ylabel('Rate [Hz]')

    ax0.legend()

pdf = PdfPages("/Users/lonewolf/Desktop/nsb_rates.pdf")

fig_rate.tight_layout()

pdf.savefig(fig_rate)

pdf.close()