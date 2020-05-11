#! /anaconda3/envs/Flasher/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import interpolate

path_spe = '/Volumes/GreyWind/CTA/Data/LST/LST_small_pixels/SPE_distribution'
path_template = '/Volumes/GreyWind/CTA/Data/LST/LST_small_pixels/template'

spe = ['SPE_12ns.dat', 'SPE_12ns.dat', 'SPE_20ns.dat', 'SPE_Gentile_oxt0d08_spe0d05_d2018-10-04.txt']
template = ['Template07ns_original.txt', 'Template12ns_original.txt', 'Template20ns_original.txt', 'pulse_CTA-Fx3.dat']

label_spe = ['SPE 12 ns', 'SPE 12 ns', 'SPE 20 ns', 'SPE Gentile']
label_template = ['FWHM : 07 ns', 'FWHM : 12 ns', 'FWHM : 20 ns', 'Fx3']

for i in range(len(template)):
    template[i] = os.path.join(path_template, template[i])
    spe[i] = os.path.join(path_spe, spe[i])

pdf = PdfPages("/Volumes/GreyWind/CTA/Data/LST/LST_small_pixels/spe_dist_and_templates.pdf")

pe_vec = []
prob_vec = []

time_vec = []
amp_vec = []

time_inter_vec = []
amp_inter_vec = []

for i in range(len(template)):

    with open(spe[i], 'r') as f:
        pe, prob = np.loadtxt(spe[i], delimiter=None, usecols=(0, 1), unpack=True)

    with open(template[i], 'r') as f:
        time, amp = np.loadtxt(template[i], delimiter=None, usecols=(0, 1), unpack=True)

    pe_vec.append(pe)
    prob_vec.append(prob)
    time_vec.append(time)
    amp_vec.append(amp)

for i in range(len(template)):

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(time_vec[i], amp_vec[i], label=label_template[i])
    ax1.legend()
    ax1.set_ylabel('Normalized Amplitude')
    ax1.set_xlabel('Time [ns]')

    ax2.semilogy(pe_vec[i], prob_vec[i], label=label_spe[i])
    ax2.legend()
    ax2.set_ylim(1e-8, 10)
    ax2.set_xlabel('Number of p.e')
    ax2.set_ylabel('SPE distribution')

    plt.tight_layout()
    pdf.savefig(fig)

for i in range(len(template)-1):

    f = interpolate.interp1d(time_vec[i], amp_vec[i])
    time_temp = np.arange(0.0, 100., 0.2)
    amp_temp = f(time_temp)
    time_inter_vec.append(time_temp)
    amp_inter_vec.append(amp_temp)

for i in range(len(template)-1):

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(time_inter_vec[i], amp_inter_vec[i], label=label_template[i])
    ax1.legend()
    ax1.set_ylabel('Normalized Amplitude')
    ax1.set_xlabel('Time [ns]')

    ax2.semilogy(pe_vec[i], prob_vec[i], label=label_spe[i])
    ax2.legend()
    ax2.set_ylim(1e-8, 10)
    ax2.set_xlabel('Number of p.e')
    ax2.set_ylabel('SPE distribution')

    plt.tight_layout()
    pdf.savefig(fig)


for i in range(len(template)-1):

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(time_inter_vec[i], amp_inter_vec[i], label=label_template[i])
    ax1.legend()
    ax1.set_ylabel('Normalized Amplitude')
    ax1.set_xlabel('Time [ns]')

    ax2.semilogy(pe_vec[i], prob_vec[i], label=label_spe[i])
    ax2.legend()
    ax2.set_ylim(1e-8, 10)
    ax2.set_xlabel('Number of p.e')
    ax2.set_ylabel('SPE distribution')

    plt.tight_layout()
    pdf.savefig(fig)


fig, (ax1, ax2) = plt.subplots(2, 1)
line_style = ['solid', 'dashed', 'solid']
for i in range(len(time_inter_vec)):

    ax1.plot(time_inter_vec[i], amp_inter_vec[i], label=label_template[i], linestyle=line_style[i])
    ax2.semilogy(pe_vec[i], prob_vec[i], label=label_spe[i], linestyle=line_style[i])

ax1.legend()
ax2.legend()
ax2.set_ylim(1e-8, 10)

ax1.set_ylabel('Normalized Amplitude')
ax1.set_xlabel('Time [ns]')
ax2.set_xlabel('Number of p.e')
ax2.set_ylabel('SPE distribution')

plt.tight_layout()
pdf.savefig(fig)
pdf.close()


for i in range(len(template)-1):
    file_name = template[i].split('_original')[0] + template[i].split('_original')[1]
    np.savetxt(file_name, np.c_[time_inter_vec[i], amp_inter_vec[i]], fmt=['%.2f', '%10.10f'], header='Waveform with ' + label_template[i])

