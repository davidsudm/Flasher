#! /anaconda3/envs/Flasher/bin/python

from readout import read_file
from readout import read_pindiode_photo_sensitivity
from xkcd_colors import xkcd_colors

import numpy as np
import computations as comp
import matplotlib.pyplot as plt
from scipy.interpolate import spline
import matplotlib.dates as md
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.backends


file_flash = 'others/FlasherOn_16V_5kHz_5us.txt'

# OPTIONS :
illuminated_area = 0.028**2 # m^2
distance = 1.0 # m


# Getting the desired photo-sensitivity for the peak wavelength (1st approximation)
ph_wavelength, photosensitivity = read_pindiode_photo_sensitivity()

ph_wavelength_smooth = np.linspace(ph_wavelength.min(), ph_wavelength.max(), 1000)
photosensitivity_smooth = spline(ph_wavelength, photosensitivity, ph_wavelength_smooth)

pin_diode_wavelength = ph_wavelength_smooth[68]
pin_diode_pde = photosensitivity_smooth[68]

plt.plot(ph_wavelength_smooth, photosensitivity_smooth)
plt.xlabel("wavelength [nm]")
plt.ylabel("Photosensitivity [A/W]")

x, y, current, current_std, current_time, current_timestamp = read_file(path_to_file=file_flash, scan_type='time')

systematic_err = comp.calculate_systematic_error(current)
statistic_err = comp.calculate_statistic_error(current_std)

total_err = np.sqrt(systematic_err**2 + statistic_err**2)

factor = pin_diode_pde * illuminated_area
scaling = 1e-6 # to go to micro Watts

irradiance = (current / factor) / scaling
irradiance_err = (total_err[0] / factor) / scaling

initial_bin = 80
final_bin = 350

p, cov = np.polyfit(current_time[initial_bin:final_bin]/3600, irradiance[initial_bin:final_bin], deg=1, cov=True)
error_bars = np.sqrt(np.diag(cov))
x_fit = np.linspace(current_time.min()/3600, current_time.max()/3600, 1000)
y_fir = p[0] * x_fit + p[1]
time_in_seconds = current_time
current_time = current_timestamp

for i, coefficient in enumerate(p):
    p[i] = np.format_float_scientific(p[i], unique=True, precision=2)
    print(p[i])

sigma = np.sqrt(np.diag(cov))
for i, coeff_error in enumerate(sigma):
    sigma[i] = np.format_float_scientific(sigma[i], unique=True, precision=2)


fig, ax1 = plt.subplots()

ax1.plot(time_in_seconds,
         irradiance,
         color=sns.xkcd_rgb['sky blue'],
         label='Irradiance')
ax1.fill_between(time_in_seconds,
                 irradiance - irradiance_err,
                 irradiance + irradiance_err,
                 color=sns.xkcd_rgb['amber'])
ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
ax1.set_ylim(bottom=77, top=87)
ax1.set_xlabel('time [s]')
ax1.set_ylabel(r'E [$\mu W/m^2$]')

ax2 = ax1.twiny()

#xfmt = md.DateFormatter('%m/%d %H:%M')
xfmt = md.DateFormatter('%H:%M')
ax2.xaxis.set_major_formatter(xfmt)
ax2.xaxis.set_tick_params(rotation=0)
ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
ax2.tick_params(axis='x', direction='in', pad=-18)
ax2.set_xlabel('Local time')
ax2.plot(current_timestamp,
         irradiance,
         color=sns.xkcd_rgb['sky blue'])

# These are in unitless percentages of the figure size. (0,0 is bottom left)
left, bottom, width, height = [0.22, 0.36, 0.5, 0.4]
ax3 = fig.add_axes([left, bottom, width, height])

ax3.xaxis.set_tick_params(rotation=0)
ax3.xaxis.set_major_locator(plt.MaxNLocator(2))
ax3.yaxis.set_major_locator(plt.MaxNLocator(3))
ax3.plot(time_in_seconds[initial_bin: final_bin],
         irradiance[initial_bin:final_bin],
         color=sns.xkcd_rgb['sky blue'])
ax3.fill_between(time_in_seconds[initial_bin: final_bin],
                 (irradiance - irradiance_err)[initial_bin:final_bin],
                 (irradiance + irradiance_err)[initial_bin:final_bin],
                 color=sns.xkcd_rgb['amber'])
ax3.plot(time_in_seconds[initial_bin: final_bin],
         y_fir[initial_bin:final_bin],
         color='green')

fit_label = 'Linear fit : $E = m*time + b$\n m = {} ± {} \n b = {} ± {}'.format(p[0], sigma[0], p[1], sigma[1])

ax3.set_ylim(bottom=76, top=79.7)

ax3.legend()
ax3.legend(frameon=False)

sky_blue_line = mpatches.Patch(color=sns.xkcd_rgb['sky blue'], label='Irradiance')
amber_patch = mpatches.Patch(color=sns.xkcd_rgb['amber'], label='Irradiance', linewidth=7.0)
green_fit_line = mpatches.Patch(color=sns.xkcd_rgb['green'], label='fit', linewidth=0)

ax3.legend([(amber_patch, sky_blue_line), green_fit_line], ['Irradiance', fit_label], frameon=False)

figure_name = './Output/Others/stability_in_time_pt2.png'
plt.savefig(figure_name)



fig, ax = plt.subplots()

fit_label = 'Linear fit : $E = m * time + b$\n m = {} ± {} [$\mu W / m^2 h$]\n b = {} ± {} [$\mu W / m^2$]'.format(p[0], sigma[0], p[1], sigma[1])
data_label = 'Irradiance'

time_in_seconds = time_in_seconds / 3600.

ax.xaxis.set_tick_params(rotation=0)
#ax.xaxis.set_major_locator(plt.MaxNLocator(2))
ax.yaxis.set_major_locator(plt.MaxNLocator(6))
ax.errorbar(time_in_seconds[initial_bin: final_bin],
        irradiance[initial_bin:final_bin], irradiance_err[initial_bin:final_bin]/2,
        color=sns.xkcd_rgb['sky blue'], linestyle='None', marker='o', ecolor=sns.xkcd_rgb['amber'], ms=3, label='Irradiance')
# ax.fill_between(time_in_seconds[initial_bin: final_bin],
#                 (irradiance - irradiance_err)[initial_bin:final_bin],
#                 (irradiance + irradiance_err)[initial_bin:final_bin],
#                 color=sns.xkcd_rgb['amber'],
#                 label=data_label)
ax.plot(time_in_seconds[initial_bin: final_bin],
        y_fir[initial_bin:final_bin],
        color='green',
        label=fit_label)
ax.legend(frameon=False)

mean_value = 78.6 - 0.3
edge = 1.1
ax.set_xlabel('time [hours]')
ax.set_ylabel(r'E [$\mu W/m^2$]')
ax.set_ylim(bottom=mean_value - edge, top=mean_value + edge)


sky_blue_line = mpatches.Patch(color=sns.xkcd_rgb['sky blue'], label='Irradiance')
amber_patch = mpatches.Patch(color=sns.xkcd_rgb['amber'], label='Irradiance', linewidth=7.0)
green_fit_line = mpatches.Patch(color=sns.xkcd_rgb['green'], label=fit_label, linewidth=0)

#ax.legend([(amber_patch, sky_blue_line), green_fit_line], [data_label, fit_label], frameon=False, ncol=2, loc=9)

figure_name = './Output/Others/stability_in_time.png'
plt.savefig(figure_name)

#plt.show()
