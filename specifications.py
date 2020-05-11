#! /anaconda3/envs/Flasher/bin/python

from readout import read_file
from readout import read_cerenkov_spectra
from readout import read_pindiode_photo_sensitivity
from readout import read_pindiode_temperature_coefficient
from readout import read_led_spectra
from scipy import interpolate
import doplots as dp
import numpy as np
import computations as comp
import matplotlib.pyplot as plt
import mydoplots as mydp
import datetime as dt
import matplotlib.dates as md

from xkcd_colors import xkcd_colors
import declaration
import seaborn as sns

from thermal_coefficients import coefficient_result
from thermal_coefficients import coeff_error_result
from thermal_coefficients import names
from thermal_coefficients import measured_wavelengths

print(coefficient_result)
print(coeff_error_result)
print(names)
print(measured_wavelengths)

w1, coefficient = read_pindiode_temperature_coefficient()
w2, led_intensity_M470L3 = read_led_spectra(led_spectra_file='./Data/Specification/M470L3_led.txt')
w3, led_intensity_M405L2 = read_led_spectra(led_spectra_file='./Data/Specification/M405L2_led.txt')
coefficient_f = interpolate.interp1d(w1, coefficient)
led_intensity_M470L3_f = interpolate.interp1d(w2, led_intensity_M470L3)
led_intensity_M405L2_f = interpolate.interp1d(w3, led_intensity_M405L2)

x = measured_wavelengths
y = coefficient_result
y_error = coeff_error_result
decimals = 3

name_array = names
point_array = measured_wavelengths

name_array = ['value at 470 nm', 'value at 400 nm', 'value at 405 nm']
point_array = measured_wavelengths
ax = mydp.plot_thermal_coefficient(coefficient_f, x_limits=[400, 1100], point_array=point_array, name_array=name_array, axes=None)

names = ['PinDiode + LED [470 nm]', 'PinDiode + Flasher [400 mn]', 'PinDiode + LED [405 nm]']
ax.errorbar(x[0], y[0], y_error[0],
                    label=names[0]+': {} ± {} %/$^\circ$C'.format(np.around(y[0], decimals=decimals), np.around(y_error[0], decimals=decimals)), fmt='o', color=sns.xkcd_rgb['cherry red'],
                    ecolor=sns.xkcd_rgb['amber'], elinewidth=3, capsize=4, ms=3, fillstyle='full')

ax.errorbar(x[1], y[1], y_error[1],
                    label=names[1]+': {} ± {} %/$^\circ$C'.format(np.around(y[1], decimals=decimals), np.around(y_error[1], decimals=decimals)), fmt='o', color=sns.xkcd_rgb['emerald green'],
                    ecolor=sns.xkcd_rgb['amber'], elinewidth=3, capsize=4, ms=3, fillstyle='full')

ax.errorbar(x[2], y[2], y_error[2],
                    label=names[2]+': {} ± {} %/$^\circ$C'.format(np.around(y[2], decimals=decimals), np.around(y_error[2], decimals=decimals)), fmt='o', color=sns.xkcd_rgb['blue violet'],
                    ecolor=sns.xkcd_rgb['amber'], elinewidth=3, capsize=4, ms=3, fillstyle='full')

ax.legend()
figure_name = './Output/Specifications/{}.png'.format('Photosensitivity_temperature_zoomed')
plt.savefig(figure_name)
plt.show()


name_array = ['value at 470 nm', 'value at 400 nm', 'value at 405 nm']
point_array = measured_wavelengths
ax = mydp.plot_thermal_coefficient(coefficient_f, x_limits=[400, 500], point_array=point_array, name_array=name_array, axes=None)

names = ['PinDiode + LED [470 nm]', 'PinDiode + Flasher [400 mn]', 'PinDiode + LED [405 nm]']
ax.errorbar(x[0], y[0], y_error[0],
                    label=names[0]+': {} ± {} %/$^\circ$C'.format(np.around(y[0], decimals=decimals), np.around(y_error[0], decimals=decimals)), fmt='o', color=sns.xkcd_rgb['cherry red'],
                    ecolor=sns.xkcd_rgb['amber'], elinewidth=3, capsize=4, ms=3, fillstyle='full')

ax.errorbar(x[1], y[1], y_error[1],
                    label=names[1]+': {} ± {} %/$^\circ$C'.format(np.around(y[1], decimals=decimals), np.around(y_error[1], decimals=decimals)), fmt='o', color=sns.xkcd_rgb['emerald green'],
                    ecolor=sns.xkcd_rgb['amber'], elinewidth=3, capsize=4, ms=3, fillstyle='full')

ax.errorbar(x[2], y[2], y_error[2],
                    label=names[2]+': {} ± {} %/$^\circ$C'.format(np.around(y[2], decimals=decimals), np.around(y_error[2], decimals=decimals)), fmt='o', color=sns.xkcd_rgb['blue violet'],
                    ecolor=sns.xkcd_rgb['amber'], elinewidth=3, capsize=4, ms=3, fillstyle='full')

ax.legend()
figure_name = './Output/Specifications/{}.png'.format('Photosensitivity_temperature_zoomed')
plt.savefig(figure_name)
plt.show()
