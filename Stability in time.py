#! /anaconda3/envs/Flasher/bin/python

import sys
import csv
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker
from numpy.fft import fft, fftfreq, ifft


def read_file(path_to_file, scan_type, initial_temperature_time=None):
    """
    read file of data
    :param path_to_file:                (string) file location on drive
    :param scan_type:                   (string) scan_type has only 3 possible values: time, space, or temp
    :param initial_temperature_time:    (string) date from the first measurement of temperature from Climatic Chamber, taken by hand after acquisition.
                                                It should have the following format : %d/%m/%Y %H:%M:%S
    :return:    data array of several variables taken from acquisition like,
                for 'time' and 'space' scan:
                    x :                     (float) x-axis position in a space scan in millimetres
                    y :                     (float) y-axis position in a space scan in millimetres
                    current :               (float) average current of 10 consecutive measurement for a given point, in Amperes
                    current_std :           (float) standard deviation from the current measurements done at acquisition, Amperes
                    current_timestamp:      (datetime) timestamps of the format '%d/%m/%Y %H:%M:%S' of each measurement of current done
                for 'temp' scan:
                    temperature_time :      time in seconds for each measurement of temperature done by the climatic chamber for a given point in time
                    temperature_timestamp : timestamps of the format '%d/%m/%Y %H:%M:%S' of each measurement of temperature done
                                            empty list if parameter 'initial_temperature_time' is None
                    measured_temperature :  temperature measurement done by the climatic chamber for a given point in time
    """
    # path_to_file :    Relative path to the file

    if scan_type != 'time' and scan_type != 'space' and scan_type != 'temp':
        print('scan_type value is : {}'.format(scan_type))
        print('scan_type has only 3 possible values: time or space or temp')
        sys.exit()

    if scan_type != 'temp':

        x = []
        y = []
        current = []
        current_std = []
        current_time = []
        current_timestamp = []

        if scan_type == 'space':
            file_address = path_to_file
            with open(file_address) as file:
                data = csv.reader(file, delimiter=' ')
                for i, row in enumerate(data):
                    if i == 0:
                        continue
                    x.append(float(row[0]))
                    y.append(float(row[1]))
                    #current.append(float(row[2]) / 1e-9)        # nano Ampere
                    #current_std.append(float(row[3]) / 1e-9)    # nano Ampere
                    current.append(float(row[2]))        # Ampere
                    current_std.append(float(row[3]))    # Ampere
                    current_timestamp.append(dt.datetime.strptime(str(row[4] + ' ' + str(row[5])), '%d/%m/%Y %H:%M:%S'))
            x = np.array(x)
            y = np.array(y)
            current = np.array(current)
            current_std = np.array(current_std)
            current_timestamp = np.array(current_timestamp)

        else: # scan_type == 'time'
            file_address = path_to_file
            with open(file_address) as file:
                data = csv.reader(file, delimiter=' ')
                for i, row in enumerate(data):
                    if i == 0:
                        continue
                    x.append(float(float(0)))
                    y.append(float(float(0)))
                    #current.append(float(row[2]) / 1e-9)        # nano Ampere
                    #current_std.append(float(row[3]) / 1e-9)    # nano Ampere
                    current.append(float(row[2]))  # Ampere
                    current_std.append(float(row[3]))  # Ampere
                    current_timestamp.append(dt.datetime.strptime(str(row[0] + ' ' + str(row[1])), '%d/%m/%Y %H:%M:%S'))

            x = np.array(x)
            y = np.array(y)
            current = np.array(current)
            current_std = np.array(current_std)
            current_timestamp = np.array(current_timestamp)

        for i, date in enumerate(current_timestamp):
            current_time.append(current_timestamp[i] - current_timestamp[0])
        for i, delta_time in enumerate(current_time):
            current_time[i] = delta_time.total_seconds()
        current_time = np.array(current_time)

        return x, y, current, current_std, current_time, current_timestamp

    else: # scan_type == 'temp'

        temperature_time = []      # time in seconds from the starting point
        temperature_timestamp = []  # timestamps of temperature measurements
        measured_temperature = []   # measured temperature in Celsius given via the Climate Chamber
        demanded_temperature = []   # demanded temperature in Celsius by the user (programmed)
        measured_humidity = []      # measured humidity given via the Climate Chamber
        demanded_humidity = []      # demanded humidity by the user (programmed)

        file_address = path_to_file
        with open(file_address) as file:
            data = csv.reader(file, delimiter=';')
            for i, row in enumerate(data):
                if i == 0:
                    continue
                if i == 1:
                    continue
                temperature_time.append(float(row[0]))
                measured_temperature.append(float(row[1]))
                demanded_temperature.append(float(row[2]))
                measured_humidity.append(float(row[3]))
                demanded_humidity.append(float(row[4]))

        if initial_temperature_time is not None:
            initial_temperature_time = dt.datetime.strptime(initial_temperature_time, '%d/%m/%Y %H:%M:%S')
            for i, seconds in enumerate(temperature_time):
                temperature_timestamp.append(dt.timedelta(seconds=seconds))
            for i, seconds in enumerate(temperature_timestamp):
                temperature_timestamp[i] = initial_temperature_time + seconds

        temperature_time = np.array(temperature_time)
        temperature_timestamp = np.array(temperature_timestamp)
        measured_temperature = np.array(measured_temperature)
        # demanded_temperature = np.array(demanded_temperature)
        # measured_humidity = np.array(measured_humidity)
        # demanded_humidity = np.array(demanded_humidity)

        #return time, measured_temperature, demanded_temperature, measured_humidity, demanded_humidity
        return measured_temperature, temperature_time, temperature_timestamp


def read_pindiode_photo_sensitivity(cerenkov_file='./Data/Specification/PinDiode_Spectral_response.csv'):
    wavelength = []
    photo_sensitivity = []

    file_address = cerenkov_file

    with open(file_address) as file:
        data = csv.reader(file, delimiter=' ')
        for i, row in enumerate(data):
            if i == 0:
                continue
            wavelength.append(float(row[0]))
            photo_sensitivity.append(float(row[1]))

    wavelength = np.array(wavelength)
    photo_sensitivity = np.array(photo_sensitivity)

    return wavelength, photo_sensitivity


def calculate_systematic_error(mean):
    """
    calculates the statistical and systematic errors and their combination (total error) for each point of the array.
    Each point of the array is a mean of N points previously taken by the acquisition device.

    SYSTEMATIC ERROR : Keithley 6487 Picoammeter Specification
    Computing systematic error due to the measurement apparatus
    According to Manual : range: 2    nA    : 0.30 % of RDG + 400 fA
                          range: 20   nA    : 0.20 % of RDG + 1   pA
                          range: 200  nA    : 0.15 % of RDG + 10  pA
                          range: 2    µA    : 0.15 % of RDG + 100 pA
                          range: 20   µA    : 0.10 % of RDG + 1   nA
                          range: 200  µA    : 0.10 % of RDG + 10  nA
                          range: 2    mA    : 0.10 % of RDG + 100 nA
                          range: 20   mA    : 0.10 % of RDG + 1   µA
    :param mean:
    :param std:
    :return:
    """

    range_array = 1e-9 * np.array([2, 20, 200, 2e+3, 20e+3, 200e+3, 2e+3, 20e+6])
    factor = np.array([0.30, 0.20, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10]) / np.array(100)
    correction = 1e-9 * np.array([400e-6, 1e-3, 10e-3, 100e-3, 1, 10, 100, 1e+3])

    error_sys = []

    for i, value in enumerate(mean):
        if np.abs(value) < 0.:
            print('value out of range : value lower than expected')
        elif (np.abs(value) > 0.) & (np.abs(value) <= range_array[0]):
            index = 0
        elif (np.abs(value) > range_array[0]) & (np.abs(value) <= range_array[1]):
            index = 1
        elif (np.abs(value) > range_array[1]) & (np.abs(value) <= range_array[2]):
            index = 2
        elif (np.abs(value) > range_array[2]) & (np.abs(value) <= range_array[3]):
            index = 3
        elif (np.abs(value) > range_array[3]) & (np.abs(value) <= range_array[4]):
            index = 4
        elif (np.abs(value) > range_array[4]) & (np.abs(value) <= range_array[5]):
            index = 5
        elif (np.abs(value) > range_array[5]) & (np.abs(value) <= range_array[6]):
            index = 6
        elif (np.abs(value) > range_array[6]) & (np.abs(value) <= range_array[7]):
            index = 7
        else:
            print('value out of range : value higher than max')

        error_sys.append(factor[index] * mean[i] + correction[index])

    error_sys = np.array([error_sys])

    return error_sys


def calculate_statistic_error(std, N=10):
    """

    :param std:
    :param N:
    :return:
    """
    error_stat = std / np.sqrt(N)

    return error_stat


pdf = PdfPages("/Users/lonewolf/Desktop/stability_in_time.pdf")
pin_wavelength, pin_pde = read_pindiode_photo_sensitivity()

f = interpolate.interp1d(pin_wavelength, pin_pde)
pde_390nm = f(390)
pin_diode_surface = 0.028**2 # [m^2]

# values to zoom into the constant range of the current I
ini = 50
end = 350

fig, ax = plt.subplots()
ax.scatter(pin_wavelength, pin_pde, label='Pin-diode PDE')
ax.set_xlabel(r'$\lambda$ [nm]')
ax.set_ylabel('PDE [A/W]')
ax.legend()
pdf.savefig(fig)

file_flash = '/Users/lonewolf/PyCharmProjects/Flasher/Data/others/FlasherOn_16V_5kHz_5us.txt'
x, y, current, current_std, time, current_timestamp = read_file(path_to_file=file_flash, scan_type='time')

systematic_err = calculate_systematic_error(current)
statistic_err = calculate_statistic_error(current_std)
total_err = np.sqrt(systematic_err**2 + statistic_err**2)[0]

fig, ax = plt.subplots()
ax.errorbar(time/3600, current, total_err, label=r'$E_{flasher}$', color=sns.xkcd_rgb['sky blue'], linestyle='None', marker='o', ecolor=sns.xkcd_rgb['amber'], ms=3)
ax.set_xlabel('Time [hours]')
ax.set_ylabel(r'I [A]')
ax.legend()
pdf.savefig(fig)

fig, ax = plt.subplots()
ax.errorbar(time[ini:end]/3600, current[ini:end], total_err[ini:end], label=r'$E_{flasher}$', color=sns.xkcd_rgb['sky blue'], linestyle='None', marker='o', ecolor=sns.xkcd_rgb['amber'], ms=3)
ax.set_xlabel('Time [hours]')
ax.set_ylabel(r'I [A]')
ax.legend()
pdf.savefig(fig)

# Histogram of values in constant range
values = current[ini:end]
mean = np.mean(values)
std = np.std(values)

N_points = len(values)
n_bins = N_points/10
delta = (values.max()-values.min())/n_bins
bins = np.arange(values.min(), values.max() + delta/2, (values.max()-values.min())/n_bins)

fig, ax = plt.subplots()
# We can set the number of bins with the `bins` kwarg
ax.hist(values, bins=bins)

text = 'mean = {}\n' \
       'std = {}'.format(mean, std)
anchored_text = AnchoredText(text, loc=9)
ax.add_artist(anchored_text)
pdf.savefig(fig)

int_time = np.array(time[ini:end])
int_current = np.array(current[ini:end])
int_error = np.array(total_err[ini:end])

val_i = 30
val_j = 10
int_time = int_time.reshape((val_i, val_j))
int_current = int_current.reshape((val_i, val_j))
int_error = int_error.reshape((val_i, val_j))

for i, row in enumerate(int_time):
    point_time = (row[0] + row[-1])/2
    int_time[i] = point_time
for i, row in enumerate(int_error):
    int_error[i] = np.sqrt(np.sum(row**2))
int_current = np.mean(int_current, axis=1)

int_time = int_time[:, 0]
int_error = int_error[:, 0]

fig, ax = plt.subplots()
ax.errorbar(int_time/3600, int_current, int_error, label=r'$I_{flasher}$, mean among 10 points', color=sns.xkcd_rgb['sky blue'], linestyle='None', marker='o', ecolor=sns.xkcd_rgb['amber'], ms=3)
ax.set_xlabel('Time [hours]')
ax.set_ylabel(r'I [A]')
ax.legend()
pdf.savefig(fig)

int_time = np.array(time[ini:end])
int_current = np.array(current[ini:end])
int_error = np.array(total_err[ini:end])

val_i = 20
val_j = 15
int_time = int_time.reshape((val_i, val_j))
int_current = int_current.reshape((val_i, val_j))
int_error = int_error.reshape((val_i, val_j))

for i, row in enumerate(int_time):
    point_time = (row[0] + row[-1])/2
    int_time[i] = point_time
for i, row in enumerate(int_error):
    int_error[i] = np.sqrt(np.sum(row**2))
int_current = np.mean(int_current, axis=1)

int_time = int_time[:, 0]
int_error = int_error[:, 0]

fig, ax = plt.subplots()
ax.errorbar(int_time/3600, int_current, int_error, label=r'$I_{flasher}$, mean among 20 points', color=sns.xkcd_rgb['sky blue'], linestyle='None', marker='o', ecolor=sns.xkcd_rgb['amber'], ms=3)
ax.set_xlabel('Time [hours]')
ax.set_ylabel(r'I [A]')
ax.legend()
pdf.savefig(fig)


# I [A] = PDE [A/W] * E [W/m^2] * S [m^2]
# I = current in Amperes
# PDE = photo detection efficiency of the pin-diode in Ampere/Watts
# E =  Irradiance : Watts / unit of surface
# S =  surface of the photo-detector in surface unit
# E = I / (PED * S) [W/m^2]

conversion_factor = 1 / (pde_390nm * pin_diode_surface)
irradiance = current * conversion_factor
irradiance_err = total_err * conversion_factor

fig, ax = plt.subplots()
ax.errorbar(time/3600, irradiance, irradiance_err, label=r'$E_{flasher}$', color=sns.xkcd_rgb['sky blue'], linestyle='None', marker='o', ecolor=sns.xkcd_rgb['amber'], ms=3)
y_labels = ax.get_yticks()
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
ax.set_xlabel('Time [hours]')
ax.set_ylabel(r'E [$W/m^2$]')
ax.legend()
pdf.savefig(fig)

fig, ax = plt.subplots()
ax.errorbar(time[ini: end]/3600, irradiance[ini: end], irradiance_err[ini: end], label=r'$E_{flasher}$', color=sns.xkcd_rgb['sky blue'], linestyle='None', marker='o', ecolor=sns.xkcd_rgb['amber'], ms=3)
y_labels = ax.get_yticks()
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
ax.set_xlabel('Time [hours]')
ax.set_ylabel(r'E [$W/m^2$]')
ax.legend()
pdf.savefig(fig)

# Histogram of values in constant range
values = irradiance[ini:end]
mean = np.mean(values)
std = np.std(values)

N_points = len(values)
n_bins = N_points/10
delta = (values.max()-values.min())/n_bins
bins = np.arange(values.min(), values.max() + delta/2, (values.max()-values.min())/n_bins)
fig, ax = plt.subplots()
# We can set the number of bins with the `bins` kwarg
ax.hist(values, bins=bins)

text = 'mean = {}\n' \
       'std = {}'.format(mean, std)
anchored_text = AnchoredText(text, loc=9)
ax.add_artist(anchored_text)
pdf.savefig(fig)

int_time = np.array(time[ini:end])
int_irradiance = np.array(irradiance[ini:end])
int_irradiance_error = np.array(irradiance_err[ini:end])

val_i = 30
val_j = 10

int_time = int_time.reshape((val_i, val_j))
int_irradiance = int_irradiance.reshape((val_i, val_j))
int_irradiance_error = int_irradiance_error.reshape((val_i, val_j))

for i, row in enumerate(int_time):
    point_time = (row[0] + row[-1])/2
    int_time[i] = point_time
for i, row in enumerate(int_irradiance_error):
    int_irradiance_error[i] = np.sqrt(np.sum(row**2))
int_irradiance = np.mean(int_irradiance, axis=1)

int_time = int_time[:, 0]
int_irradiance_error = int_irradiance_error[:, 0]

fig, ax = plt.subplots()
ax.errorbar(int_time/3600, int_irradiance/1e-6, int_irradiance_error/1e-6, label=r'$E_{flasher}$, mean among 10 points', color=sns.xkcd_rgb['sky blue'], linestyle='None', marker='o', ecolor=sns.xkcd_rgb['amber'], ms=3)
ax.set_xlabel('Time [hours]')
ax.set_ylabel(r'E [$\mu W/m^2$]')
ax.legend()
pdf.savefig(fig)

int_time = np.array(time[ini:end])
int_irradiance = np.array(irradiance[ini:end])
int_irradiance_error = np.array(irradiance_err[ini:end])

val_i = 20
val_j = 15

int_time = int_time.reshape((val_i, val_j))
int_irradiance = int_irradiance.reshape((val_i, val_j))
int_irradiance_error = int_irradiance_error.reshape((val_i, val_j))

for i, row in enumerate(int_time):
    point_time = (row[0] + row[-1])/2
    int_time[i] = point_time
for i, row in enumerate(int_irradiance_error):
    int_irradiance_error[i] = np.sqrt(np.sum(row**2))
int_irradiance = np.mean(int_irradiance, axis=1)

int_time = int_time[:, 0]
int_irradiance_error = int_irradiance_error[:, 0]

fig, ax = plt.subplots()
ax.errorbar(int_time/3600, int_irradiance/1e-6, int_irradiance_error/1e-6, label=r'$E_{flasher}$, mean among 20 points', color=sns.xkcd_rgb['sky blue'], linestyle='None', marker='o', ecolor=sns.xkcd_rgb['amber'], ms=3)
ax.set_xlabel('Time [hours]')
ax.set_ylabel(r'E [$\mu W/m^2$]')
ax.legend()
pdf.savefig(fig)

# FOURIER TRANSFORM
int_time = np.array(time[ini:end])
int_current = np.array(current[ini:end])
int_error = np.array(total_err[ini:end])

num_points = len(int_current)

# find all the frequencies
freqs = fftfreq(num_points)
# ignoring 1/2 values, they are complex conjugates
mask = freqs > 0
# FFT and power spectra calculations
# fft values
fft_values = fft(int_current)

# Taking into account what was not taking into account
fft_theo = 2.0 * np.abs(fft_values/num_points)

fig, ax = plt.subplots(2, 1)
ax[0].plot(int_time/3600, int_current, label='Original signal')
ax[0].set_xlabel('Time [hours]')
ax[0].set_ylabel('I [A]')
ax[0].legend()
ax[1].plot(freqs, fft_values, label='FFT values')
ax[1].set_xlim(-1/5, 1/5)
ax[1].legend()
fig.tight_layout()
pdf.savefig(fig)

fig, ax = plt.subplots(2, 1)
ax[0].plot(int_time/3600, int_current, label='Original signal')
ax[0].set_xlabel('Time [hours]')
ax[0].set_ylabel('I [A]')
ax[0].legend()
ax[1].plot(freqs[mask], fft_theo[mask], label='FFT values')
ax[1].legend()
fig.tight_layout()
pdf.savefig(fig)

pdf.close()