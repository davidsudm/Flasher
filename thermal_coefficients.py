#! /anaconda3/envs/Flasher/bin/python

from readout import read_file
from readout import read_file_of_diodes

import numpy as np
import computations as comp
import matplotlib.pyplot as plt
import mydoplots as mydp



def make_thermal_coefficients(temperature_initial_date, temp_file, int_file, name, initial_cut=40, flasher=False, decimals=3):
    """

    :param temperature_initial_date:    initial timestamp for temperature, taken manually from ClimPilot
    :param temp_file:                   path to the file for the temperature
    :param int_file:                    path to the file for the current or intensity
    :param name:                        name of the event
    :return:
    """
    temperature_array = [25., 20., 15., 10., 5.]

    temperature, temperature_time, temperature_timestamp = read_file(temp_file, 'temp', temperature_initial_date)
    x, y, current, current_std, current_time, current_timestamp = read_file(int_file, 'time')
    statistic, systematic, total_error = comp.calculate_errors(current, current_std)

    new_time, new_current, new_current_errors, new_temperature = comp.give_matched_arrays(
        temperature_time=temperature_time, temperature=temperature, current_time=current_time, current=current,
        current_std=total_error, initial_cut=initial_cut)
    if flasher:
        mask, long_mask = comp.make_mask(temperature=new_temperature, cut=70)
    else:
        mask, long_mask = comp.make_mask(temperature=new_temperature)

    means, mean_errors, mean_error_prop = comp.compute_masked_values(mask_array=mask, current=new_current,
                                                                     current_error=new_current_errors)

    ax = mydp.plot_masked_intensity_and_temperature(mask_array=mask, time=new_time, current=new_current,
                                                    current_error=new_current_errors, temperature=new_temperature,
                                                    means=means, error_propagation=mean_error_prop, decimals=decimals, axes=None)
    figure_name = './Output/Thermal_Coefficients/{}_{}.png'.format(name, 'masked_intensity_and_temperature')
    plt.savefig(figure_name)

    ax1, ax2 = mydp.plot_full_intensity_and_temperature_vs_time(temperature_time=temperature_time,
                                                                temperature=temperature, current_time=current_time,
                                                                current=current, current_std=current_std,
                                                                initial_cut=50, axes=None)
    figure_name = './Output/Thermal_Coefficients/{}_{}.png'.format(name, 'full_intensity_and_temperature')
    plt.savefig(figure_name)

    tdf, rdf, rdf_error = comp.compute_relative_difference(current=means, current_error=mean_error_prop, temperature_array=temperature_array)
    slope, slope_error, intersect, intersect_error = comp.interpolate_thermal_coefficient(tdf, rdf)
    mean_slope, error_prop_slope, mean_intersect, error_prop_intersect = comp.get_thermal_coefficient(current_at_temperature=means,
                                                                                                      current_error_at_temperature=mean_error_prop,
                                                                                                      temperature_array=temperature_array)

    axes_array = mydp.plot_slope_for_thermal_coefficient(tdf=tdf, rdf=rdf, rdf_error=rdf_error, slopes=slope,
                                                         slopes_error=slope_error, intersects=intersect,
                                                         intersects_error=intersect_error, name=name, temperature_array=temperature_array, axes=None)

    coefficient = mean_slope * 100
    error = error_prop_slope * 100

    print(name)
    print('Thermal Coefficient: {} ± {}'.format(coefficient, error))

    return coefficient, error

names = []
coefficient_result = []
coeff_error_result = []
measured_wavelengths = []


# PHOTODIODE and LED M470L3
temperature_initial_date = '16/05/2019 15:28:02'
temp_file = '2019_05_16/Clim_Photodiode_Scan00_30s.csv'
int_file = '2019_05_16/Photodiode_Scan00.txt'
name = 'LED M470L3'
nominal_wavelength = 470.0
coef, err = make_thermal_coefficients(temperature_initial_date, temp_file, int_file, name, initial_cut=40)

coefficient_result.append(coef)
coeff_error_result.append(err)
names.append(name)
measured_wavelengths.append(nominal_wavelength)

# FLASHER
temperature_initial_date = '02/05/2019 19:38:11'
temp_file = '2019_05_03/Clim_Photodiode_Scan00_30s.csv'
int_file = '2019_05_03/Flasher_Scan00.txt'
name = 'Flasher'
nominal_wavelength = 400.0
coef, err = make_thermal_coefficients(temperature_initial_date, temp_file, int_file, name, initial_cut=40, flasher=True)

coefficient_result.append(coef)
coeff_error_result.append(err)
names.append(name)
measured_wavelengths.append(nominal_wavelength)


# For the new "improved" setup

temperature_initial_date = '03/07/2019 22:46:10'
temp_file = '2019_07_03/Clim_Pilot_thermal_405nm_30s.csv'
int_file = '2019_07_03/thermal_405nm.txt'
name = 'LED M405L3'
nominal_wavelength = 405.0

time, timestamp, internal_current, internal_current_std, external_current, external_current_std, thermoresitor_temperature = read_file_of_diodes(int_file)
chamber_temperature, chamber_temperature_time, chamber_temperature_timestamp = read_file(temp_file, 'temp', temperature_initial_date)

internal_statistic, internal_systematic, internal_total_error = comp.calculate_errors(internal_current, internal_current_std)
external_statistic, external_systematic, external_total_error = comp.calculate_errors(external_current, external_current_std)

ax1, ax2 = mydp.plot_full_intensity_and_temperature_vs_time(temperature_time=time,
                                                temperature=thermoresitor_temperature, current_time=time,
                                                current=internal_current, current_std=internal_current_std,
                                                initial_cut=0, axes=None)
ax2.set_ylabel("Internal Current [nA]")

figure_name = './Output/Thermal_Coefficients/{}_{}_internal.png'.format(name, 'full_intensity_and_temperature')
plt.savefig(figure_name)
#plt.show()

ax1, ax2 = mydp.plot_full_intensity_and_temperature_vs_time(temperature_time=time,
                                                temperature=thermoresitor_temperature, current_time=time,
                                                current=external_current/1000, current_std=external_current_std/1000,
                                                initial_cut=0, axes=None)
ax2.set_ylabel("External Current [$\mu$A]")

figure_name = './Output/Thermal_Coefficients/{}_{}_external.png'.format(name, 'full_intensity_and_temperature')
plt.savefig(figure_name)
#plt.show()

decimals = 2
pre_cut = 30
temperature_array = [45.02, 35.05, 25.08, 15.20, 5.285]

ratio = external_current / internal_current
ratio_error = np.abs(ratio) * np.sqrt((external_total_error/external_current)**2 + (internal_total_error/internal_current)**2)

ax1, ax2 = mydp.plot_full_intensity_and_temperature_vs_time(temperature_time=time,
                                                temperature=thermoresitor_temperature, current_time=time,
                                                current=ratio, current_std=ratio_error,
                                                initial_cut=pre_cut, axes=None)
ax2.set_ylabel("Normalized Current")

figure_name = './Output/Thermal_Coefficients/{}_{}_ratio.png'.format(name, 'full_intensity_and_temperature')
plt.savefig(figure_name)
plt.title('Normalized current [ ]')
#plt.show()

mask, long_mask = comp.make_mask(temperature=thermoresitor_temperature, temperature_array=temperature_array)
means, mean_errors, mean_error_prop = comp.compute_masked_values(mask_array=mask, current=ratio, current_error=ratio_error)

ax1, ax2 = mydp.plot_masked_intensity_and_temperature(mask_array=mask, time=time, current=ratio,
                                                    current_error=ratio_error, temperature=thermoresitor_temperature,
                                                    means=means, error_propagation=mean_error_prop, decimals=decimals, axes=None)
ax2.set_ylabel("Normalized current [ ]")

figure_name = './Output/Thermal_Coefficients/{}_{}_ratios_masked.png'.format(name, 'masked_intensity_and_temperature')
plt.savefig(figure_name)
plt.title('Normalized current')
#plt.show()

tdf, rdf, rdf_error = comp.compute_relative_difference(current=means,
                                                       current_error=mean_error_prop,
                                                       temperature_array=temperature_array)

slope, slope_error, intersect, intersect_error = comp.interpolate_thermal_coefficient(delta_temperature=tdf,
                                                                                      relative_differences=rdf)

mean_slope, error_prop_slope, mean_intersect, error_prop_intersect = comp.get_thermal_coefficient(current_at_temperature=means,
                                                                                                  current_error_at_temperature=mean_error_prop,
                                                                                                  temperature_array=temperature_array)

axes_array = mydp.plot_slope_for_thermal_coefficient(tdf=tdf,
                                                     rdf=rdf,
                                                     rdf_error=rdf_error,
                                                     slopes=slope,
                                                     slopes_error=slope_error,
                                                     intersects=intersect,
                                                     intersects_error=intersect_error,
                                                     name=name,
                                                     temperature_array=temperature_array,
                                                     axes=None)
#plt.show()

coefficient = mean_slope * 100
error = error_prop_slope * 100
print(name)
print('Thermal Coefficient: {} ± {}'.format(coefficient, error))

coefficient_result.append(coefficient)
coeff_error_result.append(error)
names.append(name)
measured_wavelengths.append(nominal_wavelength)


print(coefficient_result)
print(coeff_error_result)
print(names)
print(measured_wavelengths)