#! /anaconda3/envs/Flasher/bin/python

from readout import read_file
from readout import read_pindiode_temperature_coefficient
from scipy import interpolate
import seaborn as sns

import numpy as np
import computations as comp
import matplotlib.pyplot as plt
import mydoplots as mydp

decimals = 3
output_dir = '/Users/lonewolf/PyCharmProjects/Flasher/Output/temperature_characteristics/'

def yield_thermal_coefficients(temperature_initial_date, temperature_file, current_file, source_name,
                               temperature_array=[25., 20., 15., 10., 5.], initial_cut=40, flasher=False, decimals=3,
                               scaling=1e-6, output_dir=output_dir):
    """

    :param temperature_initial_date:
    :param temperature_file:
    :param current_file:
    :param source_name:
    :param temperature_array:
    :param initial_cut:
    :param flasher:
    :param decimals:
    :param scaling:
    :return:
    """

    temperature, temperature_time, temperature_timestamp = read_file(temperature_file, 'temp', temperature_initial_date)
    x, y, current, current_std, current_time, current_timestamp = read_file(current_file, 'time')
    current_errors = comp.calculate_errors_in_thermal_range(current, current_std)

    irradiance, irradiance_errors = comp.transform_from_current_to_irradiance(current, current_errors)

    irradiance = irradiance / scaling
    irradiance_errors = irradiance_errors / scaling

    del x, y, current, current_std, current_errors

    time, irradiance, irradiance_errors, temperature = comp.give_matched_arrays(temperature_time=temperature_time,
                                                                                temperature=temperature,
                                                                                current_time=current_time,
                                                                                current=irradiance,
                                                                                current_std=irradiance_errors,
                                                                                initial_cut=initial_cut)

    if flasher:
        mask, long_mask = comp.make_mask(temperature=temperature, cut=50)
    else:
        mask, long_mask = comp.make_mask(temperature=temperature)

    mean, error_on_mean, error_propagation = comp.compute_masked_values(mask_array=mask,
                                                                        data=irradiance,
                                                                        data_error=irradiance_errors)

    ax1 = mydp.plot_masked_intensity_and_temperature(mask_array=mask,
                                                     time=time,
                                                     data=irradiance,
                                                     data_error=irradiance_errors,
                                                     temperature=temperature,
                                                     means=mean,
                                                     error_propagation=error_propagation,
                                                     decimals=decimals,
                                                     y_units=r'$\mu W/m^2$',
                                                     y_label=r'E [$\mu W/m^2$]',
                                                     axes=None)
    figure_name = output_dir + '{}_{}.png'.format(source_name, 'irradiance_temperature_vs_time_masked')
    plt.savefig(figure_name)

    end_cut = -55
    ax2 = mydp.plot_full_intensity_and_temperature_vs_time(temperature_time=time[:end_cut],
                                                           temperature=temperature[:end_cut],
                                                           current_time=time[:end_cut],
                                                           current=irradiance[:end_cut],
                                                           current_std=irradiance_errors[:end_cut],
                                                           initial_cut=50,
                                                           axes=None)
    figure_name = output_dir + '{}_{}.png'.format(source_name, 'irradiance_temperature_vs_time_unmasked')
    plt.savefig(figure_name)

    tdf, rdf, rdf_error = comp.compute_relative_difference(data=mean,
                                                           data_error=error_propagation,
                                                           temperature_array=temperature_array)

    # y = m*x + b in array because we compute for different graphs
    m_array, m_err_array, b_array, b_err_array = comp.interpolate_thermal_coefficient(delta_temperature=tdf,
                                                                                      relative_differences=rdf)
    # Here we get the average of each array
    slope, slope_err, intersect, intersect_err = comp.get_thermal_coefficient(data_at_temperature=mean,
                                                                              data_error_at_temperature=error_propagation,
                                                                              temperature_array=temperature_array)

    for i, line in enumerate(tdf):
        x_array = np.linspace(np.min(tdf[i]), np.max(tdf[i]), 1000)
        y_array = m_array[i] * x_array + b_array[i]

        fig, ax = plt.subplots()

        ax.errorbar(x=tdf[i],
                    y=rdf[i],
                    yerr=rdf_error[i],
                    label='Relative differences\n'
                          'with respect to {}$^\circ$C'.format(temperature_array[i]),
                    fmt='o',
                    color='black',
                    ecolor=sns.xkcd_rgb['amber'],
                    elinewidth=3,
                    capsize=4,
                    ms=2,
                    fillstyle='full')

        ax.plot(x_array,
                y_array,
                color=sns.xkcd_rgb['carolina blue'],
                label='Linear fit : m $\Delta$T + b \n'
                      'm = {} ± {} %/$^\circ$C \n'
                      'b = {} ± {} %'.format(np.around(m_array[i], decimals=decimals),
                                             np.around(m_err_array[i], decimals=decimals),
                                             np.around(b_array[i], decimals=decimals),
                                             np.around(b_err_array[i], decimals=decimals)))
        y_label = r'$1 - \frac{E_T}{E_{(T=%2.0f ^\circ C)}}$' % (temperature_array[i]) + ' [%]'
        ax.set_ylabel(y_label)
        ax.set_xlabel(r'$\Delta$T [$^\circ$C]')
        ax.legend(frameon=False)
        figure_name = output_dir + '{}_slope_wrt_to_{}.png'.format(source_name, temperature_array[i])
        plt.savefig(figure_name)

    thermal_coefficient = slope
    thermal_coefficient_error = slope_err

    print(source_name)
    print('Thermal Coefficient: {} ± {} %/C'.format(thermal_coefficient, thermal_coefficient_error))

    return thermal_coefficient, thermal_coefficient_error



# LED M470L3
initial_date = '16/05/2019 15:28:02'
temperature_file = '2019_05_16/Clim_Photodiode_Scan00_30s.csv'
current_file = '2019_05_16/Photodiode_Scan00.txt'
source_name = 'LED M470L3'
led_wavelength = 470.0

led_ther_coeff, led_ther_ceff_err = yield_thermal_coefficients(temperature_initial_date=initial_date,
                                                               temperature_file=temperature_file,
                                                               current_file=current_file,
                                                               source_name=source_name,
                                                               temperature_array=[25., 20., 15., 10., 5.],
                                                               initial_cut=40,
                                                               flasher=False,
                                                               decimals=3,
                                                               scaling=1e-6)

# FLASHER INSIDE
initial_date = '02/05/2019 19:38:11'
temperature_file = '2019_05_03/Clim_Photodiode_Scan00_30s.csv'
current_file = '2019_05_03/Flasher_Scan00.txt'
source_name = 'Flasher Inside'
flasher_wavelength = 390.0

flash_in_ther_coeff, flash_in_ther_ceff_err = yield_thermal_coefficients(temperature_initial_date=initial_date,
                                                                         temperature_file=temperature_file,
                                                                         current_file=current_file,
                                                                         source_name=source_name,
                                                                         temperature_array=[25., 20., 15., 10., 5.],
                                                                         initial_cut=40,
                                                                         flasher=True,
                                                                         decimals=3,
                                                                         scaling=1e-6)

# FLASHER OUTSIDE
initial_date = '16/05/2019 15:28:02'
temperature_file = '2019_05_16/Clim_Photodiode_Scan00_30s.csv'
current_file = '2019_05_16/Photodiode_Scan00.txt'
source_name = 'Flasher Outside'
flasher_wavelength = 390.0

'''
initial_date = '13/06/2019 19:11:58'
temperature_file = '2019_06_13/Clim_Photodiode_Scan_01_30s.csv'
current_file = '2019_06_13/Photodiode_Scan01.txt'
source_name = 'Flasher Outside'
flasher_wavelength = 390.0

initial_date = '16/05/2019 12:19:00'
temperature_file = '2019_05_15/Clim_Photodiode_Scan00_30s.csv'
current_file = '2019_05_15/Photodiode_Scan00.txt'
source_name = 'Flasher Outside'
flasher_wavelength = 390.0
'''

flash_out_ther_coeff, flash_out_ther_ceff_err = yield_thermal_coefficients(temperature_initial_date=initial_date,
                                                                           temperature_file=temperature_file,
                                                                           current_file=current_file,
                                                                           source_name=source_name,
                                                                           temperature_array=[25., 20., 15., 10., 5.],
                                                                           initial_cut=40,
                                                                           flasher=True,
                                                                           decimals=3,
                                                                           scaling=1e-6)

wavelength, coefficient = read_pindiode_temperature_coefficient()
coefficient_function = interpolate.interp1d(wavelength, coefficient)

wavelength = np.linspace(400, 550, 700)
coefficient = coefficient_function(wavelength)


fig, ax = plt.subplots()

ax.plot(wavelength, coefficient, label='Typical photodiode', linestyle=':', color=sns.xkcd_rgb['azure'])
ax.errorbar(x=led_wavelength,
            y=led_ther_coeff,
            yerr=led_ther_ceff_err,
            label='LED @ 470 nm :\n'
                  r'K = {} ± {} %$/^\circ$C'.format(np.around(led_ther_coeff, decimals=decimals),
                                                    np.around(led_ther_ceff_err, decimals=decimals)),
            fmt='o',
            color=sns.xkcd_rgb['cherry red'],
            ecolor=sns.xkcd_rgb['amber'],
            elinewidth=3,
            capsize=4,
            ms=3,
            fillstyle='full')

ax.errorbar(x=flasher_wavelength,
            y=flash_in_ther_coeff,
            yerr=flash_in_ther_ceff_err,
            label='Flasher IN @ 390 nm :\n'
                  r'K = {} ± {} %$/^\circ$C'.format(np.around(flash_in_ther_coeff, decimals=decimals),
                                                    np.around(flash_in_ther_ceff_err, decimals=decimals)),
            fmt='o',
            color=sns.xkcd_rgb['emerald green'],
            ecolor=sns.xkcd_rgb['amber'],
            elinewidth=3,
            capsize=4,
            ms=3,
            fillstyle='full')

ax.errorbar(x=flasher_wavelength,
            y=flash_out_ther_coeff,
            yerr=flash_out_ther_ceff_err,
            label='Flasher OUT @ 390 nm :\n'
                  r'K = {} ± {} %$/^\circ$C'.format(np.around(flash_out_ther_coeff, decimals=decimals),
                                                    np.around(flash_out_ther_ceff_err, decimals=decimals)),
            fmt='o',
            color=sns.xkcd_rgb['blue violet'],
            ecolor=sns.xkcd_rgb['amber'],
            elinewidth=3,
            capsize=4,
            ms=3,
            fillstyle='full')

ax.set_ylabel(r'$K_{thermal}$ [%/$^\circ$C]')
ax.set_xlabel('Wavelength [mn]')
#ax.set_ylim(bottom=-0.5, top=0.1)
ax.set_xlim(left=350, right=550)
ax.legend(frameon=False, loc=1)

figure_name = output_dir + 'temperature_characteristics.png'
plt.savefig(figure_name)
