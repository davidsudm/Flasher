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
import declaration
import seaborn as sns

from matplotlib import gridspec

# Surface scan files:

# '2019_04_24/Scan_Flasher01.txt' excluded since it is dark current
file_list1 = ['2019_04_24/Scan_Flasher02.txt',
              '2019_04_24/Scan_Flasher03.txt',
              '2019_04_24/Scan_Flasher04.txt',
              '2019_04_24/Scan_Flasher05.txt',
              '2019_04_24/Scan_Flasher06.txt']

# '2019_04_25/Scan_Flasher00.txt' excluded since it is dark current
file_list2 = ['2019_04_25/Scan_Flasher10.txt',
              '2019_04_25/Scan_Flasher02.txt',
              '2019_04_25/Scan_Flasher03.txt',
              '2019_04_25/Scan_Flasher04.txt',
              '2019_04_25/Scan_Flasher05.txt',
              '2019_04_25/Scan_Flasher06.txt',
              '2019_04_25/Scan_Flasher07.txt',
              '2019_04_25/Scan_Flasher08.txt',
              '2019_04_25/Scan_Flasher09.txt',
              '2019_04_25/Scan_Flasher10.txt',
              '2019_04_25/Scan_Flasher11.txt',
              '2019_04_25/Scan_Flasher12.txt']

file_list3 = ['2019_04_27/Scan_Flasher00.txt',
              '2019_04_27/Scan_Flasher01.txt',
              '2019_04_27/Scan_Flasher02.txt',
              '2019_04_27/Scan_Flasher03.txt',
              '2019_04_27/Scan_Flasher04.txt',
              '2019_04_27/Scan_Flasher05.txt',
              '2019_04_27/Scan_Flasher06.txt',
              '2019_04_27/Scan_Flasher07.txt',
              '2019_04_27/Scan_Flasher08.txt',
              '2019_04_27/Scan_Flasher09.txt',
              '2019_04_27/Scan_Flasher10.txt']

# OPTIONS :
xo = yo = 0.0 # m
xf = yf = 0.3 # m
steps = 11
number_of_cells = steps * steps
photosensitivity = 0.2 # A/W
illuminated_area = 0.03*0.03 # m^2
distance = 1.0 # m
scaling = 1e-6 # to micro


import fitfunction as fitf


# INTERPOLATION OF PROJECTIONS OF AXIS : X AND Y
space_Data = [file_list2, file_list3]
space_Data = [file_list3]
for file_list in space_Data:
    for file in file_list:

        """ folder and plot_name came from one component of a file list which is a string type, divided into 2 strings """
        folder = file.split("/")[0]
        filename = file.split("/")[1]
        plot_name = filename.split('.')[0]

        amplitude_vector = []
        frequency_vector = []
        offset_vector = []
        amplitude_error_vector = []
        frequency_error_vector = []
        offset_error_vector = []
        distance_error_vector = []



        x, y, current, current_std, current_time, current_timestamp = read_file(path_to_file=file,
                                                                                scan_type='space')

        statistic, systematic, total = comp.calculate_errors(mean=current,
                                                             std=current_std)

        # Transform current values to irradiance values in Watts / m^2
        irradiance = current/(photosensitivity * illuminated_area)
        irradiance_error = total/(photosensitivity * illuminated_area)
        irradiance_error = total

        x_center, y_center = 0.15, 0.15
        x_label = x_position = np.linspace(xo, xf, steps)
        x_position = x_position - x_center
        y_label = y_position = np.linspace(yo, yf, steps)
        y_position = y_position - y_center


        data_irradiance, data_irradiance_error = comp.create_data_grid(data=irradiance,
                                                                       data_errors=irradiance_error,
                                                                       steps=steps,
                                                                       rel_label=False)

        data_current, data_current_error = comp.create_data_grid(data=current,
                                                                 data_errors=total,
                                                                 steps=steps,
                                                                 rel_label=False)

        ax1, ax_cl = mydp.plot_intensity_scan_xy_2D(data=data_irradiance,
                                                    x_limits=[-xf / 2, xf / 2],
                                                    y_limits=[-yf / 2, yf / 2],
                                                    data_label=True,
                                                    axes=None,
                                                    scaling=scaling)
        # The scaling factor defines the final unites for the color bar IN THE PLOT ONLY
        ax1.set_xlabel('x position [m]')
        ax1.set_ylabel('y position [m]')
        ax_cl.set_label('E [$\mu W/m^2$]')

        figure_name = './Output/Irradiance_interpolation/{}/{}_irradiance.png'.format(folder, plot_name)
        plt.savefig(figure_name)


        ax1, ax_cl = mydp.plot_intensity_scan_xy_2D(data=data_current,
                                                    x_limits=[-xf/2, xf/2],
                                                    y_limits=[-yf/2, yf/2],
                                                    data_label=True,
                                                    axes=None,
                                                    scaling=1e-9)
        # The scaling factor defines the final unites for the color bar IN THE PLOT ONLY
        ax1.set_xlabel('x position [m]')
        ax1.set_ylabel('y position [m]')
        ax_cl.set_label('Current [$nA$]')
        ax1.set_aspect(aspect=1)

        figure_name = './Output/Irradiance_interpolation/{}/{}_current.png'.format(folder, plot_name)
        plt.savefig(figure_name)


        x_center, y_center = 0.15, 0.15
        x_label = x_position = np.linspace(xo, xf, steps)
        x_position = x_position - x_center
        y_label = y_position = np.linspace(yo, yf, steps)
        y_position = y_position - y_center

        # we use steps, but for fitting we could use more points
        x_fit = np.linspace(np.min(x_position), np.max(x_position), steps)
        y_fit = np.linspace(np.min(y_position), np.max(y_position), steps)

        path = './Output/Irradiance_interpolation/{}/{}_parameters.txt'.format(folder, filename)
        output = open(path, "w")
        output.write("run_name\t x_pos [m]\t y_pos [m]\t Intensity [W/sr]\t frequency\t offset [mm]\t")
        #output.write("run_name\t x_pos [m]\t y_pos [m]\t Intensity [W/sr]\t frequency\t offset [mm]\t")

        for i in range(0, steps):

            # for a fixed y position
            irradiance_in_x = data[i, :]
            irradiance_in_x_error = data_error[i, :]
            print('y :', y_position[i])
            print(irradiance_in_x)

            """"""
            # fitting irradiance
            guessed_irradiance = np.max(irradiance_in_x)
            guessed_frequency = np.pi
            guessed_offset = 0.03
            initial_params = [guessed_irradiance, guessed_frequency, guessed_offset]
            pfit_leastsq, perr_leastsq = fitf.fit_irradiance(distance=distance,
                                                             initial_params=initial_params,
                                                             x=x_position,
                                                             y=irradiance_in_x)
            amplitude = pfit_leastsq[0]
            frequency = pfit_leastsq[1]
            offset = pfit_leastsq[2]

            amplitude_err = perr_leastsq[0]
            frequency_err = perr_leastsq[1]
            offset_err = perr_leastsq[2]

            print(amplitude, frequency, offset)
            amplitude_vector.append(amplitude)
            frequency_vector.append(frequency)
            offset_vector.append(offset)

            amplitude_error_vector.append(amplitude_err)
            frequency_error_vector.append(frequency_err)
            offset_error_vector.append(offset_err)


            irradiance_fit = fitf.irradiance(x=x_fit,
                                             distance=distance,
                                             amplitude=amplitude,
                                             frequency=frequency,
                                             offset=offset)



            fig = plt.figure()
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

            ax1 = plt.subplot(gs[0])

            ax1.plot(x_position, irradiance_in_x/scaling, 'r-', label='data', linewidth=1.0)
            ax1.fill_between(x=x_position,
                             y1=(irradiance_in_x - irradiance_in_x_error)/scaling,
                             y2=(irradiance_in_x + irradiance_in_x_error)/scaling,
                             color=sns.xkcd_rgb["amber"], alpha=0.7)
            ax1.plot(x_fit, irradiance_fit/scaling, linestyle='dashed',
                     label='$E(\Theta) = \dfrac{I_{max}}{D_{Flasher}^2} \cos(\Theta)$ $\,$ $\Theta = tan^{-1}(\dfrac{x-x_{shift}}{D_{Flasher}})$')
            ax1.legend(frameon=False, loc='lower left', ncol=2)
            ax1.set_ylabel('E [$\mu W/m^2$]')

            ax1.text(0.97, 0.70, """
                        $I_{max}$ : %.2f $\mu$W/sr
                        $freq.$ : %.3f
                        $shift$ : %.3f m
                        $D_{Flasher}$ : %.3f m """ % (amplitude/scaling, frequency, offset, distance),
                        fontsize=10, color='blue', horizontalalignment='right',
                        verticalalignment='bottom', transform=ax1.transAxes)


            ax2 = plt.subplot(gs[1])

            ax1.get_shared_x_axes().join(ax1, ax2)
            ax1.set_xticklabels([])

            ax2.scatter(x_position, (irradiance_fit - irradiance_in_x)/amplitude_err)
            ax2.set_ylabel('$g_{I_{max}}$')
            ax2.set_xlabel('x position [m]')

            #figure_name = './Output/Irradiance_interpolation/{}_y_{}.png'.format(file, y_label[i])
            #plt.savefig(figure_name)



            path = "/Users/lonewolf/PyCharmProjects/Flasher/Output/ave_irradiance_pixel.txt"
            path = './Output/Irradiance_interpolation/{}/{}_parameters.txt'.format(folder, filename)
            output = open(path, "w")
            output.write("pixel_id\t irradiance\n")
            for k in range(len(pixid)):
                output.write("{0}\t {1}\n".format(pixid[k], ave_irradiance_pixel[k]))
            output.close()

            plt.show()




del space_Data

