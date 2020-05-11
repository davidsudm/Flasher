# in Run.py



'''
# MAKE FIGURE FO THE SURFACE SCAN
dp.plot_cells(dim_x=steps, dim_y=steps, xo=xo, xf=xf, yo=yo, yf=yf, steps=steps)

# INTENSITY SURFACES - ABSOLUTE AND RELATIVE
space_Data = [file_list1, file_list2, file_list3]
for file_list in space_Data:
    for file in file_list:
        x, y, current, current_std, timestamp = read_file(file, 'space')

        dp.plot_intensity_scan_xy_2D(xo=xo, xf=xf, yo=yo, yf=yf, z=current, steps=steps, path_to_file=file, rel_label=True, cam_label=False)
        dp.plot_intensity_scan_xy_2D(xo=xo, xf=xf, yo=yo, yf=yf, z=current, steps=steps, path_to_file=file, rel_label=False, cam_label=False)

        dp.plot_intensity_scan_xy_3D(xo=xo, xf=xf, yo=yo, yf=yf, z=current, steps=steps, path_to_file=file, rel_label=True)
        dp.plot_intensity_scan_xy_3D(xo=xo, xf=xf, yo=yo, yf=yf, z=current, steps=steps, path_to_file=file, rel_label=False)

        del x, y, current, current_std, timestamp

del space_Data



# STABILITY IN CHANNELS
space_Data = [file_list2, file_list3]
for file_list in space_Data:
    for i, file in enumerate(file_list):
        x, y, current, current_std, timestamp = read_file(file, 'space')

        dp.plot_current_vs_channel(number_of_cell=number_of_cells, current=current, current_std=current_std, path_to_file=file, color=xkcd_colors[i], rel_label=True)
        dp.plot_current_vs_channel(number_of_cell=number_of_cells, current=current, current_std=current_std, path_to_file=file, color=xkcd_colors[i], rel_label=False)

        del x, y, current, current_std, timestamp

del space_Data



# STABILITY OF CHANNELS : MEAN DIFFERENCES
space_Data = [file_list2, file_list3]
for file_list in space_Data:
    dp.plot_mean_differences_current_vs_channel(number_of_cells=number_of_cells, file_list=file_list, rel_label=True)
    dp.plot_mean_differences_current_vs_channel(number_of_cells=number_of_cells, file_list=file_list, rel_label=False)

del space_Data
'''

'''
# PROJECTIONS OF AXIS : X AND Y
space_Data = [file_list2, file_list3]
for file_list in space_Data:
    for file in file_list:
        x, y, current, current_std, timestamp = read_file(file, 'space')

        dp.plot_projections(xo=xo, xf=xf, yo=yo, yf=yf, current=current, current_std=current_std, steps=steps, path_to_file=file, rel_label=False)
        dp.plot_projections(xo=xo, xf=xf, yo=yo, yf=yf, current=current, current_std=current_std, steps=steps, path_to_file=file, rel_label=True)

        del x, y, current, current_std, timestamp

del space_Data
'''

'''
x_off_list_file2 = []
y_off_list_file2 = []
x_off_list_file3 = []
y_off_list_file3 = []
# INTERPOLATION OF PROJECTIONS OF AXIS : X AND Y
space_Data = [file_list2, file_list3]
for file_list in space_Data:
    for file in file_list:
        x, y, current, current_std, timestamp = read_file(file, 'space')

        x_off_mean = dp.plot_projection_interpolation(xo=xo, xf=xf, current=current, current_std=current_std, steps=steps, path_to_file=file, rel_label=False, which_axis='x')
        y_off_mean = dp.plot_projection_interpolation(xo=yo, xf=yf, current=current, current_std=current_std, steps=steps, path_to_file=file, rel_label=False, which_axis='y')

        if file_list == file_list2:
            x_off_list_file2.append(x_off_mean)
            y_off_list_file2.append(y_off_mean)
        if file_list == file_list3:
            x_off_list_file3.append(x_off_mean)
            y_off_list_file3.append(y_off_mean)

        del x, y, current, current_std, timestamp

del space_Data
'''

'''
x_off_list_file2 = [96.28533023781698, 96.33290784514512, 96.26851257189254, 96.26714875400806, 96.28837846412944, 96.28777104458541, 96.29116775593614, 96.28407376338627, 96.25846073092333, 96.25711217088165, 96.59263196947005, 97.16424303201804]
y_off_list_file2 = [119.31044408552584, 119.32397325887884, 119.32231317081951, 119.31135779447139, 119.29710011373481, 119.294555434131, 119.29831256746283, 119.28651801871406, 119.29039141724296, 119.28692316884407, 119.33564250637878, 119.52970384262294]

x_off_list_file3 = [115.19459621449577, 117.38726785215225, 114.22214459998212, 115.55639082197885, 114.72631786541201, 114.66252806125175, 114.92880210699782, 114.91624548965206, 114.90880060235601, 114.91180800094266, 114.91737885121319]
y_off_list_file3 = [126.8800875390838, 126.84481602643679, 126.84342765506268, 126.84478234383526, 126.88119599423058, 126.7745595636221, 126.74114426960313, 126.7241558050257, 126.71949584096807, 126.71567503830961, 126.70893188617998]

x_off_list_file2 = np.around(x_off_list_file2, decimals=2)
y_off_list_file2 = np.around(y_off_list_file2, decimals=2)

x_off_list_file3 = np.around(x_off_list_file2, decimals=2)
y_off_list_file3 = np.around(y_off_list_file2, decimals=2)



# HOMOGENEITY : INTENSITY SURFACES - ABSOLUTE AND RELATIVE
space_Data = [file_list2, file_list3]
for file_list in space_Data:
    for i, file in enumerate(file_list):
        x, y, current, current_std, timestamp = read_file(file, 'space')

        if file_list == file_list2:
            x_cam = x_off_list_file2[i]
            y_cam = y_off_list_file2[i]
        if file_list == file_list3:
            x_cam = x_off_list_file3[i]
            y_cam = y_off_list_file3[i]

        #dp.plot_homogeneity(xo=xo, xf=xf, yo=yo, yf=yf, z=current, steps=steps, path_to_file=file, rel_label=True, cam_label=True, gauss_label=True, plot_interpolated_data=True, x_cam=x_cam, y_cam=y_cam)
        dp.plot_homogeneity(xo=xo, xf=xf, yo=yo, yf=yf, z=current, steps=steps, path_to_file=file, rel_label=True, cam_label=True, gauss_label=True, plot_interpolated_data=False, x_cam=x_cam, y_cam=y_cam)
        #dp.plot_homogeneity(xo=xo, xf=xf, yo=yo, yf=yf, z=current, steps=steps, path_to_file=file, rel_label=True, cam_label=True, gauss_label=False, plot_interpolated_data=True, x_cam=x_cam, y_cam=y_cam)
        dp.plot_homogeneity(xo=xo, xf=xf, yo=yo, yf=yf, z=current, steps=steps, path_to_file=file, rel_label=True, cam_label=True, gauss_label=False, plot_interpolated_data=False, x_cam=x_cam, y_cam=y_cam)

        del x, y, current, current_std, timestamp

del space_Data
'''

'''
# STABILITY IN TIME

space_Data = [file_list2, file_list3]

for file_list in space_Data:
    dp.plot_stability_in_time(file_list=file_list, rel_label=True)
    dp.plot_stability_in_time(file_list=file_list, rel_label=False)
del space_Data
'''

for i in range(0, steps):
    # for a fixed x position
    irradiance_in_y = data[:, i]
    irradiance_in_y_error = data_error[:, i]
    print('x :', x_position[i])
    print(irradiance_in_y)

    # fitting irradiance
    guessed_irradiance = np.max(irradiance_in_y)
    guessed_frequency = np.pi
    guessed_offset = 0.03
    initial_params = [guessed_irradiance, guessed_frequency, guessed_offset]
    pfit_leastsq, perr_leastsq = fitf.fit_irradiance(distance=distance,
                                                     initial_params=initial_params,
                                                     x=y_position,
                                                     y=irradiance_in_y)
    amplitude = pfit_leastsq[0]
    frequency = pfit_leastsq[1]
    offset = pfit_leastsq[2]

    amplitude_err = perr_leastsq[0]
    frequency_err = perr_leastsq[1]
    offset_err = perr_leastsq[2]

    print(amplitude, frequency, offset)
    irradiance_fit = fitf.irradiance(x=y_fit,
                                     distance=distance,
                                     amplitude=amplitude,
                                     frequency=frequency,
                                     offset=offset)

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

    ax1 = plt.subplot(gs[0])

    ax1.plot(y_position, irradiance_in_y / scaling, 'r-', label='data', linewidth=1.0)
    ax1.fill_between(x=y_position,
                     y1=(irradiance_in_y - irradiance_in_y_error) / scaling,
                     y2=(irradiance_in_y + irradiance_in_y_error) / scaling,
                     color=sns.xkcd_rgb["amber"], alpha=0.7)
    ax1.plot(y_fit, irradiance_fit / scaling, linestyle='dashed', label='fit')
    ax1.legend(frameon=False, loc='lower left')
    ax1.set_ylabel('E [$\mu W/m^2$]')

    ax1.text(0.97, 0.70, """
                $I_{max}$ : %.2f $\mu$W/sr
                $freq.$ : %.3f
                $shift$ : %.3f m
                $D_{Flasher}$ : %.3f m """ % (amplitude / scaling, frequency, offset, distance),
             fontsize=10, color='blue', horizontalalignment='right',
             verticalalignment='bottom', transform=ax1.transAxes)

    ax2 = plt.subplot(gs[1])

    ax1.get_shared_x_axes().join(ax1, ax2)
    ax1.set_xticklabels([])

    ax2.scatter(y_position, (irradiance_fit - irradiance_in_y) / amplitude_err)
    ax2.set_ylabel('$g_{I_{max}}$')
    ax2.set_xlabel('y position [m]')

    figure_name = './Output/Irradiance_interpolation/{}_x_{}.png'.format(file, x_label[i])
    plt.savefig(figure_name)




    FOR THE CASE OF NO FREQUENCY:

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
        shift_vector = []
        distance_vector = []

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


        data, data_error = comp.create_data_grid(data=irradiance,
                                                 data_errors=irradiance_error,
                                                 steps=steps,
                                                 rel_label=False)

        ax1, ax_cl = mydp.plot_intensity_scan_xy_2D(data=data,
                                                    x_limits=[-xf/2, xf/2],
                                                    y_limits=[-yf/2, yf/2],
                                                    data_label=True,
                                                    axes=None,
                                                    scaling=scaling)
        # The scaling factor defines the final unites for the color bar IN THE PLOT ONLY
        ax1.set_xlabel('x position [m]')
        ax1.set_ylabel('y position [m]')
        ax_cl.set_label('E [$\mu W/m^2$]')

        ax1.set_aspect(aspect=1)

        figure_name = './Output/Irradiance_interpolation/{}.png'.format(file)
        plt.savefig(figure_name)



        x_center, y_center = 0.15, 0.15
        x_label = x_position = np.linspace(xo, xf, steps)
        x_position = x_position - x_center
        y_label = y_position = np.linspace(yo, yf, steps)
        y_position = y_position - y_center

        # we use steps, but for fitting we could use more points
        x_fit = np.linspace(np.min(x_position), np.max(x_position), steps)
        y_fit = np.linspace(np.min(y_position), np.max(y_position), steps)

        for i in range(0, steps):

            # for a fixed y position
            irradiance_in_x = data[i, :]
            irradiance_in_x_error = data_error[i, :]
            print('y :', y_position[i])
            print(irradiance_in_x)

            """"""
            # fitting irradiance
            guessed_irradiance = np.max(irradiance_in_x)
            #guessed_frequency = np.pi
            guessed_offset = 0.03
            #initial_params = [guessed_irradiance, guessed_frequency, guessed_offset]
            initial_params = [guessed_irradiance, guessed_offset]
            pfit_leastsq, perr_leastsq = fitf.fit_irradiance(distance=distance,
                                                             initial_params=initial_params,
                                                             x=x_position,
                                                             y=irradiance_in_x)
            amplitude = pfit_leastsq[0]
            #frequency = pfit_leastsq[1]
            #offset = pfit_leastsq[2]
            offset = pfit_leastsq[1]

            amplitude_err = perr_leastsq[0]
            #frequency_err = perr_leastsq[1]
            #offset_err = perr_leastsq[2]
            offset_err = perr_leastsq[1]

            #print(amplitude, frequency, offset)
            print(amplitude, offset)
            """
            irradiance_fit = fitf.irradiance(x=x_fit,
                                             distance=distance,
                                             amplitude=amplitude,
                                             frequency=frequency,
                                             offset=offset)
             """
            irradiance_fit = fitf.irradiance(x=x_fit,
                                             distance=distance,
                                             amplitude=amplitude,
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

            '''
            ax1.text(0.97, 0.70, """
                        $I_{max}$ : %.2f $\mu$W/sr
                        $freq.$ : %.3f
                        $shift$ : %.3f m
                        $D_{Flasher}$ : %.3f m """ % (amplitude/scaling, frequency, offset, distance),
                        fontsize=10, color='blue', horizontalalignment='right',
                        verticalalignment='bottom', transform=ax1.transAxes)
            '''

            ax1.text(0.97, 0.70, """
                                    $I_{max}$ : %.2f $\mu$W/sr
                                    $shift$ : %.3f m
                                    $D_{Flasher}$ : %.3f m """ % (amplitude/scaling, offset, distance),
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



del space_Data
