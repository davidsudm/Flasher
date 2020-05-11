#! /anaconda3/envs/Flasher/bin/python

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib.patches import Polygon
from matplotlib import colors
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import datetime

from readout import read_file
import datetime as dt
import seaborn as sns
import numpy as np
import scipy
import fitfunction as ff
import readout as readout
from xkcd_colors import xkcd_colors
from scipy import optimize


def plot_intensity_scan_xy_2D(data, x_limits=[0., 300], y_limits=[0., 300], data_label=False, scaling=1, axes=None):
    """
    Plots the scanned surface data value on a 2-D graph
    :param data:            (float array) 2-D data array
    :param x_limits:        (float) list with the beginning and end of X-axis, i.e. [xo, xf]
    :param y_limits:        (float) list with the beginning and end of Y-axis, i.e. [yo, yf]
    :param data_label:      (bool) True : show the data values in each cell of the array -  False : don't show data values
    :param axes:            (axes class) None : a new axes created as a default
    :return: axes
    """
    dimension = data.shape
    x = np.linspace(x_limits[0], x_limits[1], dimension[0])
    y = np.linspace(y_limits[0], y_limits[1], dimension[1])

    detector_size = np.array([x[1]-x[0], y[1]-y[0]])

    if axes is None:
        fig, ax = plt.subplots()
    else:
        ax = axes

    im = ax.imshow(data/scaling, cmap=cm.viridis, interpolation='none', origin='lower', extent=[-detector_size[0]/2 + x.min(),
                                                                                        detector_size[0]/2 + x.max(),
                                                                                        -detector_size[1]/2 + y.min(),
                                                                                        detector_size[1]/2 + y.max()])

    ax.set_xlabel('x position [mm]')
    ax.set_ylabel('y position [mm]')

    label = 'Intensity [nA]'
    #label = 'Intensity [%]'

    color_axes = plt.colorbar(im, label=label)
    #print(fig.axes)

    # Python works with matrix representation which means
    # lines and columns and in that language
    # x axis points are columns, so 'j' symbol and
    # y axis points are lines, so 'i' symbol
    if data_label:
        for i, y_pos in enumerate(y):
            for j, x_pos in enumerate(x):
                text = ax.text(x=x_pos, y=y_pos, s=np.around(data[i, j]/scaling, decimals=2),
                               ha="center",
                               va="center",
                               color="black",
                               size='6')

    ax.set_aspect(aspect=1)

    return ax, color_axes, fig


def draw_camera(axes=None, vertex_to_vertex=1120.1, focal_length=5.6, camera_centre=[150.0, 150.0], **kwargs):
    """
    Draws the camera profile as it was place a focal length away from the flasher
    :param axes:                (axes class) None : a new axes created as a default
    :param vertex_to_vertext:   (float) the vertex-to-vertex dimension of the camera, by default 1120.1 mm
    :param focal_length:        (float) the focal length of the telescope, by default 5.6 m, useful for the scalling to the camera on a plane a focal length aways from it
    :param camera_centre:       (float array) the centre of the camera
    :param kwargs:              (kwargs) option for matplotlib.polygon
    :return: axes
    """

    if axes is None:
        fig, ax = plt.subplots()
    else:
        ax = axes

    r = vertex_to_vertex / 2.  # mm
    xo_hexagon = camera_centre[0]
    yo_hexagon = camera_centre[1]

    angles = np.array((30., 90., 150., 210., 270., 330.)) * (np.pi / 180.)
    x_position, y_position = [r * np.cos(angles), r * np.sin(angles)]
    points = [[x_position[0], y_position[0]],
              [x_position[1], y_position[1]],
              [x_position[2], y_position[2]],
              [x_position[3], y_position[3]],
              [x_position[4], y_position[4]],
              [x_position[5], y_position[5]]] / np.array(focal_length)
    points = points + np.array([xo_hexagon, yo_hexagon])
    hexagon = Polygon(points, fill=False, edgecolor='black', **kwargs)
    ax.add_patch(hexagon)
    ax.set_aspect(aspect=1)

    return ax

def plot_intensity_contour(data, x_limits=[0., 300], y_limits=[0., 300], data_label=False, contour_levels=[80., 90., 100.], axes=None):
    """
    Draws contour lines
    :param data:            (float array) 2-D data array
    :param x_limits:        (float) list with the beginning and end of X-axis, i.e. [xo, xf]
    :param y_limits:        (float) list with the beginning and end of Y-axis, i.e. [yo, yf]
    :param data_label:      (bool) True : show the data values in each cell of the array -  False : don't show data values
    :param axes:            (axes class) None : a new axes created as a default
    :return: axes
    """

    class nf(float):
        def __repr__(self):
            s = f'{self:.1f}'
            return f'{self:.0f}' if s[-1] == '0' else s

    if axes is None:
        fig, ax = plt.subplots()
    else:
        ax = axes

    dimension = data.shape
    x = np.linspace(x_limits[0], x_limits[1], dimension[1])
    y = np.linspace(y_limits[0], y_limits[1], dimension[0])

    cp_levels = contour_levels

    x, y = np.meshgrid(x, y)
    cp = ax.contour(x, y, np.flipud(data), cmap=cm.cividis, levels=cp_levels, linestyles='dashed')
    #cp = ax.contour(x, y, np.flipud(data), cmap=cm.rainbow, levels=cp_levels, linestyles='dashed')
    # Recast levels to new class
    cp.levels = [nf(val) for val in cp.levels]

    # Label levels with specially formatted floats
    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%%'
    else:
        fmt = '%r %%'

    #ax.clabel(cp, inline=True, fontsize=10)
    ax.clabel(cp, cp.levels, inline=True, fmt=fmt, fontsize=10)
    ax.set_aspect(aspect=1)

    return ax


def plot_led_spectra(interpolated_function, x_limits=[200, 1000], point_array=None, name_array=None, axes=None):
    """

    :param interpolated_function:
    :param x_limits:
    :param point_array:
    :param name_array:
    :param axes:
    :return: axes
    """

    if axes is None:
        fig, ax = plt.subplots()
    else:
        ax = axes

    x = np.linspace(np.min(x_limits), np.max(x_limits), np.max(x_limits) - np.min(x_limits) + 1)
    y = interpolated_function(x)

    ax.plot(x, y, linestyle='dashed')

    if point_array is not None:
        for i, name in enumerate(name_array):
            x_point = point_array[i]
            y_point = interpolated_function(x_point)
            ax.plot(x_point, y_point, 'o', label=name_array[i]+' at {} [nm] : {}'.format(np.around(x_point, decimals=3), np.around(y_point, decimals=3)))
        ax.legend()

    ax.set_ylabel(r'Normalized intensity []')
    ax.set_xlabel('Wavelength [mn]')

    return ax


def plot_thermal_coefficient(interpolated_function, x_limits=[400, 1100], point_array=None, name_array=None, axes=None):
    """

    :param interpolated_function:
    :param x_limits:
    :param point_array:
    :param name_array:
    :param axes:
    :return: axes
    """

    if axes is None:
        fig, ax = plt.subplots()
    else:
        ax = axes

    x = np.linspace(np.min(x_limits), np.max(x_limits), np.max(x_limits) - np.min(x_limits) + 1)
    y = interpolated_function(x)

    ax.plot(x, y, linestyle=':', color=sns.xkcd_rgb['azure'])

    if point_array is not None:
        for i, name in enumerate(name_array):
            x_point = point_array[i]
            y_point = interpolated_function(x_point)
            ax.plot(x_point, y_point, 's', color=xkcd_colors[i], label=name_array[i]+' : {} %/$^\circ$C'.format(np.around(y_point, decimals=3)))
        ax.legend()

    ax.set_ylabel(r'Temperature Coefficient [%/$^\circ$C]')
    ax.set_xlabel('Wavelength [mn]')

    return ax


def plot_masked_intensity_and_temperature(mask_array, time, data, data_error, temperature, means, error_propagation, y_label, y_units, decimals=3, temperature_array=[25, 20, 15, 10, 5], axes=None):
    """

    :param mask_array:
    :param time:
    :param data:
    :param data_error:
    :param temperature:
    :param means:
    :param error_propagation:
    :param y_label:
    :param y_units:
    :param decimals:
    :param temperature_array:
    :param axes:
    :return:
    """


    if axes is None:
        fig, ax1 = plt.subplots()
    else:
        ax1 = axes

    temperature_array = np.array(temperature_array)
    decimals = decimals

    time = time / 3600.

    color = sns.xkcd_rgb['neon pink']
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylabel('Temperature [°C]', color=color)  # we already handled the x-label with ax1
    ax1.plot(time, temperature, color=color, linestyle='dashed')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlabel('Time [hours]')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = sns.xkcd_rgb['purple blue']
    ax2.set_ylabel(y_label, color=color)

    for i, a_temperature in enumerate(temperature_array):
        ax2.plot(time[mask_array[i]], data[mask_array[i]],
                 label=r'$E (T={}^\circ C)$ = {} ± {} '.format(np.around(temperature_array[i], decimals=decimals),
                                                               np.around(means[i], decimals=decimals),
                                                               np.around(error_propagation[i], decimals=decimals))
                       + y_units)
        if data_error is not None:
            ax2.fill_between(time[mask_array[i]], data[mask_array[i]] - data_error[mask_array[i]], data[mask_array[i]] + data_error[mask_array[i]], color=sns.xkcd_rgb["amber"], alpha=0.7)

    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(fontsize='small', frameon=False)
    fig.tight_layout()

    return ax1, ax2


def plot_full_intensity_and_temperature_vs_time(temperature_time, temperature, current_time, current, current_std=None, initial_cut=None, axes=None):
    """

    :param temperature_time:         primordial_temperature_time
    :param temperature:              primordial_temperature
    :param current_time:             primordial_current_time
    :param current:                  primordial_current
    :param current_std:              primordial_current_std
    :param initial_cut:              initial_cut
    :param axes:                     axes
    :return:
    """

    if axes is None:
        fig, ax1 = plt.subplots()
    else:
        ax1 = axes

    if isinstance(current_time[0], datetime.date):
        ax1.set_xlabel('Date, Time')
        rotation = 30.
    else:
        ax1.set_xlabel('Hours')
        temperature_time = temperature_time / 3600.
        current_time = current_time / 3600.
        rotation = 0.

    if initial_cut is not None:
        temperature_time = temperature_time[initial_cut:]
        temperature = temperature[initial_cut:]
        current_time = current_time[initial_cut:]
        current = current[initial_cut:]
        if current_std is not None:
            current_std = current_std[initial_cut:]

    color = sns.xkcd_rgb['neon pink']
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylabel('T [°C]', color=color)  # we already handled the x-label with ax1
    ax1.plot(temperature_time, temperature, color=color, linestyle='dashed', label='Temperature')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = sns.xkcd_rgb['purple blue']
    ax2.set_ylabel(r'E [$\mu W/m^2$]', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax2.plot(current_time, current, color=color, label='Irradiance')

    if current_std is not None:
        ax2.fill_between(current_time, current - current_std, current + current_std, color=sns.xkcd_rgb['amber'], alpha=0.7, label='current error')

    ax1.tick_params(axis='x', rotation=rotation)

    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    line_purple = mlines.Line2D([], [], color=sns.xkcd_rgb['purple blue'], label='Irradiance', linewidth=2)
    line_pink = mlines.Line2D([], [], color=sns.xkcd_rgb['neon pink'], label='Temperature', linestyle='dashed')
    amber_patch = mpatches.Patch(color=sns.xkcd_rgb['amber'], label='Irradiance', linewidth=2)

    ax1.legend([(amber_patch, line_purple), line_pink], ['Irradiance', 'Temperature'], frameon=False)

    fig.tight_layout()

    return ax1


def plot_slope_for_thermal_coefficient(tdf, rdf, rdf_error, slopes, slopes_error, intersects, intersects_error, name, temperature_array=[25, 20, 15, 10, 5], axes=None):
    """

    :param tdf:
    :param rdf:
    :param rdf_error:
    :param slopes:
    :param slopes_error:
    :param intersects:
    :param intersects_error:
    :param name:
    :param axes:
    :return:
    """

    rdf = rdf * np.array([100])
    rdf_error = rdf_error * np.array([100])

    slopes = slopes * np.array([100])
    slopes_error = slopes_error * np.array([100])

    intersects = intersects * np.array([100])
    intersects_error = intersects_error * np.array([100])

    decimals = 3
    my_temperatures = np.array(temperature_array)
    axes_array = []
    for i in range(0, len(my_temperatures)):
        if axes is None:
            fig, ax = plt.subplots()
        else:
            ax = axes

        x_array = np.linspace(np.min(tdf[i]), np.max(tdf[i]), 1000)
        y_array = slopes[i] * x_array + intersects[i]

        ax.errorbar(tdf[i], rdf[i], rdf_error[i],
                    label='Relative differences wrt {}$^\circ$C'.format(my_temperatures[i]), fmt='o', color='black',
                    ecolor=sns.xkcd_rgb['amber'], elinewidth=3, capsize=4, ms=4, fillstyle='full')

        color = sns.xkcd_rgb['carolina blue']
        ax.plot(x_array, y_array, color=color, label='Linear fit: ({} ± {} %/$^\circ$C) $\Delta$T + ({} ± {} %)'.format(
            np.around(slopes[i], decimals=decimals), np.around(slopes_error[i], decimals=decimals),
            np.around(intersects[i], decimals=decimals), np.around(intersects_error[i], decimals=decimals)))

        ax.set_ylabel('Relative Difference [%]')
        ax.set_xlabel('$\Delta$T [$^\circ$C]')
        ax.legend()
        fig.tight_layout()
        axes_array.append(ax)
        figure_name = './Output/Thermal_Coefficients/{}_at_{}_Celsius.png'.format(name, my_temperatures[i])
        plt.savefig(figure_name)


    return axes_array