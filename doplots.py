#! /anaconda3/envs/Flasher/bin/python

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib.patches import Polygon
from matplotlib import colors
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import datetime as dt
import seaborn as sns
import numpy as np
import scipy
import fitfunction as ff
import readout as readout
from xkcd_colors import xkcd_colors
from scipy import optimize




def plot_intensity_vs_time(time, current, current_std, color, path_to_file, time_label, rel_label):
    # time          : time numpy array
    # current       : current numpy array
    # std           : standard deviation of the measurement numpy array
    # color         : color as string type
    # path_to_file  : path to the file as string type (usually a element of a file_list)
    # time_label    : Boolean (True : print x labels as time format - False : print x labels as index format)
    # rel_label     : Boolean (True : compute relative intensity wrt the maximum -  False : just the absolute intensity)

    # folder and plot_name came from one component of a file list which is a string type, divided into 2 strings
    folder = path_to_file.split("/")[0]
    filename = path_to_file.split("/")[1]
    plot_name = filename.split('.')[0]

    ax = plt.gca()
    line_width = 0.4

    if time_label:
        plt.xticks(rotation=60)
        xfmt = md.DateFormatter('%Y/%m/%d %H:%M:%S')
        ax.xaxis.set_major_formatter(xfmt)
        if rel_label:
            rel_current = current / np.max(current)
            error_ratio = current_std / current
            index_of_max = np.argmax(current)
            delta_rel_current = rel_current * np.sqrt(error_ratio ** 2 + error_ratio[index_of_max] ** 2)

            rel_current *= np.array(100)
            delta_rel_current *= np.array(np.abs(100))

            plt.ylabel('Relative Intensity [%]')
            plt.title('Relative Intensity in time : {}'.format(plot_name))
            plt.plot(time, rel_current, '{}'.format(color), marker=',', linewidth=line_width)
            plt.fill_between(time, rel_current - delta_rel_current, rel_current + delta_rel_current, color=sns.xkcd_rgb["amber"])
            plt.savefig('./Output/{}/{}_relative_time_stamps.png'.format(folder, plot_name), bbox_inches='tight')
        else:
            plt.ylabel('Current [nA]')
            plt.title('Intensity in time : {}'.format(plot_name))
            plt.plot(time, current, '{}'.format(color), marker=',', linewidth=line_width)
            plt.fill_between(time, current - current_std, current + current_std, color=sns.xkcd_rgb['amber'])
            plt.savefig('./Output/{}/{}_absolute_time_stamps.png'.format(folder, plot_name), bbox_inches='tight')
    else:
        if rel_label:
            rel_current = current / np.max(current)
            error_ratio = current_std / current
            index_of_max = np.argmax(current)
            delta_rel_current = rel_current * np.sqrt(error_ratio ** 2 + error_ratio[index_of_max] ** 2)

            rel_current *= np.array(100)
            delta_rel_current *= np.array(np.abs(100))

            plt.ylabel('Relative Intensity [%]')
            plt.title('Relative Intensity in time : {}'.format(plot_name))
            plt.plot(range(len(current)), current, '{}'.format(color), marker=',', linewidth=line_width)
            plt.fill_between(range(len(rel_current)), rel_current - delta_rel_current, rel_current + delta_rel_current, color=sns.xkcd_rgb["amber"])
            plt.savefig('./Output/{}/{}_relative_time_points.png'.format(folder, plot_name), bbox_inches='tight')
        else:
            plt.ylabel('Current [nA]')
            plt.title('Intensity in time : {}'.format(plot_name))
            plt.plot(range(len(current)), current, '{}'.format(color), marker=',', linewidth=line_width)
            plt.fill_between(range(len(current)), current - std, current + std, color=sns.xkcd_rgb["amber"])
            plt.savefig('./Output/{}/{}_absolute_time_points.png'.format(folder, plot_name), bbox_inches='tight')
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    plt.show()
    plt.clf()


def plot_intensity_scan_xy_2D(xo, xf, yo, yf, z, steps, path_to_file, rel_label, cam_label):
    # xo            : beginning of X-axis
    # xf            : end of X-axis
    # yo            : beginning of Y-axis
    # yf            : end of Y-axis
    # steps         : number of measurements in along the axis X and Y (it should be equal in both axis)
    # path_to_file  : path to the file as string type (usually a element of a file_list)
    # rel_label     : Boolean (True : compute relative intensity wrt the maximum -  False : just the absolute intensity)
    # cam_label     : Boolean (True : draw camera -  False : doesn't draw camera)

    # folder and plot_name came from one component of a file list which is a string type, divided into 2 strings
    folder = path_to_file.split("/")[0]
    filename = path_to_file.split("/")[1]
    plot_name = filename.split('.')[0]

    prefix = ''

    x = np.linspace(xo, xf, steps)
    y = np.linspace(yo, yf, steps)

    if rel_label:
        z = z / np.max(z)
        z = np.reshape(z, (steps, steps)) * np.array(100)
        prefix += 'relative'
    else:
        z = np.reshape(z, (steps, steps))
        prefix += 'absolute'

    # We need to take the transpose, otherwise we can't match the actually file position (x,y)
    z = z.T

    fig, ax = plt.subplots()
    im = ax.imshow(z, cmap=cm.viridis, interpolation='none', origin='lower', extent=[-15., 315., -15., 315.])

    ax.set_xticks(x)
    ax.set_yticks(y)

    plt.xlabel('x position [mm]')
    plt.ylabel('y position [mm]')

    if rel_label:
        plt.colorbar(im, label='Relative Intensity [%]')
        plt.title('Relative Data : {}'.format(plot_name))
        #plt.savefig('./Output/{}/{}_Data_2D_rel.png'.format(folder, plot_name), bbox_inches='tight')
    else:
        plt.colorbar(im, label='Intensity [nA]')
        plt.title('Absolute Data : {}'.format(plot_name))
        #plt.savefig('./Output/{}/{}_Data_2D.png'.format(folder, plot_name), bbox_inches='tight')

    # Loop over data dimensions and create text annotations.
    for i, xx in enumerate(x):
        for j, yy in enumerate(y):
            text = ax.text(yy, xx, np.around(z[i, j], decimals=2), ha="center", va="center", color="black", size='6')

    fig.tight_layout()

    # Draw the Camera at a distance of 5.6 m wrt the flasher if "cam_level" si set True
    if cam_label:
        prefix += '_camON'
        r = 1120. / 2.  # mm
        distance = 5.6  # m ~scaling factor for small angles
        x_center = 120.
        y_center = 120.

        angles = np.array((30., 90., 150., 210., 270., 330.)) * (np.pi / 180.)
        x_position, y_position = [r * np.cos(angles), r * np.sin(angles)]

        points = [[x_position[0], y_position[0]], [x_position[1], y_position[1]], [x_position[2], y_position[2]], [x_position[3], y_position[3]], [x_position[4], y_position[4]], [x_position[5], y_position[5]]] / np.array(distance)
        points = points + np.array([x_center, y_center])
        hexagon = Polygon(points, fill=False, edgecolor='black', linestyle='-', linewidth=0.5)
        #hexagon.set_alpha(0.1)
        ax.add_patch(hexagon)
        ax.set_aspect(aspect=1.0)
        ax.set_xlim((-15, 315))
        ax.set_ylim((-15, 315))
        ax.set_aspect(1)
    else:
        prefix += '_camOFF'

    plt.savefig('./Output/{}/{}_{}_2D.png'.format(folder, plot_name, prefix), bbox_inches='tight')
    plt.show()
    plt.clf()


def plot_intensity_scan_xy_3D(xo, xf, yo, yf, z, steps, path_to_file, rel_label):
    # xo            : beginning of X-axis
    # xf            : end of X-axis
    # yo            : beginning of Y-axis
    # yf            : end of Y-axis
    # steps         : number of measurements in along the axis X and Y (it should be equal in both axis)
    # path_to_file  : path to the file as string type (usually a element of a file_list)
    # rel_label     : Boolean (True : compute relative intensity wrt the maximum -  False : just the absolute intensity)

    # folder and plot_name came from one component of a file list which is a string type, divided into 2 strings
    folder = path_to_file.split("/")[0]
    filename = path_to_file.split("/")[1]
    plot_name = filename.split('.')[0]

    prefix = ''

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(30, -45)
    xx = x = np.linspace(xo, xf, steps)
    yy = y = np.linspace(yo, yf, steps)
    x, y = np.meshgrid(x, y)

    if rel_label:
        z = z / np.max(z)
        z = np.reshape(z, (steps, steps)) * np.array(100)
        prefix += 'relative'
    else:
        z = np.reshape(z, (steps, steps))
        prefix += 'absolute'

    # We need to take the transpose, otherwise we can't match the actually file position (x,y)
    z = z.T

    surf = ax.plot_surface(x, y, z, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))

    ax.set_xticks(xx, minor=False)
    ax.set_yticks(yy, minor=False)

    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel('X position [mm]')
    ax.set_ylabel('Y position [mm]')
    # Add a color bar which maps values to colors.
    plt.xticks(rotation=10)
    plt.yticks(rotation=-10)
    if rel_label:
        plt.title('Relative Data : {}'.format(plot_name))
        fig.colorbar(surf, shrink=0.5, aspect=10, label='Relative Intensity [%]')
    else:
        plt.title('Relative Data : {}'.format(plot_name))
        fig.colorbar(surf, shrink=0.5, aspect=10, label='Intensity [nA]')

    plt.savefig('./Output/{}/{}_{}_3D.png'.format(folder, plot_name, prefix), bbox_inches='tight')
    plt.show()
    plt.clf()


def plot_intensity_contour(xo, xf, yo, yf, z, steps, path_to_file, rel_label):

    # folder and plot_name came from one component of a file list which is a string type, divided into 2 strings
    folder = path_to_file.split("/")[0]
    filename = path_to_file.split("/")[1]
    plot_name = filename.split('.')[0]

    fig, ax = plt.subplots()
    # Draw the contour plots from the relative max. intensity
    x = np.linspace(xo, xf, steps)
    y = np.linspace(yo, yf, steps)
    x, y = np.meshgrid(x, y)

    if rel_label:
        z = np.reshape(z, (steps, steps)) * np.array(100)
    else:
        z = np.reshape(z, (steps, steps))

    # We need to take the transpose, otherwise we can't match the actually file position (x,y)
    z = z.T

    cs_levels = 8
    cpf = ax.contourf(x, y, z, levels=cs_levels, cmap=cm.viridis)
    colours = ['w' if level < 0 else 'k' for level in cpf.levels]

    cs = ax.contour(x, y, z, levels=cs_levels, colors=colours, extent=[-15, 315, -15, 315])
    ax.set_aspect(aspect=1)
    ax.clabel(cs, inline=1, fontsize=10, colors=colours)
    ax.set_xticks(np.linspace(xo, xf, steps), minor=False)
    ax.set_yticks(np.linspace(yo, yf, steps), minor=False)

    plt.xlabel('x position [mm]')
    plt.ylabel('y position [mm]')
    plt.grid()

    if rel_label:
        plt.title('Relative Intensity : {}'.format(plot_name))
        fig.colorbar(cpf, cmap=cm.viridis, label='Relative Intensity [%]')
        plt.savefig('./Output/{}/{}_Contour_rel.png'.format(folder, plot_name), bbox_inches='tight')
    else:
        plt.title('Intensity : {}'.format(plot_name))
        fig.colorbar(cpf, cmap=cm.viridis, label='Intensity [nA]')
        plt.savefig('./Output/{}/{}_Contour.png'.format(folder, plot_name), bbox_inches='tight')

    plt.savefig('/Users/lonewolf/Desktop/{}_{}_Data_3D.png'.format(folder, plot_name), bbox_inches='tight')
    plt.show()
    plt.clf()



def plot_cells(dim_x, dim_y, xo, xf, yo, yf, steps):
    x = np.linspace(xo, xf, steps)
    y = np.linspace(yo, yf, steps)
    matrix_of_cells = (np.arange(0, dim_x * dim_y, 1)).reshape((dim_x, dim_y)).T

    # Drawing
    fig, ax = plt.subplots()
    ax = plt.gca()

    im = ax.imshow(matrix_of_cells, origin='lower', extent=[-15, 315, -15, 315])

    # Loop over data dimensions and create text annotations.
    for i, xx in enumerate(x):
        for j, yy in enumerate(y):
            text = ax.text(yy, xx, matrix_of_cells[i, j], ha="center", va="center", color="black", size='6')

    fig.colorbar(im, ax=ax, label='Time [steps]')
    ax.set_xticks(np.linspace(xo, xf, steps))
    ax.set_yticks(np.linspace(xo, xf, steps))
    plt.xlabel('x position [mm]')
    plt.ylabel('y position [mm]')
    plt.title('Channels or cells in surface scan')
    plt.savefig('./Output/Others/Scanned_Surface.png', bbox_inches='tight')
    plt.show()
    plt.clf()

def plot_current_vs_channel(number_of_cell, current, current_std, path_to_file, color, rel_label):
    # folder and plot_name came from one component of a file list which is a string type, divided into 2 strings
    folder = path_to_file.split("/")[0]
    filename = path_to_file.split("/")[1]
    plot_name = filename.split('.')[0]

    cells = np.arange(0, number_of_cell, 1)

    if rel_label:
        string = 'relative'
        axis_label = 'Relative Intensity [%]'
        rel_current = current / np.max(current)
        error_ratio = current_std / current
        index_of_max = np.argmax(current)
        delta_rel_current = rel_current * np.sqrt(error_ratio ** 2 + error_ratio[index_of_max] ** 2)

        matrix = rel_current * 100
        matrix_std = delta_rel_current * np.abs(100)

    else:
        string = 'absolute'
        axis_label = 'Intensity [nA]'
        matrix = current
        matrix_std = current_std

    plt.plot(cells, matrix, 'k-', linewidth='0.5', color='black')
    plt.fill_between(cells, matrix - matrix_std, matrix + matrix_std, color=color)

    plt.ylabel(axis_label)
    plt.xlabel('Cell number')
    plt.title('{} : {}'.format(folder, plot_name))
    plt.savefig('./Output/{}/{}_{}_current_vs_channels.png'.format(folder, plot_name, string), bbox_inches='tight')
    plt.show()
    plt.clf()

def plot_mean_differences_current_vs_channel(number_of_cells, file_list, rel_label):

    # For the mean :
    cells = np.arange(0, number_of_cells, 1)
    current_matrix = np.zeros((number_of_cells, len(file_list)))
    current_std_matrix = np.zeros((number_of_cells, len(file_list)))

    for i, file in enumerate(file_list):
        x, y, current, current_std, timestamp = readout.read_file(file, 'space')

        # folder and plot_name came from one component of a file list which is a string type, divided into 2 strings
        folder = file.split("/")[0]
        filename = file.split("/")[1]
        plot_name = filename.split('.')[0]

        if rel_label:
            string = 'relative'
            axis_label = 'Relative Intensity [%]'
            rel_current = current / np.max(current)
            error_ratio = current_std / current
            index_of_max = np.argmax(current)
            delta_rel_current = rel_current * np.sqrt(error_ratio ** 2 + error_ratio[index_of_max] ** 2)

            current = rel_current * 100
            current_std = delta_rel_current * np.abs(100)

        else:
            string = 'absolute'
            axis_label = 'Intensity [nA]'

        current_matrix[:, i] = current
        current_std_matrix[:, i] = current_std

        plt.scatter(cells, current, linewidth='0.3', label=plot_name, color=xkcd_colors[i])
        del x, y, current, current_std, timestamp


    mean = current_matrix.mean(axis=1)
    mean_std = current_matrix.std(axis=1)

    plt.plot(cells, mean, 'k-', linewidth='0.5', label='mean of runs')
    plt.fill_between(cells, mean - mean_std, mean + mean_std, color=sns.xkcd_rgb['amber'])
    plt.legend(bbox_to_anchor=(0, 1.10, 1, 0.2), loc="lower left", mode='expand', ncol=4, fontsize=8)

    plt.ylabel(axis_label)
    plt.xlabel('Cell number')
    plt.title('Pixel Stability : {}'.format(folder))
    plt.savefig('./Output/{}/all_{}_mean_current_vs_channels.png'.format(folder, string), bbox_inches='tight')
    plt.show()
    plt.clf()
    del string

    # For the differences :
    string = ''
    for i, file in enumerate(file_list):
        x, y, current, current_std, timestamp = readout.read_file(file, 'space')

        # folder and plot_name came from one component of a file list which is a string type, divided into 2 strings
        folder = file.split("/")[0]
        filename = file.split("/")[1]
        plot_name = filename.split('.')[0]

        if rel_label:
            string += 'relative'
            axis_label = 'Relative - not much meaning'
            rel_current = current / np.max(current)
            error_ratio = current_std / current
            index_of_max = np.argmax(current)
            delta_rel_current = rel_current * np.sqrt(error_ratio ** 2 + error_ratio[index_of_max] ** 2)

            current = rel_current * 100
            current_std = delta_rel_current * np.abs(100)

        else:
            string += 'absolute'
            axis_label = 'Relative Difference wrt Mean [%]'

        differences = ((mean - current) / mean) * np.array(100)
        plt.plot(cells, differences, 'k-', linewidth='1.', label=plot_name, color=xkcd_colors[i])

    plt.legend(bbox_to_anchor=(0, 1.10, 1, 0.2), loc="lower left", mode='expand', ncol=4, fontsize=8)
    plt.ylabel(axis_label)
    plt.xlabel('Cell number')
    plt.title('Pixel Stability : {}'.format(folder))
    plt.savefig('./Output/{}/all_{}_differences_current_vs_channels.png'.format(folder, string), bbox_inches='tight')
    plt.show()
    plt.clf()


def plot_projections(xo, xf, yo, yf, current, current_std, steps, path_to_file, rel_label):
    # xo                     : beginning of X-axis
    # xf                     : end of X-axis
    # yo                     : beginning of Y-axis
    # yf                     : end of Y-axis
    # steps                  : number of measurements in along the axis X and Y (it should be equal in both axis)
    # path_to_file           : path to the file as string type (usually a element of a file_list)
    # rel_label              : Boolean (True : compute relative intensity wrt the maximum -  False : just the absolute intensity)


    # folder and plot_name came from one component of a file list which is a string type, divided into 2 strings
    folder = path_to_file.split("/")[0]
    filename = path_to_file.split("/")[1]
    plot_name = filename.split('.')[0]

    x_axis = np.linspace(xo, xf, steps)
    y_axis = np.linspace(yo, yf, steps)

    string = ''

    if rel_label:
        string += 'relative'
        axis_label = 'Relative Intensity [%]'
        rel_current = current / np.max(current)
        error_ratio = current_std / current
        index_of_max = np.argmax(current)
        delta_rel_current = rel_current * np.sqrt(error_ratio**2 + error_ratio[index_of_max]**2)

        matrix = (np.reshape(rel_current, (steps, steps))).T * 100
        matrix_std = (np.reshape(delta_rel_current, (steps, steps))).T * np.abs(100)

    else:
        string += 'absolute'
        axis_label = 'Intensity [nA]'
        matrix = (np.reshape(current, (steps, steps))).T
        matrix_std = (np.reshape(current_std, (steps, steps))).T

    for i in range(len(x_axis)):
        plt.plot(x_axis, matrix[i, :], color='black', linewidth=0.5, marker='o', markerfacecolor='black', markersize=2)
        plt.fill_between(x_axis, matrix[i, :] - matrix_std[i, :], matrix[i, :] + matrix_std[i, :], color=xkcd_colors[i])
        #for j in range(len(y_axis)):
        #    plt.annotate(np.around(matrix[i, j], decimals=2), ((x_axis[j]), matrix[i, j]))
    plt.title('{} : {} - X projection'.format(plot_name, folder))
    plt.xlabel('x axis [mm]')
    plt.ylabel(axis_label)
    plt.savefig('./Output/{}/{}_{}_Xproj.png'.format(folder, string, plot_name), bbox_inches='tight')
    plt.show()

    for j in range(len(y_axis)):
        plt.plot(y_axis, matrix[:, j], color='black', linewidth=0.5, marker='o', markerfacecolor='black', markersize=2)
        plt.fill_between(y_axis, matrix[:, j] - matrix_std[:, j], matrix[:, j] + matrix_std[:, j], color=xkcd_colors[j])
        #for i in range(len(x_axis)):
        #    plt.annotate(np.around(matrix[j, i], decimals=2), ((y_axis[j]), matrix[j, i]))

    plt.title('{} : {} - Y projection'.format(plot_name, folder))
    plt.xlabel('y axis [mm]')
    plt.ylabel(axis_label)
    plt.savefig('./Output/{}/{}_{}_Yproj.png'.format(folder, string, plot_name), bbox_inches='tight')
    plt.show()
    plt.clf()


def plot_stability_in_time(file_list, rel_label):
    # file_list     : list of file paths
    # rel_label     : Boolean (True : compute relative intensity wrt the maximum -  False : just the absolute intensity)


    fig, ax = plt.subplots()
    xfmt = md.DateFormatter('%Y/%m/%d %H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)


    for i, file in enumerate(file_list):
        # folder and plot_name came from one component of a file list which is a string type, divided into 2 strings
        folder = file.split("/")[0]
        filename = file.split("/")[1]
        plot_name = filename.split('.')[0]

        x, y, current, current_std, timestamp = readout.read_file(file, 'space')

        if rel_label:
            string = 'relative'
            axis_label = 'Relative Intensity [%]'
            rel_current = current / np.max(current)
            error_ratio = current_std / current
            index_of_max = np.argmax(current)
            delta_rel_current = rel_current * np.sqrt(error_ratio ** 2 + error_ratio[index_of_max] ** 2)

            rel_current *= np.array(100)
            delta_rel_current *= np.array(np.abs(100))

            current= rel_current
            current_std = delta_rel_current
        else:
            string = 'absolute'
            axis_label = 'Intensity [nA]'


        plt.plot(timestamp, current, 'k-', linewidth='0.5', label=plot_name, color=xkcd_colors[i])
        plt.fill_between(timestamp, current - current_std, current + current_std, color=xkcd_colors[i])

        del x, y, current, current_std, timestamp

    plt.ylabel(axis_label)
    plt.legend(bbox_to_anchor=(0, 1.10, 1, 0.2), loc="lower left", mode='expand', ncol=4, fontsize=10)
    plt.xlabel('Time')
    plt.title('Intensity in time : {}'.format(folder))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.savefig('./Output/{}/Current_in_Time_runs.png'.format(folder), bbox_inches='tight')
    plt.show()
    plt.clf()


def plot_projection_interpolation(xo, xf, current, current_std, steps, path_to_file, rel_label, which_axis):
    # xo                     : beginning of X-axis (or Y-axis)
    # xf                     : end of X-axis (or Y-axis)
    # steps                  : number of measurements in along the axis X and Y (it should be equal in both axis)
    # path_to_file           : path to the file as string type (usually a element of a file_list)
    # rel_label              : Boolean (True : compute relative intensity wrt the maximum -  False : just the absolute intensity)
    # which_axis             : Scanned axis, only two possible values: 'x' or 'y'

    radius = np.array(1.02) # m
    I_max = np.max(current) # nA
    list_of_off = []

    # folder and plot_name came from one component of a file list which is a string type, divided into 2 strings
    folder = path_to_file.split("/")[0]
    filename = path_to_file.split("/")[1]
    plot_name = filename.split('.')[0]

    if rel_label:
        string = 'proj_inter_relative'
        y_axis_label = 'Relative Intensity [%]'

        rel_current = current / np.max(current)
        error_ratio = current_std / current

        index_of_max = np.argmax(current)
        delta_rel_current = rel_current * np.sqrt(error_ratio ** 2 + error_ratio[index_of_max] ** 2)

        matrix = (np.reshape(rel_current, (steps, steps))).T * 100
        matrix_std = (np.reshape(delta_rel_current, (steps, steps))).T * np.abs(100)

    else:
        string = 'proj_inter_absolute'
        y_axis_label = 'Intensity [nA]'

        matrix = (np.reshape(current, (steps, steps))).T
        matrix_std = (np.reshape(current_std, (steps, steps))).T

    if which_axis == 'y':
        matrix = matrix
        matrix_std = matrix_std
    if which_axis == 'x':
        matrix = matrix.T
        matrix_std = matrix_std.T


    distance = (np.linspace(xo, xf, steps) / 1000)

    for i, position in enumerate(distance):
        # Fit function
        # Irradiance ~ cos^3 (angle)
        fitfunc = lambda p, x: (I_max / radius ** 2) * np.cos(p[0]*(np.arctan(x/radius - p[1]/radius)))**3 + p[2]
        # Distance to the target function
        errfunc = lambda p, x, y: fitfunc(p, x) - y

        # Initial guess for the parameters
        data_points = matrix[:, i]
        data_errors = matrix_std[:, i]

        initial_params = [np.pi , distance[np.argmax(data_points)], 2]
        params, success = optimize.leastsq(errfunc, initial_params[:], args=(distance, data_points))
        (freq, distance_off, shift) = params

        list_of_off.append(distance_off)


        # bigger list of angles to use in the interpolated function
        new_angles = np.linspace(distance.min(), distance.max(), 100)
        data_fitted = fitfunc(params, new_angles)
        fit_maximum = np.max(data_fitted)

        angles_fit = np.linspace(distance.min(), distance.max(), 100) - np.array(distance_off)
        angles_fit = np.arctan(angles_fit / radius)
        angles_fit = np.degrees(angles_fit)

        angles_data = np.arctan((distance - np.array(distance_off)) / radius)
        angles_data = np.degrees(angles_data)

        #print(params)

        fig, ay1 = plt.subplots()

        label = '$\dfrac{I_{max}}{D_{flasher}^2}\cos^3\Theta $'

        x_axis_label = 'Scan position [mm]'
        axis_color = 'tab:blue'
        ay1.set_ylabel(y_axis_label)
        ay1.set_xlabel(x_axis_label, color=axis_color)
        ay1.tick_params(axis='x', labelcolor=axis_color)

        ay1.plot(distance*1000, data_points, label='Data')
        ay1.fill_between(distance*1000, data_points - data_errors, data_points + data_errors, color=sns.xkcd_rgb['yellow orange'])
        ay1.xaxis.set_ticks(distance*1000)
        ay1.plot()

        #for i, txt in enumerate(data_points):
        #    ay1.annotate(np.around(txt, decimals=2), (distance[i]*1000, data_points[i]))


        ay2 = ay1.twiny()
        x_axis_label = '$\Theta$ [$^\circ$]'
        axis_color = 'tab:red'
        ay2.set_ylabel(y_axis_label)
        ay2.set_xlabel(x_axis_label, color=axis_color)
        ay2.tick_params(axis='x', labelcolor=axis_color)
        ay2.plot(angles_data, data_points, label='Data')
        ay2.plot(angles_fit, data_fitted, 'r--', label=label)

        plt.legend(bbox_to_anchor=(0, 0.0, 0.5, 0.20), loc="lower left", mode='expand', ncol=1, fontsize=12)

        if which_axis == 'x':
            at_string = 'y'
            plt.text(0.60, 0.80, """
            $x_{flasher}$ : %.1f mm
            $y$ : %.1f mm
            """ % (distance_off*1000, position*1000),
                     fontsize=10, color='black', horizontalalignment='left', verticalalignment='bottom', transform=ay1.transAxes)
        if which_axis == 'y':
            at_string = 'x'
            plt.text(0.60, 0.80, """
            $y_{flasher}$ : %.1f mm
            $x$ : %.1f mm
            """ % (distance_off * 1000, position*1000),
                     fontsize=10, color='black', horizontalalignment='left', verticalalignment='bottom', transform=ay1.transAxes)


        figure_name = './Output/{}/{}_{}_{}_at_{}.png'.format(folder, plot_name, string, at_string, distance[i]*1000)
        plt.savefig(figure_name)
        plt.show()
        plt.clf()
    list_of_off = np.array(list_of_off) * np.array(1000)
    mean_of_off = np.average(list_of_off)
    return mean_of_off
