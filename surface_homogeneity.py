#! /anaconda3/envs/Flasher/bin/python

from readout import read_file
from readout import output_dir

import numpy as np
import computations as comp
import matplotlib.pyplot as plt
import mydoplots as mydp


"""
project_dir = './'
data_dir = './Data'
output_dir = './Output'
specification_dir = './Data/Specification'
"""


# Surface scan files:

# '2019_04_24/Scan_Flasher01.txt' excluded since it is dark current
file_list1 = ['2019_04_24/Scan_Flasher02.txt', '2019_04_24/Scan_Flasher03.txt', '2019_04_24/Scan_Flasher04.txt',
              '2019_04_24/Scan_Flasher05.txt', '2019_04_24/Scan_Flasher06.txt']

# '2019_04_25/Scan_Flasher00.txt' excluded since it is dark current
file_list2 = ['2019_04_25/Scan_Flasher01.txt', '2019_04_25/Scan_Flasher02.txt', '2019_04_25/Scan_Flasher03.txt',
              '2019_04_25/Scan_Flasher04.txt', '2019_04_25/Scan_Flasher05.txt', '2019_04_25/Scan_Flasher06.txt',
              '2019_04_25/Scan_Flasher07.txt', '2019_04_25/Scan_Flasher08.txt', '2019_04_25/Scan_Flasher09.txt',
              '2019_04_25/Scan_Flasher10.txt', '2019_04_25/Scan_Flasher11.txt', '2019_04_25/Scan_Flasher12.txt']

file_list3 = ['2019_04_27/Scan_Flasher00.txt', '2019_04_27/Scan_Flasher01.txt', '2019_04_27/Scan_Flasher02.txt',
              '2019_04_27/Scan_Flasher03.txt', '2019_04_27/Scan_Flasher04.txt', '2019_04_27/Scan_Flasher05.txt',
              '2019_04_27/Scan_Flasher06.txt', '2019_04_27/Scan_Flasher07.txt', '2019_04_27/Scan_Flasher08.txt',
              '2019_04_27/Scan_Flasher09.txt', '2019_04_27/Scan_Flasher10.txt']

# OPTIONS :
xo = yo = 0.
xf = yf = 300.
steps = 11
# number_of_cells = steps * steps


def make_surface_homogeneity(path_to_the_file):
    """"""

    """ folder and plot_name came from one component of a file list which is a string type, divided into 2 strings """
    folder = path_to_the_file.split("/")[0]
    filename = path_to_the_file.split("/")[1]
    plot_name = filename.split('.')[0]

    x, y, current, current_std, current_time, current_timestamp = read_file(file, 'space')
    mat_current, mat_current_std = comp.create_data_grid(current, current_std, steps, rel_label=True)

    """ interpolate matrix """
    interpolated_current = comp.interpolate_data_points(mat_current, points_array=[301, 301], interpolation='linear')

    """ make gaussian interpolation from interpolated data """
    params, fitted_gaus = comp.fit_2d_gaussian(interpolated_current, points_array=[300, 300])

    """ plot raw data """
    ax1 = mydp.plot_intensity_scan_xy_2D(mat_current, data_label=False)

    """ plot camera centered at x,y found by gaussian interpolation """
    mydp.draw_camera(axes=ax1, linestyle='-', linewidth=0.5, camera_centre=[params[1], params[2]])

    """ Plot intensity contour from gauss interpolation """
    mydp.plot_intensity_contour(axes=ax1, data=fitted_gaus)

    """ Plot intensity contour from gauss interpolation """
    # mydp.plot_intensity_contour(axes=ax1, data=mat_current)


    ax1.text(0.95, 0.75, """
            $x_{centre}^{camera}$ : %.1f mm
            $y_{centre}^{camera}$ : %.1f mm
            $\sigma_x$ : %.1f mm
            $\sigma_y$ : %.1f mm """ % (params[1], params[2], params[3], params[4]),
             fontsize=10, color='white', horizontalalignment='right',
             verticalalignment='bottom', transform=ax1.transAxes)

    ax1.set_xticks(np.linspace(xo, xf, steps))
    ax1.set_yticks(np.linspace(yo, yf, steps))

    plt.savefig(output_dir+'/Homogeneity/{}/Homogeneity_{}_relative_cnt_gaussian.png'.format(folder, plot_name), bbox_inches='tight')
    # plt.savefig(output_dir+'/{}/Homogeneity_{}_relative_cnt_rawdata.png'.format(folder, plot_name), bbox_inches='tight')


space_Data = [file_list1, file_list2, file_list3]
space_Data = [file_list2]

for file_list in space_Data:
    for file in file_list:
        make_surface_homogeneity(file)

