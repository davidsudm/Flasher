#! /anaconda3/envs/Flasher/bin/python

from readout import read_file
from readout import read_cerenkov_spectra
from readout import read_pindiode_photo_sensitivity
from readout import read_pindiode_temperature_coefficient
from readout import read_led_spectra
from scipy import interpolate
from xkcd_colors import xkcd_colors

import doplots as dp
import numpy as np
import computations as comp
import matplotlib.pyplot as plt
import mydoplots as mydp
import datetime as dt
import matplotlib.dates as md
import declaration
import seaborn as sns

#import surface_homogeneity as homo
#import thermal_coefficients as thermal
#import specifications as spec
#import profile_interpolation as profit defective <--
#import mean_value_pixel as pixel

from computations import calculate_statistic_error
from computations import calculate_systematic_error
import matplotlib.backends.backend_pdf


# First matrix

flash = ['flash/flash_00.txt', 'flash/flash_01.txt', 'flash/flash_02.txt', 'flash/flash_03.txt',
         'flash/flash_04.txt', 'flash/flash_05.txt', 'flash/flash_06.txt', 'flash/flash_07.txt',
         'flash/flash_08.txt', 'flash/flash_09.txt', 'flash/flash_10.txt', 'flash/flash_11.txt']

dark = ['flash/dark_00.txt', 'flash/dark_01.txt', 'flash/dark_02.txt', 'flash/dark_03.txt',
        'flash/dark_04.txt', 'flash/dark_05.txt', 'flash/dark_06.txt', 'flash/dark_07.txt',
        'flash/dark_08.txt', 'flash/dark_09.txt', 'flash/dark_10.txt', 'flash/dark_11.txt']

flash_array = []
flash_err_array = []

dark_array = []
dark_err_array = []

result_array = []
result_err_array = []

relative_array = []
relative_err_array = []



measured_distance = 5.6

detector_size = [30, 30]
#xx = np.array([-450., 450.])
#yy = np.array([-600., 600.])
xx = np.array([-437, 463])
yy = np.array([-681.7, 518.3])
#x_ticks = [150, 180, 210]
#y_ticks = [30, 60, 90]
extent = [-detector_size[0]/2 + xx.min(), detector_size[0]/2 + xx.max(), -detector_size[1]/2 + yy.min(), detector_size[1]/2 + yy.max()]
steps = 11

for cnt in range(0, 12):

    x, y, flash_current, flash_current_std, flash_current_time, flash_current_timestamp = read_file(flash[cnt], 'space', initial_temperature_time=None)
    flash_stat_err = calculate_statistic_error(flash_current_std)
    flash_sys_err = calculate_systematic_error(flash_current)
    flash_current_err = np.sqrt(flash_stat_err ** 2 + flash_sys_err ** 2)

    x, y, dark_current, dark_current_std, dark_current_time, dark_current_timestamp = read_file(dark[cnt], 'space', initial_temperature_time=None)
    dark_stat_err = calculate_statistic_error(dark_current_std)
    dark_sys_err = calculate_systematic_error(dark_current)
    dark_current_err = np.sqrt(dark_stat_err ** 2 + dark_sys_err ** 2)

    result = flash_current - dark_current
    result_err = np.sqrt(flash_current_err ** 2 + dark_current_err ** 2)

    flash_current = flash_current.reshape((steps, steps)).T
    flash_current_err = flash_current_err.reshape((steps, steps)).T

    dark_current = dark_current.reshape((steps, steps)).T
    dark_current_err = dark_current_err.reshape((steps, steps)).T

    result = result.reshape((steps, steps)).T
    result_err = result_err.reshape((steps, steps)).T

    flash_array.append(flash_current)
    flash_err_array.append(flash_current_err)
    dark_array.append(dark_current)
    dark_err_array.append(dark_current_err)
    result_array.append(result)
    result_err_array.append(result_err)

    del x, y, flash_current_timestamp, flash_current_time, dark_current_time, dark_current_timestamp, dark_stat_err, dark_sys_err, flash_stat_err, flash_sys_err
    del flash_current, flash_current_err, dark_current, dark_current_err, result, result_err

relative_array = result_array/np.max(result_array)
analysed_matrices = [flash_array, flash_err_array, dark_array, dark_err_array, result_array, result_err_array, relative_array]
matrices_names = ["flash", "flash errors", "dark current", "dark current error", "flash subtracted", "flash subtracted error", "flash relative"]
matrix = []

pdf = matplotlib.backends.backend_pdf.PdfPages("/Users/lonewolf/Desktop/camera_surface_scan.pdf")

for i, a_matrix in enumerate(analysed_matrices):

    row_00 = np.hstack((a_matrix[2], a_matrix[1], a_matrix[0]))
    row_01 = np.hstack((a_matrix[5], a_matrix[4], a_matrix[3]))
    row_02 = np.hstack((a_matrix[8], a_matrix[7], a_matrix[6]))
    row_03 = np.hstack((a_matrix[11], a_matrix[10], a_matrix[9]))

    a_matrix = np.vstack((row_03, row_02, row_01, row_00))

    column_a = (a_matrix[:, 10] + a_matrix[:, 11]) / 2
    column_b = (a_matrix[:, 21] + a_matrix[:, 22]) / 2

    row_a = (a_matrix[10, :] + a_matrix[11, :]) / 2
    row_b = (a_matrix[21, :] + a_matrix[22, :]) / 2
    row_c = (a_matrix[32, :] + a_matrix[33, :]) / 2

    a_matrix[10, :] = row_a
    a_matrix[21, :] = row_b
    a_matrix[32, :] = row_c

    a_matrix[:, 10] = column_a
    a_matrix[:, 21] = column_b

    a_matrix = np.delete(a_matrix, [11, 22, 33], axis=0)
    a_matrix = np.delete(a_matrix, [11, 22], axis=1)

    a_matrix = np.fliplr(np.flipud(a_matrix))

    # Deleting last 7 rows and 1 column to much more or less camera size
    #a_matrix = np.delete(a_matrix, [0, 1, 2, 3, 4, 5, 6], axis=0)
    #a_matrix = np.delete(a_matrix, [30], axis=1)
    matrix.append(a_matrix)

    if matrices_names[i] == "flash subtracted":
        print('maximum for {}'.format(matrices_names[i]))
        print(np.argwhere(a_matrix == np.max(a_matrix))[0])

    del row_00, row_01, row_02, row_03, column_a, column_b, row_a, row_b, row_c

    fig = plt.figure()
    plt.imshow(a_matrix, extent=extent)
    plt.title(matrices_names[i] + ' at {} m'.format(str(measured_distance)))
    plt.xlabel('Horizontal distance [mm]')
    plt.ylabel('Vertical distance [mm]')
    plt.colorbar(label='current [A]')
    pdf.savefig(fig)

pdf.close()

pdf = matplotlib.backends.backend_pdf.PdfPages("/Users/lonewolf/Desktop/camera_surface_scan_relative.pdf")
# Relative wrt to the maximum
relative_matrix = 100 * matrix[4] / np.max(matrix[4])

max_indices = np.argwhere(relative_matrix == np.max(relative_matrix))[0]
print('maximum at {}'.format(max_indices))

fig, ax = plt.subplots()
im = ax.imshow(relative_matrix, extent=extent)
label='$I_{i,j} / I_{max}$ [%]'
colorbar = plt.colorbar(im, label=label)
ax = mydp.plot_intensity_contour(relative_matrix, x_limits=xx, y_limits=yy, data_label=False, axes=ax)
ax.set_title('Raw Data at {} m : Relative current wrt. maximum'.format(str(measured_distance)))
ax.set_xlabel('Horizontal distance [mm]')
ax.set_ylabel('Vertical distance [mm]')
pdf.savefig(fig)

from scipy.interpolate import interp2d
x = np.linspace(xx[0], xx[1], 31)
y = np.linspace(yy[0], yy[1], 41)
f_z = interp2d(x, y, relative_matrix, kind='cubic')
x_intp = np.linspace(xx[0], xx[1], 901)
y_intp = np.linspace(yy[0], yy[1], 1201)
my_z = f_z(x_intp, y_intp)

fig, ax = plt.subplots()
im = ax.imshow(my_z, extent=extent)
label='$I_{i,j} / I_{max}$ [%]'
colorbar = plt.colorbar(im, label=label)
ax = mydp.plot_intensity_contour(my_z, x_limits=xx, y_limits=yy, data_label=False, axes=ax)
ax.set_title('Interpolated data at {} m : Relative current wrt. maximum'.format(str(measured_distance)))
ax.set_xlabel('Horizontal distance [mm]')
ax.set_ylabel('Vertical distance [mm]')
#ax.xaxis.set_ticks(x_ticks)
#ax.yaxis.set_ticks(y_ticks)
pdf.savefig(fig)



pdf.close()

# Make interpolation of the data array
flash_matrix = matrix[4]
pdf = matplotlib.backends.backend_pdf.PdfPages("/Users/lonewolf/Desktop/interpolation.pdf")

x = np.linspace(xx[0], xx[1], 31)
y = np.linspace(yy[0], yy[1], 41)

f_z = interp2d(x, y, flash_matrix, kind='cubic')
x_intp = np.linspace(xx[0], xx[1], 101)
y_intp = np.linspace(yy[0], yy[1], 201)
my_z = f_z(x_intp, y_intp)

fig, ax = plt.subplots()
im = ax.imshow(my_z, extent=extent)
label='I [A]'
colorbar = plt.colorbar(im, label=label)
ax.set_title('Interpolated data at {} m'.format(str(measured_distance)))
ax.set_xlabel('Horizontal distance [mm]')
ax.set_ylabel('Vertical distance [mm]')
pdf.savefig(fig)



from digicampipe.instrument.camera import DigiCam
from ctapipe.visualization import CameraDisplay

#loading pixels id and coordinates from camera config file
pixid, x_pix, y_pix = np.loadtxt(fname='/Users/lonewolf/PyCharmProjects/Flasher/camera_config.cfg.txt',
                                 unpack=True,
                                 skiprows=47,
                                 usecols=(8, 9, 10))

pixid, x_pix, y_pix = zip(*sorted(zip(pixid, x_pix, y_pix)))
pixid=list(pixid)
x_pix=list(x_pix)
y_pix=list(y_pix)
pixid=[int(x) for x in pixid]


def cartesian_irradiance(x, y, intensity, distance, frequency, offset):
    """"""
    r = np.sqrt(x**2 + y**2)
    angle = np.arctan(r / distance)
    return (intensity/distance**2) * np.cos(frequency * (angle - offset))**3


def hexagon(pos, pixpos):
    """ IMPORTANT
    :param pos: (x,y) coordinates of the point
    :param pixpos: (x,y) coordinates of the pixel
    :return: True if the point is inside the hexagon, False if the point is outside the hexagon
    """
    b = list(pixpos)
    b[0] = -b[0]
    pixpos = tuple(b)
    s = 13.4
    x, y = map(abs, tuple(map(sum, zip(pos, pixpos))))
    return x < 3**0.5 * min(s-y, s/2)


def extract_irradiance_perpix(x, y, z, fname):
    """ IMPORTANT
    extraction value for each pixel
    Extract mean value per pixel of trans_norm_cher
    :param window_number: 1 or 2 to choose between the first or second window
    :return: text file with mean trans_norm_cher values per pixel
    """

    ave_irradiance_pixel = []
    for k in range(len(pixid)):
        pix_values = []
        for i in range(len(x)):
            for j in range(len(x[0])):
                if hexagon((x[i][j], y[i][j]), (x_pix[k], y_pix[k])):
                    pix_values.append(z[i][j])
        ave_irradiance_pixel.append(np.mean(pix_values))
        print("pixel {0}: {1}, {2}".format(pixid[k], ave_irradiance_pixel[k], len(pix_values)))

    #path = "/Users/lonewolf/PyCharmProjects/Flasher/Output/ave_irradiance_pixel_flasher.txt"
    path = fname
    output = open(path, "w")
    output.write("pixel_id\t irradiance\n")
    for k in range(len(pixid)):
        output.write("{0}\t {1}\n".format(pixid[k], ave_irradiance_pixel[k]))
    output.close()


def plot_window_trans_perpix(fname):
    """ IMPORTANT
    display mean value on the camera
    Plot of the mean value of trans_norm_cher per pixel
    :param window_number: 1 or 2 to choose between the first or second window
    :return: plot the camera pixels with the mean value of trans_norm_cher for each of the pixels
    """
    ave_irradiance_per_pixel = np.loadtxt(fname=fname,
                                          usecols=1,
                                          skiprows=1,
                                          unpack=True)

    fig = plt.figure()
    ax = plt.gca()
    ax.set_alpha(0)
    CameraDisplay(DigiCam.geometry,
                  ave_irradiance_per_pixel,
                  cmap='viridis',
                  title='').highlight_pixels(range(1296), color='k', linewidth=0.2)

    CameraDisplay(DigiCam.geometry,
                  ave_irradiance_per_pixel,
                  cmap='viridis', title='').add_colorbar(label="I [A]")
    # plt.annotate("mean: {0:.2f}\%".format(np.mean(ave_irradiance_per_pixel)*100.),
    #              xy=(-430, 490),
    #              xycoords="data",
    #              va="center",
    #              ha="center",
    #              bbox=dict(boxstyle="round", fc="w", ec="silver"))

    plt.xlim(-550, 550)
    plt.ylim(-550, 550)

    return fig, ax

my_x, my_y = np.meshgrid(x_intp, y_intp)

#extract_irradiance_perpix(x=my_x, y=my_y, z=my_z, fname='/Users/lonewolf/Desktop/ave_vals.txt')
fig, ax = plot_window_trans_perpix(fname='/Users/lonewolf/Desktop/ave_vals.txt')
ax.set_title('From interpolated data at {} m'.format(str(measured_distance)))
ax.set_xlabel('Horizontal distance [mm]')
ax.set_ylabel('Vertical distance [mm]')
pdf.savefig(fig)
pdf.close()

#import mean_value_pixel

