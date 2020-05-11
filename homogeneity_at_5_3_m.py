#! /anaconda3/envs/Flasher/bin/python

from readout import read_file
import numpy as np
import matplotlib.pyplot as plt
import mydoplots as mydp
from readout import read_pindiode_photo_sensitivity
from scipy.interpolate import spline
from scipy.interpolate import interp2d
from computations import calculate_statistic_error
from computations import calculate_systematic_error
import matplotlib.backends.backend_pdf

matplotlib.rcParams['axes.formatter.limits'] = [-5, 5]

# First matrix

flash = ['flash/flash_00.txt', 'flash/flash_01.txt', 'flash/flash_02.txt', 'flash/flash_03.txt',
         'flash/flash_04.txt', 'flash/flash_05.txt', 'flash/flash_06.txt', 'flash/flash_07.txt',
         'flash/flash_08.txt', 'flash/flash_09.txt', 'flash/flash_10.txt', 'flash/flash_11.txt']

dark = ['flash/dark_00.txt', 'flash/dark_01.txt', 'flash/dark_02.txt', 'flash/dark_03.txt',
        'flash/dark_04.txt', 'flash/dark_05.txt', 'flash/dark_06.txt', 'flash/dark_07.txt',
        'flash/dark_08.txt', 'flash/dark_09.txt', 'flash/dark_10.txt', 'flash/dark_11.txt']

output_dir = '/Users/lonewolf/PyCharmProjects/Flasher/Output/flasher_irradiance_at_5_3_m/'

flash_array = []
flash_err_array = []

dark_array = []
dark_err_array = []

result_array = []
result_err_array = []

relative_array = []
relative_err_array = []

# OPTIONS :
illuminated_area = 0.028**2 # m^2
measured_distance = 5.3
distance = 1.0 # m
detector_size = [30, 30]
#x_surface_limits = np.array([-437, 463]) # x surface limits
#y_surface_limits = np.array([-681.7, 518.3]) # y surface limits
x_surface_limits = np.array([-870/2, 870/2]) # x surface limits
y_surface_limits = np.array([-990/2, 990/2]) # y surface limits

extent = [-detector_size[0]/2 + x_surface_limits.min(),
          detector_size[0]/2 + x_surface_limits.max(),
          -detector_size[1]/2 + y_surface_limits.min(),
          detector_size[1]/2 + y_surface_limits.max()]

steps = 11

# Getting the desired photo-sensitivity for the peak wavelength (1st approximation)
ph_wavelength, photosensitivity = read_pindiode_photo_sensitivity()
ph_wavelength_smooth = np.linspace(ph_wavelength.min(), ph_wavelength.max(), 1000)
photosensitivity_smooth = spline(ph_wavelength, photosensitivity, ph_wavelength_smooth)
pin_diode_wavelength = ph_wavelength_smooth[68]
pin_diode_pde = photosensitivity_smooth[68]

for cnt in range(0, 12):

    x, y, flash_current, flash_current_std, flash_current_time, flash_current_timestamp = read_file(flash[cnt], 'space', initial_temperature_time=None)
    flash_stat_err = calculate_statistic_error(flash_current_std)
    flash_sys_err = calculate_systematic_error(flash_current)
    flash_current_err = np.sqrt(flash_stat_err ** 2 + flash_sys_err ** 2)

    flash_irradiance = flash_current / (pin_diode_pde * illuminated_area)
    flash_irradiance_err = flash_current_err / np.abs(pin_diode_pde * illuminated_area)

    x, y, dark_current, dark_current_std, dark_current_time, dark_current_timestamp = read_file(dark[cnt], 'space', initial_temperature_time=None)
    dark_stat_err = calculate_statistic_error(dark_current_std)
    dark_sys_err = calculate_systematic_error(dark_current)
    dark_current_err = np.sqrt(dark_stat_err ** 2 + dark_sys_err ** 2)

    dark_irradiance = dark_current / (pin_diode_pde * illuminated_area)
    dark_irradiance_err = dark_current_err / np.abs(pin_diode_pde * illuminated_area)

    result = flash_irradiance - dark_irradiance
    result_err = np.sqrt(flash_irradiance_err ** 2 + dark_irradiance_err ** 2)

    # Transforming unidimensional arrays in matrices
    flash_irradiance = flash_irradiance.reshape((steps, steps)).T
    flash_irradiance_err = flash_irradiance_err.reshape((steps, steps)).T
    dark_irradiance = dark_irradiance.reshape((steps, steps)).T
    dark_irradiance_err = dark_irradiance_err.reshape((steps, steps)).T
    result = result.reshape((steps, steps)).T
    result_err = result_err.reshape((steps, steps)).T

    flash_array.append(flash_irradiance)
    flash_err_array.append(flash_irradiance_err)
    dark_array.append(dark_irradiance)
    dark_err_array.append(dark_irradiance_err)
    result_array.append(result)
    result_err_array.append(result_err)

    del flash_current, flash_current_std, flash_current_time, flash_current_timestamp
    del dark_current, dark_current_std, dark_current_time, dark_current_timestamp
    del flash_stat_err, flash_sys_err, flash_current_err, flash_irradiance, flash_irradiance_err
    del dark_stat_err, dark_sys_err, dark_current_err, dark_irradiance, dark_irradiance_err
    del result, result_err, x, y

analysed_matrices = [flash_array, flash_err_array, dark_array, dark_err_array, result_array, result_err_array]
matrices_names = ["flash", "flash errors", "dark current", "dark current error", "flash subtracted", "flash subtracted error"]
matrix = []

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

    # Deleting last 7 rows and 1 column to match more or less camera size
    a_matrix = np.delete(a_matrix, [30], axis=1)
    a_matrix = np.delete(a_matrix, [34, 35, 36, 37, 38, 39, 40], axis=0)
    matrix.append(a_matrix)

    del row_00, row_01, row_02, row_03, column_a, column_b, row_a, row_b, row_c


relative_matrix = 100 * matrix[4] / np.max(matrix[4])
matrices_names.append('relative_wrt_max')
matrix.append(relative_matrix)

np.savetxt('relative.out', relative_matrix, delimiter=' ')

for i, a_matrix in enumerate(matrix):
    print('For matrix : {}'.format(matrices_names[i]))
    max_indices = np.argwhere(a_matrix == np.max(a_matrix))[0]
    print('maximum at {}'.format(max_indices))

pdf = matplotlib.backends.backend_pdf.PdfPages(output_dir + 'irradiance_at_5_3_m.pdf')

for i, a_matrix in enumerate(matrix):
    fig = plt.figure()
    plt.imshow(a_matrix, extent=extent)
    #plt.title(matrices_names[i] + ' at {} m'.format(str(measured_distance)))
    plt.xlabel('Horizontal distance [mm]')
    plt.ylabel('Vertical distance [mm]')
    if matrices_names[i] == 'flash relative':
        label = '$E_{x,y} / E_{max}$ [%]'
    else:
        label = r'E [$w/m^2$]'
    plt.colorbar(label=label)
    pdf.savefig(fig)
    plt.savefig(output_dir + '{}.png'.format(matrices_names[i]), bbox_inches='tight')

fig, ax = plt.subplots()
im = ax.imshow(matrix[-1], extent=extent)
label='$E_{x,y} / E_{max}$ [%]'
colorbar = plt.colorbar(im, label=label)
ax = mydp.plot_intensity_contour(relative_matrix,
                                 x_limits=x_surface_limits,
                                 y_limits=y_surface_limits,
                                 data_label=False,
                                 contour_levels=[80., 90.],
                                 axes=ax)
#ax.set_title('flash relative'.format(str(measured_distance)))
ax.set_xlabel('Horizontal distance [mm]')
ax.set_ylabel('Vertical distance [mm]')
pdf.savefig(fig)
plt.savefig(output_dir + 'relative_wrt_max_contour.png', bbox_inches='tight')


# Make interpolation of the data array
interpolated_flash = matrix[4]

x = np.linspace(x_surface_limits[0], x_surface_limits[1], 30)
y = np.linspace(y_surface_limits[0], y_surface_limits[1], 34)

f_z = interp2d(x, y, interpolated_flash, kind='linear')
x_intp = np.linspace(x_surface_limits[0], x_surface_limits[1], 101)
y_intp = np.linspace(y_surface_limits[0], y_surface_limits[1], 201)
my_z = f_z(x_intp, y_intp)

fig, ax = plt.subplots()
im = ax.imshow(my_z, extent=extent)
label = r'E [$w/m^2$]'
colorbar = plt.colorbar(im, label=label)
#ax.set_title('Interpolated data at {} m'.format(str(measured_distance)))
ax.set_xlabel('Horizontal distance [mm]')
ax.set_ylabel('Vertical distance [mm]')
pdf.savefig(fig)
plt.savefig(output_dir + 'flash_interpolation.png', bbox_inches='tight')


interpolated_relative = 100 * my_z / np.max(my_z)
fig, ax = plt.subplots()
im = ax.imshow(interpolated_relative, extent=extent)
label='$E_{x,y} / E_{max}$ [%]'
colorbar = plt.colorbar(im, label=label)
ax = mydp.plot_intensity_contour(interpolated_relative,
                                 x_limits=x_surface_limits,
                                 y_limits=y_surface_limits,
                                 data_label=False,
                                 contour_levels=[80., 90.],
                                 axes=ax)
#ax.set_title('Interpolated relative data at {} m'.format(str(measured_distance)))
ax.set_xlabel('Horizontal distance [mm]')
ax.set_ylabel('Vertical distance [mm]')
pdf.savefig(fig)
plt.savefig(output_dir + 'flash_interpolation_contour.png', bbox_inches='tight')


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
                  cmap='viridis', title='').add_colorbar(label='$E_{x,y} / E_{max}$ [%]')
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

#extract_irradiance_perpix(x=my_x, y=my_y, z=interpolated_relative, fname=output_dir + 'average_relative_irradiance_per_pixel.txt')
fig, ax = plot_window_trans_perpix(fname=output_dir + 'average_relative_irradiance_per_pixel.txt')
#ax.set_title('Average Irradiance per pixel at {} m'.format(str(measured_distance)))

ax = mydp.plot_intensity_contour(interpolated_relative,
                                  x_limits=x_surface_limits,
                                  y_limits=y_surface_limits,
                                  data_label=False,
                                  contour_levels=[80., 90.],
                                  axes=ax)

ax.set_ylabel('Vertical distance [mm]')
ax.set_xlabel('Horizontal distance [mm]')
pdf.savefig(fig)
plt.savefig(output_dir + 'average_relative_irradiance_per_pixel.png', bbox_inches='tight')

pdf.close()

