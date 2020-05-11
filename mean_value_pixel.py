
from readout import read_file
from mydoplots import plot_intensity_scan_xy_2D

import numpy as np
import matplotlib.pyplot as plt
import computations as comp
import matplotlib.backends.backend_pdf

from digicampipe.instrument.camera import DigiCam
from ctapipe.visualization import CameraDisplay
from ctapipe.instrument.camera import CameraGeometry
from digicampipe.visualization.plot import plot_array_camera
from scipy import interpolate
import scipy as scp
import scipy.ndimage

measured_distance = 5.6

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


def plot_window_trans_perpix(fname, label):
    """ IMPORTANT
    display mean value on the camera
    Plot of the mean value of trans_norm_cher per pixel
    :param window_number: 1 or 2 to choose between the first or second window
    :return: plot the camera pixels with the mean value of trans_norm_cher for each of the pixels
    """
    ave_irradiance_per_pixel = np.loadtxt(fname=fname,
    #ave_irradiance_per_pixel = np.loadtxt(fname="/Users/lonewolf/PyCharmProjects/Flasher/Output/ave_irradiance_pixel_flasher.txt",
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
                  cmap='viridis', title='').add_colorbar(label=label)
    # plt.annotate("mean: {0:.2f}\%".format(np.mean(ave_irradiance_per_pixel)*100.),
    #              xy=(-430, 490),
    #              xycoords="data",
    #              va="center",
    #              ha="center",
    #              bbox=dict(boxstyle="round", fc="w", ec="silver"))

    plt.xlim(-550, 550)
    plt.ylim(-550, 550)

    return fig, ax




file_2019_04_27 = ['2019_04_27/Scan_Flasher00.txt',
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

current_values = []
current_errors = []


illuminated_surface = 0.03*0.03 # m^2
photo_sensitivity = 0.2 # A/W
short_distance = 1.0 # m
long_distance = 5.3 # m
scaling = 1

detector_size = [30, 30]
#xx = np.array([-450., 450.])
#yy = np.array([-600., 600.])
xx = np.array([-437, 463])
yy = np.array([-681.7, 518.3])
#x_ticks = [150, 180, 210]
#y_ticks = [30, 60, 90]
extent = [-detector_size[0]/2 + xx.min(), detector_size[0]/2 + xx.max(), -detector_size[1]/2 + yy.min(), detector_size[1]/2 + yy.max()]
steps = 11


for i, file in enumerate(file_2019_04_27):
    x, y, current, current_std, current_time, current_timestamp = read_file(path_to_file=file,
                                                                            scan_type='space',
                                                                            initial_temperature_time=None)
    syst_err = comp.calculate_statistic_error(current_std)
    stat_err = comp.calculate_systematic_error(current)

    total_err = np.sqrt(stat_err**2 + syst_err**2)

    # finding the max. current and its error
    index = np.argmax(current)
    max_current = current[index]
    max_current_error = total_err[0][index]
    #print('{} ± {}'.format(max_current, max_current_error))

    current_values.append(max_current)
    current_errors.append(max_current_error)

current_values = np.array(current_values)
current_errors = np.array(current_errors)

current_val = np.mean(current_values)
current_err = np.sqrt(np.sum(current_errors**2)) / len(current_values)

# Get Intensity [W/sr] from short distance and use it for long distance
#intensity = current_val / (photo_sensitivity * illuminated_surface)
intensity = current_val

print(intensity)

camera_height = 1200 # 1201 mm 1600
camera_width = 900 # 1081 mm 1600
translation_height = 300
translation_width = 300

x_dist = camera_width
y_dist = camera_height

pdf = matplotlib.backends.backend_pdf.PdfPages("/Users/lonewolf/Desktop/extrapolation.pdf")

#x_dist = translation_width
#y_dist = translation_height
#flasher_distance = 1000 #actually 1020

#xo = yo = - x_dist/2
#xf = yf = + y_dist/2
xo = xx[0]
xf = xx[1]
yo = yy[0]
yf = yy[1]
npoints = 101 # odd number so we could be have a bin at (0,0)

x = np.linspace(xo, xf, npoints)
y = np.linspace(yo, yf, npoints)

distance = long_distance * 1000 # to transform from m to mm
frequency = np.pi
offset = 0.0

xx, yy = np.meshgrid(x, y)

data = 1e6 * cartesian_irradiance(x=xx-202.9,
                            y=yy-46.19,
                            intensity=intensity,
                            distance=distance,
                            frequency=frequency,
                            offset=offset)
fig = plt.figure()
ax, ax_cl, fig = plot_intensity_scan_xy_2D(data=data,
                                      x_limits=[xo, xf],
                                      y_limits=[yo, yf],
                                      data_label=False,
                                      scaling=scaling,
                                      axes=None)
ax_cl.set_label('I [A]')
ax.set_title('Extrapolated data at {} m'.format(str(measured_distance)))
ax.set_xlabel('Horizontal distance [mm]')
ax.set_ylabel('Vertical distance [mm]')
pdf.savefig(fig)


#extract_irradiance_perpix(x=xx, y=yy, z=np.flipud(data), fname='/Users/lonewolf/Desktop/extrapolation.txt')
fig, ax = plot_window_trans_perpix(fname='/Users/lonewolf/Desktop/extrapolation.txt', label='I [A]')
ax.set_title('From extrapolated data at {} m'.format(str(measured_distance)))
ax.set_xlabel('Horizontal distance [mm]')
ax.set_ylabel('Vertical distance [mm]')
pdf.savefig(fig)


my_data = np.loadtxt(fname='/Users/lonewolf/Desktop/ave_vals.txt', usecols=1, skiprows=1, unpack=True)
my_extrapolation = np.loadtxt(fname='/Users/lonewolf/Desktop/extrapolation.txt', usecols=1, skiprows=1, unpack=True)

porcentage_diff = 100 * (my_data - my_extrapolation) / my_data

path = "/Users/lonewolf/Desktop/my_porcentage.txt"
output = open(path,"w")
output.write("pixel_id\t porcentage\n")
for k in range(len(pixid)):
    output.write("{0}\t {1}\n".format(pixid[k], porcentage_diff[k]))
output.close()



fig, ax = plot_window_trans_perpix(fname='/Users/lonewolf/Desktop/my_porcentage.txt', label='(Data - Extrapolation) / Data [%]')
ax.set_xlabel('Horizontal distance [mm]')
ax.set_ylabel('Vertical distance [mm]')
ax.set_title('Difference of data and extrapolation at {} m'.format(str(measured_distance)))
pdf.savefig(fig)
pdf.close()


