#! /anaconda3/envs/Flasher/bin/python

from readout import read_file
from readout import output_dir

import os
import numpy as np
import yaml
import fitsio
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec

from digicampipe.visualization.plot import plot_histo, plot_array_camera
import digicampipe.utils.pulse_template as templates
import matplotlib.backends.backend_pdf
from scipy.optimize import curve_fit, minimize


def normalized_irradiance(x, A, B, x_off):

    irradiance = A * np.cos(B*(np.arctan(x/5600) - np.arctan(x_off/5600)))**3

    return irradiance


def radial_irradiance(x, y, A, B, x_off, y_off, D):

    irradiance = A * np.cos(B * np.arctan(np.sqrt((x - x_off)**2 + (y - y_off)**2) / D))**3

    return irradiance


def compute_theta(x, y, x_off, y_off, D):

    theta = np.sqrt((x-x_off)**2 + (y-y_off)**2) / D

    return theta


def chi2(x, y, z, A, B, x_off, y_off, D):

    chi2 = np.sum((z - radial_irradiance(x, y, A, B, x_off, y_off, D))**2)

    return chi2


def load_camera_config():

    # loading pixels id and coordinates from camera config file
    pixel_sw_id , xpix, ypix = np.loadtxt(fname='/Users/lonewolf/ctasof/digicampipe/digicampipe/tests/resources/camera_config.cfg',
                                          unpack=True,
                                          skiprows=47,
                                          usecols=(8, 9, 10))

    pixel_sw_id = pixel_sw_id.astype(int)
    xpix = xpix.astype(float)
    ypix = ypix.astype(float)

    pixel_sw_id, xpix, ypix = zip(*sorted(zip(pixel_sw_id, xpix, ypix)))
    xpix = np.array(xpix)
    ypix = np.array(ypix)
    pixel_sw_id = np.array(pixel_sw_id)

    return pixel_sw_id, xpix, ypix


ave_irradiance_per_pixel = np.loadtxt(fname='/Users/lonewolf/PyCharmProjects/Flasher/Output/flasher_irradiance_at_5_3_m/average_relative_irradiance_per_pixel.txt',
                                      usecols=1,
                                      skiprows=1,
                                      unpack=True)

pdf = matplotlib.backends.backend_pdf.PdfPages('/Users/lonewolf/Desktop/light_profile.pdf')

pixel_sw_id , xpix, ypix = load_camera_config()
detector_size = 30 # mm
x_size = 870 # mm
y_size = 990 # mm
plane_limits = [-x_size, x_size, -y_size, y_size]
extent = [-detector_size/2 + plane_limits[0], detector_size/2 + plane_limits[1],
          -detector_size/2 + plane_limits[2], detector_size/2 + plane_limits[3]]
data = np.loadtxt('relative.out', delimiter=' ')
dimension = data.shape
x = np.linspace(plane_limits[0], plane_limits[1], dimension[1])
y = np.linspace(plane_limits[2], plane_limits[3], dimension[0])
distance = 5.6*1e3


# File with the mean template of all events (53551 events)
pixels_waveforms = '/Users/lonewolf/Desktop/test_output.fits'

with fitsio.FITS(pixels_waveforms, 'r') as file:
    #print(file['PULSE_TEMPLATE'])

    time = file['PULSE_TEMPLATE']['time'].read()
    templates = file['PULSE_TEMPLATE']['original_amplitude'].read()
    templates = templates.T

maximum_array = []
charge_array = []

for pixel in range(0, 1296):

    arg_max = np.argmax(templates[pixel])

    max_in_template = np.max(templates[pixel])
    charge = np.sum((templates[pixel])[arg_max-3:arg_max+4])

    maximum_array.append(max_in_template)
    charge_array.append(charge)

maximum_array = 100 * (maximum_array / np.max(maximum_array))
charge_array = 100 * (charge_array / np.max(charge_array))

shorten_pix_amplitude = []
shorten_pix_charge = []
shorten_amplitude = []
shorten_charge = []

for pixel in range(0, 1296):
    if maximum_array[pixel] > 40:
        shorten_pix_amplitude.append(pixel)
        shorten_amplitude.append(maximum_array[pixel])

    if charge_array[pixel] > 40:
        shorten_pix_charge.append(pixel)
        shorten_charge.append(charge_array[pixel])

shorten_amplitude = 100 * (shorten_amplitude / np.max(shorten_amplitude))
shorten_charge = 100 * (shorten_charge / np.max(shorten_charge))


fig, ax = plt.subplots()
ax.plot(shorten_pix_amplitude, shorten_amplitude, label='from amplitude', color=sns.xkcd_rgb['red'])
ax.plot(shorten_pix_charge, shorten_charge, label='from charge', color=sns.xkcd_rgb['blue'], linestyle='dashed')
ax.plot(ave_irradiance_per_pixel, label='Irradiance per pixel from photo-diode', color=sns.xkcd_rgb['amber'])
ax.set_xlabel('Pixel ID')
ax.set_ylabel('Relative Intensity [%]')
ax.legend()
pdf.savefig(fig)


fig_x, ax = plt.subplots()
ax.plot(xpix[210:245], maximum_array[210:245], label='210-245', marker='.')
ax.plot(xpix[246:281], maximum_array[246:281], label='246-281', marker='.')
ax.plot(xpix[282:317], maximum_array[282:317], label='282-317', marker='.')
ax.plot(xpix[318:353], maximum_array[318:353], label='318-353', marker='.')
ax.set_xlabel('x position [mm]')
ax.set_ylabel('Relative Intensity [%] from amplitude')
ax.legend()
pdf.savefig(fig_x)

fig_x, ax = plt.subplots()
ax.plot(xpix[210:245], charge_array[210:245], label='210-245', marker='.')
ax.plot(xpix[246:281], charge_array[246:281], label='246-281', marker='.')
ax.plot(xpix[282:317], charge_array[282:317], label='282-317', marker='.')
ax.plot(xpix[318:353], charge_array[318:353], label='318-353', marker='.')
ax.set_xlabel('x position [mm]')
ax.set_ylabel('Relative Intensity [%] from charge')
ax.legend()
pdf.savefig(fig_x)


fig_x, ax = plt.subplots()
popt, pcov = curve_fit(normalized_irradiance, np.array(xpix[210:245]), np.array(charge_array[210:245]))
ax.plot(xpix[210:245], charge_array[210:245], label='pixels : 210-245', marker='.')
plt.plot(np.array(xpix[210:245]), normalized_irradiance(np.array(xpix[210:245]), *popt), 'r-', label='fit: A=%5.3f, B=%5.3f, x_offset=%5.3f' % tuple(popt))
ax.set_xlabel('x position [mm]')
ax.set_ylabel('Relative Intensity [%] from charge')
ax.legend()
pdf.savefig(fig_x)

fig_x, ax = plt.subplots()
popt, pcov = curve_fit(normalized_irradiance, np.array(xpix[246:281]), np.array(charge_array[246:281]))
ax.plot(xpix[246:281], charge_array[246:281], label='pixels : 246-281', marker='.')
plt.plot(np.array(xpix[246:281]), normalized_irradiance(np.array(xpix[246:281]), *popt), 'r-', label='fit: A=%5.3f, B=%5.3f, x_offset=%5.3f' % tuple(popt))
ax.set_xlabel('x position [mm]')
ax.set_ylabel('Relative Intensity [%] from charge')
ax.legend()
pdf.savefig(fig_x)

fig_x, ax = plt.subplots()
popt, pcov = curve_fit(normalized_irradiance, np.array(xpix[282:317]), np.array(charge_array[282:317]))
ax.plot(xpix[282:317], charge_array[282:317], label='pixels : 282-317', marker='.')
plt.plot(np.array(xpix[282:317]), normalized_irradiance(np.array(xpix[282:317]), *popt), 'r-', label='fit: A=%5.3f, B=%5.3f, x_offset=%5.3f' % tuple(popt))
ax.set_xlabel('x position [mm]')
ax.set_ylabel('Relative Intensity [%] from charge')
ax.legend()
pdf.savefig(fig_x)

fig_x, ax = plt.subplots()
popt, pcov = curve_fit(normalized_irradiance, np.array(xpix[318:353]), np.array(charge_array[318:353]))
ax.plot(xpix[318:353], charge_array[318:353], label='pixels : 318-353', marker='.')
plt.plot(np.array(xpix[318:353]), normalized_irradiance(np.array(xpix[318:353]), *popt), 'r-', label='fit: A=%5.3f, B=%5.3f, x_offset=%5.3f' % tuple(popt))
ax.set_xlabel('x position [mm]')
ax.set_ylabel('Relative Intensity [%] from charge')
ax.legend()
pdf.savefig(fig_x)



mask = charge_array > 40
func = lambda param: chi2(xpix[mask], ypix[mask], charge_array[mask], param[0], param[1], param[2], param[3], distance)
# Initialization of parameters
i_max = np.argmax(charge_array[mask])
A = charge_array[i_max]
B = 1
x_off = xpix[i_max]
y_off = ypix[i_max]
x0 = [A, B, x_off, y_off]
arg = minimize(func, x0=x0)
print(arg, x0)

# Plot data and fit according wrt theta
theta = compute_theta(xpix[mask], ypix[mask], arg.x[2], arg.x[3], distance)
x_fit = np.linspace(-400, 400, num=1000) + arg.x[2]
y_fit = np.linspace(-400, 400, num=1000) + arg.x[3]
theta_fit = compute_theta(x_fit, y_fit, arg.x[2], arg.x[3], distance)
z_fit = radial_irradiance(x_fit, y_fit, arg.x[0], arg.x[1], arg.x[2], arg.x[3], distance)

fig_fit = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

ax_fit = plt.subplot(gs[0])
ax_fit.plot(np.degrees(theta), charge_array[mask], linestyle='None', marker='*', ms=2, label='data')
ax_fit.plot(np.degrees(theta_fit), z_fit, label=r'fit : I = $A\cos^3(B\theta)$ with'+'\n'+
                                                r'$\theta = \frac{\sqrt{(x - x_{offset})^2 + (y - y_{offset})^2}}{D}$'+'\n'+
                                                'A = {} '.format(np.around(arg.x[0], decimals=3)) + '\n' + 'B = {}'.format(np.around(arg.x[1], decimals=3)) + '\n' + r'$x_{offset}=$'+' {} mm'.format(np.around(arg.x[2], decimals=3)) + '\n' + r'$y_{offset}=$'+' {} mm'.format(np.around(arg.x[3], decimals=3)))
ax_fit.set_ylabel('Relative Intensity [%] from charge')
#fit_params = 'A = {} '.format(np.around(arg.x[0], decimals=3)) + '\n' + 'B = {}'.format(np.around(arg.x[1], decimals=3)) + '\n' + r'$x_{offset}=$'+' {} mm'.format(np.around(arg.x[2], decimals=3)) + '\n' + r'$y_{offset}=$'+' {} mm'.format(np.around(arg.x[3], decimals=3))
#ax_fit.text(50.50, 50.50, fit_params, {'color': 'r', 'fontsize': 20, 'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.4)})
ax_fit.legend(loc=3, fontsize=8)

ax_res = plt.subplot(gs[1], sharex=ax_fit)
# Making residuals
residuals = charge_array[mask] - radial_irradiance(xpix[mask], ypix[mask], arg.x[0], arg.x[1], arg.x[2], arg.x[3], distance)
residuals /= charge_array[mask]
ax_res.plot(np.degrees(theta), residuals, linestyle='None', marker='o', ms=2)
ax_res.set_ylabel('residuals')
ax_res.set_xlabel(r'$\theta$ [°]')


pdf.savefig(fig_fit)

# fig_res, ax_res = plt.subplots()
# ax_res.plot(np.degrees(theta), residuals, linestyle='None', marker='o', ms=2)
# pdf.savefig(fig_res)

cam_display_chg, fig_cam_chg = plot_array_camera(data=np.ma.masked_array(charge_array, mask=(~mask + (charge_array < 40))), label='Intensity [from charge]')
pdf.savefig(fig_cam_chg)
cam_display_amp, fig_cam_amp = plot_array_camera(data=np.ma.masked_array(maximum_array, mask=(~mask + (maximum_array < 40))), label='Intensity [from amplitude]')
pdf.savefig(fig_cam_amp)

pdf.close()
