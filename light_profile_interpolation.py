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


pixel_sw_id , xpix, ypix = load_camera_config()

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

x_array = np.unique(xpix)
y_array = np.unique(ypix)

x_length = np.mean(np.diff(x_array))
y_length = np.mean(np.diff(y_array))

z_array = np.zeros((len(x_array), len(y_array)))

xx, yy = np.meshgrid(x_array, y_array)

#pdf = matplotlib.backends.backend_pdf.PdfPages('/Users/lonewolf/Desktop/interpolation_light_profile.pdf')
#pdf.close()