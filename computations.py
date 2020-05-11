#! /anaconda3/envs/Flasher/bin/python

import numpy as np
import fitfunction as ff
import scipy
from readout import read_pindiode_photo_sensitivity
from scipy import interpolate
from scipy.interpolate import spline
from scipy import optimize


def give_matched_arrays(temperature_time, temperature, current_time, current, current_std, initial_cut=None):
    """
    Use in current scanned for different temperature at a time
    Give back the array of temperature,. current, current_std, and time such as all are synchronized and spam the same
    number of elements
    :param temperature_time:    (float array) time array belonging to temperature array
    :param temperature:         (float array) temperature array
    :param current_time:        (float array) time array belonging to current array
    :param current:             (float array) current array
    :param current_std:         (float array) current_std array, same size as current array
    :param initial_cut:         (int) index where we want to impose a cut in the beginning of the array, if 'None', the
                                array isn't cut and taken from their beginning
    :return:                    4 arrays of the same size beginning in the same point in time, applied an initial cut or
                                not.
    """

    new_time = []
    # Create interpolation function of temperature
    f = interpolate.interp1d(temperature_time, temperature)

    # define the upper boundary of current_time by the length of temperature_time
    max_time_in_temperature = temperature_time[-1]
    gen = (time for time in current_time if time <= max_time_in_temperature)

    for time in gen:
        new_time.append(time)
    new_time = np.array(new_time)

    # from intensity time, create the array of time in seconds
    new_temperature = f(new_time)
    new_current = current[:len(new_time)]
    new_current_std = current_std[:len(new_time)]

    if initial_cut is None:
        initial_cut = 0

    return new_time[initial_cut:], new_current[initial_cut:], new_current_std[initial_cut:], new_temperature[initial_cut:]



def create_data_grid(data, data_errors, steps, rel_label):
    """
    IMPORTANT :  "calculate_errors" should be applied first!
    Use at a "scanned current in a surface" assuming no temperature variation
    This function take a list of data which is in 1-D array or list which is supposed to be in a matrix but isn't, and put them in a matrix corresponding to the scanned surface
    :param data:        (float) list or 1-D array containing the measured data values of each point of the scanned surface
    :param data_errors: (float) list or 1-D array containing the measured data errors values of each point of the scanned surface
    :param steps:       (int) number of measurements per axis, in other way, the dimension of the matrix will be give by (steps, steps)
    :param rel_label:   (bool) True : compute relative intensity wrt the maximum -  False : just the absolute intensity
    :return:            return a 2-D array of current of dimension (steps, steps)
    """

    if rel_label:
        # Relative data wrt the maximum
        matrix = data / np.max(data)
        # Errors over the relative data
        error_ratio = data_errors / data
        index_of_max = np.argmax(data)
        matrix_errors = np.abs(matrix) * np.sqrt(error_ratio ** 2 + error_ratio[index_of_max] ** 2)

        matrix = np.reshape(matrix, (steps, steps)) * np.abs(100)
        matrix_errors = np.reshape(matrix_errors, (steps, steps)) * np.abs(100)

    else:
        matrix = np.reshape(data, (steps, steps))
        matrix_errors = np.reshape(data_errors, (steps, steps))

    return matrix, matrix_errors



def interpolate_data_points(data, x_limits=[0, 300], y_limits=[0, 300], points_array=[11, 11], interpolation='linear'):
    """
    Use in scanned current in a surface assuming no temperature variation
    Interpolates 2-D data points to make the 2-D data in the plane smoother but also useful to fit gaussians in 2_D plane
    :param data_matrix:         (float) list or 1-D array containing the measured data values of each point of the scanned surface
    :param x_limits:            (float) list with the beginning and end of X-axis, i.e. [xo, xf]
    :param y_limits:            (float) list with the beginning and end of Y-axis, i.e. [yo, yf]
    :param number_of_points:    (int) number of points per axis, in other way, the dimension of the matrix
    :return:                    Interpolated data matrix
    """

    dimension = data.shape
    # Make interpolation function the dimensions of the data matrix
    x = np.linspace(x_limits[0], x_limits[1], dimension[0])
    y = np.linspace(y_limits[0], y_limits[1], dimension[1])
    x_mesh, y_mesh = np.meshgrid(x, y)
    f = scipy.interpolate.interp2d(x_mesh, y_mesh, data, kind=interpolation)

    # With the interpolation function, extend the definition to a bigger matrix
    # by default a 301x301, since we want to go from 0 to 300 by steps of 1
    x_new = np.linspace(x_limits[0], x_limits[1], points_array[0])
    y_new = np.linspace(y_limits[0], y_limits[1], points_array[1])
    data_matrix_new = f(x_new, y_new)

    dimension = data_matrix_new.shape
    print('Detector size from interpolation : [{}, {}]'.format(np.around(x_new[1]-x_new[0], decimals=2), np.around(y_new[1]-y_new[0], decimals=2)))

    return data_matrix_new



def fit_2d_gaussian(data, x_limits=[0, 300], y_limits=[0, 300], points_array=[300, 300]):
    """
    Use in scanned current in a surface assuming no temperature variation
    Fit a 2-D gaussian from the data
    :param data:                (float) list or 1-D array containing the measured data values of each point of the scanned surface
    :param x_limits:            (float) list with the beginning and end of X-axis, i.e. [xo, xf]
    :param y_limits:            (float) list with the beginning and end of Y-axis, i.e. [yo, yf]
    :param points_array:        (int array) number of points per axis, in other way, the dimension of the new matrix
    :return:                    a 2-D gaussian with the parameters from the fit
    """

    params = ff.fit_gaussian(data)

    (height, x_center, y_center, width_x, width_y) = params
    print('From fit 2-d gaussian')
    print('Height   = {}'.format(height))
    print('X center = {}'.format(x_center))
    print('Y center = {}'.format(y_center))
    print('X width  = {}'.format(width_x))
    print('Y width  = {}'.format(width_y))

    x_mesh = np.linspace(x_limits[0], x_limits[1], points_array[0])
    y_mesh = np.linspace(y_limits[0], y_limits[1], points_array[1])
    x_mesh, y_mesh = np.meshgrid(x_mesh, y_mesh)
    gauss_fit = ff.gaussian(*params)(x_mesh, y_mesh)

    return params, gauss_fit


def calculate_systematic_error(mean):
    """
    calculates the statistical and systematic errors and their combination (total error) for each point of the array.
    Each point of the array is a mean of N points previously taken by the acquisition device.

    SYSTEMATIC ERROR : Keithley 6487 Picoammeter Specification
    Computing systematic error due to the measurement apparatus
    According to Manual : range: 2    nA    : 0.30 % of RDG + 400 fA
                          range: 20   nA    : 0.20 % of RDG + 1   pA
                          range: 200  nA    : 0.15 % of RDG + 10  pA
                          range: 2    µA    : 0.15 % of RDG + 100 pA
                          range: 20   µA    : 0.10 % of RDG + 1   nA
                          range: 200  µA    : 0.10 % of RDG + 10  nA
                          range: 2    mA    : 0.10 % of RDG + 100 nA
                          range: 20   mA    : 0.10 % of RDG + 1   µA
    :param mean:
    :param std:
    :return:
    """

    range_array = 1e-9 * np.array([2, 20, 200, 2e+3, 20e+3, 200e+3, 2e+3, 20e+6])
    factor = np.array([0.30, 0.20, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10]) / np.array(100)
    correction = 1e-9 * np.array([400e-6, 1e-3, 10e-3, 100e-3, 1, 10, 100, 1e+3])

    error_sys = []

    for i, value in enumerate(mean):
        if np.abs(value) < 0.:
            print('value out of range : value lower than expected')
        elif (np.abs(value) > 0.) & (np.abs(value) <= range_array[0]):
            index = 0
        elif (np.abs(value) > range_array[0]) & (np.abs(value) <= range_array[1]):
            index = 1
        elif (np.abs(value) > range_array[1]) & (np.abs(value) <= range_array[2]):
            index = 2
        elif (np.abs(value) > range_array[2]) & (np.abs(value) <= range_array[3]):
            index = 3
        elif (np.abs(value) > range_array[3]) & (np.abs(value) <= range_array[4]):
            index = 4
        elif (np.abs(value) > range_array[4]) & (np.abs(value) <= range_array[5]):
            index = 5
        elif (np.abs(value) > range_array[5]) & (np.abs(value) <= range_array[6]):
            index = 6
        elif (np.abs(value) > range_array[6]) & (np.abs(value) <= range_array[7]):
            index = 7
        else:
            print('value out of range : value higher than max')

        error_sys.append(factor[index] * mean[i] + correction[index])

    error_sys = np.array([error_sys])

    return error_sys

def calculate_statistic_error(std, N=10):
    """

    :param std:
    :param N:
    :return:
    """
    error_stat = std / np.sqrt(N)

    return error_stat

def calculate_total_error(mean, std, N=10):
    """

    :param mean:    (float array) an array of mean values done by N measurements for each element at a point in time
    :param std:     (float array) an array of standard deviation values done by N measurements for each element at a point
                    in time (do not confuse it with the error on the mean that would be given by std / sqrt(N)
    :param N:       (int) number of the given measurements taken to make a mean point
    :return:        (float array) error addition of systematic and statistical errors
    """

    systematic = calculate_systematic_error(mean)
    statistic = calculate_statistic_error(std, N)

    error = np.sqrt(systematic**2 + statistic**2)

    return error

def calculate_errors_in_thermal_range(mean, std, N=10):
    """
    calculates total error from the statistical and systematic errors for each point of the array.
    Each point of the array is a mean of N points previously taken by the acquisition device.

    SYSTEMATIC ERROR : Keithley 6487 Picoammeter Specification
    Computing systematic error due to the measurement apparatus
    According to Manual : range: 2    nA    : 0.30 % of RDG + 400 fA
                          range: 20   nA    : 0.20 % of RDG + 1   pA
                          range: 200  nA    : 0.15 % of RDG + 10  pA
                          range: 2    µA    : 0.15 % of RDG + 100 pA
                          range: 20   µA    : 0.10 % of RDG + 1   nA
                          range: 200  µA    : 0.10 % of RDG + 10  nA
                          range: 2    mA    : 0.10 % of RDG + 100 nA
                          range: 20   mA    : 0.10 % of RDG + 1   µA

    :param mean:    (float array) an array of mean values done by N measurements for each element at a point in time
    :param std:     (float array) an array of standard deviation values done by N measurements for each element at a point
                    in time (do not confuse it with the error on the mean that would be given by std / sqrt(N)
    :param N:       (int) number of the given measurements taken to make a mean point
    :return:        3 arrays of errors, systematic
    """

    #range_array = np.array([2e-9, 20e-9, 200e-9, 2e-6, 20e-6, 200e-6, 2e-3, 20e-3])
    #factor = np.array([0.30, 0.20, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10]) / np.array(100)
    #correction = np.array([400e-15, 1e-12, 10e-12, 100e-12, 1e-9, 10e-9, 100e-9, 1e-6])

    range_array = 1e-9 * np.array([2, 20, 200, 2e+3, 20e+3, 200e+3, 2e+3, 20e+6])
    factor = np.array([0.30, 0.20, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10]) / np.array(100)
    correction = 1e-9 * np.array([400e-6, 1e-3, 10e-3, 100e-3, 1, 10, 100, 1e+3])

    error = []

    for i, value in enumerate(mean):
        if np.abs(value) < 0.:
            print('value out of range : value lower than expected')
        elif (np.abs(value) > 0.) & (np.abs(value) <= range_array[0]):
            index = 0
        elif (np.abs(value) > range_array[0]) & (np.abs(value) <= range_array[1]):
            index = 1
        elif (np.abs(value) > range_array[1]) & (np.abs(value) <= range_array[2]):
            index = 2
        elif (np.abs(value) > range_array[2]) & (np.abs(value) <= range_array[3]):
            index = 3
        elif (np.abs(value) > range_array[3]) & (np.abs(value) <= range_array[4]):
            index = 4
        elif (np.abs(value) > range_array[4]) & (np.abs(value) <= range_array[5]):
            index = 5
        elif (np.abs(value) > range_array[5]) & (np.abs(value) <= range_array[6]):
            index = 6
        elif (np.abs(value) > range_array[6]) & (np.abs(value) <= range_array[7]):
            index = 7
        else:
            print('value out of range : value higher than max')
            print(value, i)

        eta = factor[index]
        gamma = correction[index]

        #low_bound = range_array[index-1]
        #up_bound = range_array[index]
        #print('value : {}, range between {} and {}'.format(value, low_bound, up_bound))

        computed_error = (1/N) * np.sqrt(std[i]**2 + N * (eta * mean[i] + gamma)**2)
        error.append(computed_error)

    error = np.array(error)

    return error



def make_mask(temperature, temperature_array = [25, 20, 15, 10, 5], cut=10):
    """

    :param temperature:
    :param temperature_array:
    :param cut:
    :return:
    """

    # Creating a Mask for every temperature range and coefficient per 5°C
    mask_array = []
    offset = 0.1
    for i, fixed in enumerate(temperature_array):

        mask = (temperature > fixed - offset) & (temperature < fixed + offset)
        mask_true = np.argwhere(mask == True)

        mask[mask_true.min(): mask_true.min() + 1 + cut] = False
        mask[mask_true.max() - cut: mask_true.max() + 1] = False
        mask_array.append(np.array(mask))

    mask_array = np.array(mask_array)

    long_mask = np.full((1, len(temperature)), False)

    for a_mask in mask_array:
        long_mask = np.logical_or(long_mask, a_mask)

    return mask_array, long_mask[0]



def compute_masked_values(mask_array, data, data_error):
    """
    Compute the mean values, errors on the mean values, and error propagation of a masked array
    :param mask_array:   (bool array) array of array of bool, chaque element of the principal array is a mask
    :param data:         (float array) array of current values
    :param data_error:   (float array) array of current errors values
    :return: Returns an 3 arrays, each one of the size of the mask_array : mean, error on the mean, and error propagation
    """

    mean = []
    error_on_mean = []
    error_propagation = []

    for i, a_mask in enumerate(mask_array):
        mean.append(np.mean(data[a_mask]))
        error_on_mean.append(np.std(data[a_mask]) / np.sqrt(len(data[a_mask])))
        error_propagation.append((1/len(data_error[a_mask])) * np.sqrt(np.sum(data_error[a_mask]**2)))

    mean = np.array(mean)
    error_on_mean = np.array(error_on_mean)
    error_propagation = np.array(error_propagation)

    return mean, error_on_mean, error_propagation



def compute_relative_difference(data, data_error, temperature_array=[25, 20, 15, 10, 5]):
    """
    Compute from arguments points of type (temperature difference, relative difference ± relative difference error)
    :param current:
    :param current_error:
    :return:    Returns an array of points of :
                temperature difference called 'tdf'
                relative difference called 'rdf'
                relative difference error 'rdf_error'
    """

    #print('relative differences and errors computed with :')
    #print('current :', current)
    #print('error : ', current_error)

    temperature_array = np.array(temperature_array)

    tdf_matrix = np.zeros((len(temperature_array), (len(temperature_array))))
    rdf_matrix = np.zeros((len(temperature_array), (len(temperature_array))))
    rdf_error_matrix = np.zeros((len(temperature_array), (len(temperature_array))))

    for i, temp in enumerate(temperature_array):
        tdf = temperature_array - temperature_array[i]
        tdf = -tdf
        rdf = data[i]/data - 1

        # Error computation for the relative differences
        first = (1/data[i]) * data_error
        second = (-1*data) * (1/data[i]**2) * data_error[i]
        rdf_error = np.sqrt(first**2 + second**2)

        tdf_matrix[i] = tdf
        rdf_matrix[i] = rdf
        rdf_error_matrix[i] = rdf_error

    rdf_matrix = np.array([100]) * rdf_matrix
    rdf_error_matrix = np.array([100]) * rdf_error_matrix

    return tdf_matrix, rdf_matrix, rdf_error_matrix



def interpolate_thermal_coefficient(delta_temperature, relative_differences):
    """

    :param delta_temperature:
    :param relative_differences:
    :return:
    """

    slope = []
    slope_error = []
    intersect = []
    intersect_error = []

    for i, line in enumerate(relative_differences):

        (slope_value, intersect_value), cov_matrix = np.polyfit(delta_temperature[i], relative_differences[i], 1, cov=True)
        (slope_error_value, intersect_error_value) = np.sqrt(np.diag(cov_matrix))

        slope.append(slope_value)
        slope_error.append(slope_error_value)
        intersect.append(intersect_value)
        intersect_error.append(intersect_error_value)

    slope = np.array(slope)
    slope_error = np.array(slope_error)
    intersect = np.array(intersect)
    intersect_error = np.array(intersect_error)

    return slope, slope_error, intersect, intersect_error



def get_thermal_coefficient(data_at_temperature, data_error_at_temperature, temperature_array):
    """

    :param data_at_temperature:
    :param data_error_at_temperature:
    :return:
    """

    tdf, rdf, rdf_error = compute_relative_difference(data=data_at_temperature,
                                                      data_error=data_error_at_temperature,
                                                      temperature_array=temperature_array)

    slope, slope_error, intersect, intersect_error = interpolate_thermal_coefficient(delta_temperature=tdf,
                                                                                     relative_differences=rdf)

    mean_slope = np.mean(slope)
    error_prop_slope = (1 / len(slope_error)) * np.sqrt(np.sum(slope_error ** 2))

    mean_intersect = np.mean(intersect)
    error_prop_intersect = (1 / len(intersect_error)) * np.sqrt(np.sum(intersect_error ** 2))

    # The slope is the thermal coefficient searched in %/°C
    return mean_slope, error_prop_slope, mean_intersect, error_prop_intersect



def interpolate_array(array, number_of_points=1000, smoothness=0):
    """

    :param array:
    :param number_of_points:
    :param smoothness:
    :return:
    """

    x = np.arange(len(array))
    y = array

    s = scipy.interpolate.UnivariateSpline(x, y, s=smoothness)

    xs = np.linspace(x.min(), x.max(), number_of_points)
    ys = s(xs)

    return ys


def interpolate_1d_data(x_array, y_array, smoothness=0):
    """

    :param x_array:
    :param y_array:
    :param smoothness:
    :return:
    """

    s = scipy.interpolate.UnivariateSpline(x_array, y_array, s=smoothness)

    return s


def transform_from_current_to_irradiance(current, current_errors):

    illuminated_area = 0.028**2 # m^2

    # Getting the desired photo-sensitivity for the peak wavelength (1st approximation)
    ph_wavelength, photosensitivity = read_pindiode_photo_sensitivity()

    ph_wavelength_smooth = np.linspace(ph_wavelength.min(), ph_wavelength.max(), 1000)
    photosensitivity_smooth = UnivariateSpline(ph_wavelength, photosensitivity, ph_wavelength_smooth)

    pin_diode_wavelength = ph_wavelength_smooth[68]
    pin_diode_pde = photosensitivity_smooth[68]
    print('Photosensitivity of {} for {} nm wavelength'.format(pin_diode_pde, pin_diode_wavelength))

    factor = pin_diode_pde * illuminated_area
    irradiance = (current / factor)
    irradiance_errors = (current_errors / factor)

    return irradiance, irradiance_errors

