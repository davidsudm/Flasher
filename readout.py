#! /anaconda3/envs/Flasher/bin/python

import csv
import datetime as dt
import numpy as np
import sys

# project_dir = '/Users/lonewolf/PyCharmProjects/Flasher'
project_dir = './'
data_dir = './Data'
output_dir = './Output'
specification_dir = './Data/Specification'


def read_file_of_diodes(path_to_file):

    time = []
    timestamp = []
    big_photodiode_current = []
    big_photodiode_current_std = []
    small_photodiode_current = []
    small_photodiode_current_std = []
    thermoresitor_temperature = []

    file_address = data_dir + '/{}'.format(path_to_file)
    with open(file_address) as file:
        data = csv.reader(file, delimiter=' ')
        for i, row in enumerate(data):
            if i == 0:
                continue
            timestamp.append(dt.datetime.strptime(str(row[0] + ' ' + str(row[1])), '%d/%m/%Y %H:%M:%S'))
            big_photodiode_current.append(float(row[2]) / 1e-9) # nano Ampere
            big_photodiode_current_std.append(float(row[3]) / 1e-9)  # nano Ampere
            small_photodiode_current.append(float(row[4]) / 1e-9)  # nano Ampere
            small_photodiode_current_std.append(float(row[5]) / 1e-9)        # Ampere
            thermoresitor_temperature.append(float(row[6]))

    timestamp = np.array(timestamp)
    big_photodiode_current = np.array(big_photodiode_current)
    big_photodiode_current_std = np.array(big_photodiode_current_std)
    small_photodiode_current = np.array(small_photodiode_current)
    small_photodiode_current_std = np.array(small_photodiode_current_std)
    thermoresitor_temperature = np.array(thermoresitor_temperature)

    for i, date in enumerate(timestamp):
        time.append(timestamp[i] - timestamp[0])
    for i, delta_time in enumerate(time):
        time[i] = delta_time.total_seconds()

    time = np.array(time)

    return time, timestamp, big_photodiode_current, big_photodiode_current_std, small_photodiode_current, small_photodiode_current_std, thermoresitor_temperature


# Read one file each time the function is called
def read_file(path_to_file, scan_type, initial_temperature_time=None):
    """
    read file of data
    :param path_to_file:                (string) file location on drive
    :param scan_type:                   (string) scan_type has only 3 possible values: time, space, or temp
    :param initial_temperature_time:    (string) date from the first measurement of temperature from Climatic Chamber, taken by hand after acquisition.
                                                It should have the following format : %d/%m/%Y %H:%M:%S
    :return:    data array of several variables taken from acquisition like,
                for 'time' and 'space' scan:
                    x :                     (float) x-axis position in a space scan in millimetres
                    y :                     (float) y-axis position in a space scan in millimetres
                    current :               (float) average current of 10 consecutive measurement for a given point, in Amperes
                    current_std :           (float) standard deviation from the current measurements done at acquisition, Amperes
                    current_timestamp:      (datetime) timestamps of the format '%d/%m/%Y %H:%M:%S' of each measurement of current done
                for 'temp' scan:
                    temperature_time :      time in seconds for each measurement of temperature done by the climatic chamber for a given point in time
                    temperature_timestamp : timestamps of the format '%d/%m/%Y %H:%M:%S' of each measurement of temperature done
                                            empty list if parameter 'initial_temperature_time' is None
                    measured_temperature :  temperature measurement done by the climatic chamber for a given point in time
    """
    # path_to_file :    Relative path to the file

    if scan_type != 'time' and scan_type != 'space' and scan_type != 'temp':
        print('scan_type value is : {}'.format(scan_type))
        print('scan_type has only 3 possible values: time or space or temp')
        sys.exit()

    if scan_type != 'temp':

        x = []
        y = []
        current = []
        current_std = []
        current_time = []
        current_timestamp = []

        if scan_type == 'space':
            file_address = data_dir + '/{}'.format(path_to_file)
            with open(file_address) as file:
                data = csv.reader(file, delimiter=' ')
                for i, row in enumerate(data):
                    if i == 0:
                        continue
                    x.append(float(row[0]))
                    y.append(float(row[1]))
                    #current.append(float(row[2]) / 1e-9)        # nano Ampere
                    #current_std.append(float(row[3]) / 1e-9)    # nano Ampere
                    current.append(float(row[2]))        # Ampere
                    current_std.append(float(row[3]))    # Ampere
                    current_timestamp.append(dt.datetime.strptime(str(row[4] + ' ' + str(row[5])), '%d/%m/%Y %H:%M:%S'))
            x = np.array(x)
            y = np.array(y)
            current = np.array(current)
            current_std = np.array(current_std)
            current_timestamp = np.array(current_timestamp)

        else: # scan_type == 'time'
            file_address = data_dir + '/{}'.format(path_to_file)
            with open(file_address) as file:
                data = csv.reader(file, delimiter=' ')
                for i, row in enumerate(data):
                    if i == 0:
                        continue
                    x.append(float(float(0)))
                    y.append(float(float(0)))
                    #current.append(float(row[2]) / 1e-9)        # nano Ampere
                    #current_std.append(float(row[3]) / 1e-9)    # nano Ampere
                    current.append(float(row[2]))  # Ampere
                    current_std.append(float(row[3]))  # Ampere
                    current_timestamp.append(dt.datetime.strptime(str(row[0] + ' ' + str(row[1])), '%d/%m/%Y %H:%M:%S'))

            x = np.array(x)
            y = np.array(y)
            current = np.array(current)
            current_std = np.array(current_std)
            current_timestamp = np.array(current_timestamp)

        for i, date in enumerate(current_timestamp):
            current_time.append(current_timestamp[i] - current_timestamp[0])
        for i, delta_time in enumerate(current_time):
            current_time[i] = delta_time.total_seconds()
        current_time = np.array(current_time)

        return x, y, current, current_std, current_time, current_timestamp

    else: # scan_type == 'temp'

        temperature_time = []      # time in seconds from the starting point
        temperature_timestamp = []  # timestamps of temperature measurements
        measured_temperature = []   # measured temperature in Celsius given via the Climate Chamber
        demanded_temperature = []   # demanded temperature in Celsius by the user (programmed)
        measured_humidity = []      # measured humidity given via the Climate Chamber
        demanded_humidity = []      # demanded humidity by the user (programmed)

        file_address = data_dir + '/{}'.format(path_to_file)
        with open(file_address) as file:
            data = csv.reader(file, delimiter=';')
            for i, row in enumerate(data):
                if i == 0:
                    continue
                if i == 1:
                    continue
                temperature_time.append(float(row[0]))
                measured_temperature.append(float(row[1]))
                demanded_temperature.append(float(row[2]))
                measured_humidity.append(float(row[3]))
                demanded_humidity.append(float(row[4]))

        if initial_temperature_time is not None:
            initial_temperature_time = dt.datetime.strptime(initial_temperature_time, '%d/%m/%Y %H:%M:%S')
            for i, seconds in enumerate(temperature_time):
                temperature_timestamp.append(dt.timedelta(seconds=seconds))
            for i, seconds in enumerate(temperature_timestamp):
                temperature_timestamp[i] = initial_temperature_time + seconds

        temperature_time = np.array(temperature_time)
        temperature_timestamp = np.array(temperature_timestamp)
        measured_temperature = np.array(measured_temperature)
        # demanded_temperature = np.array(demanded_temperature)
        # measured_humidity = np.array(measured_humidity)
        # demanded_humidity = np.array(demanded_humidity)

        #return time, measured_temperature, demanded_temperature, measured_humidity, demanded_humidity
        return measured_temperature, temperature_time, temperature_timestamp



# Read a bunch of file (from a list) each time the function is called
def read_list_of_files(filelist):
    x = []
    y = []
    current = []
    current_std = []
    timestamp = []

    for filename in filelist:
        # _temp stands for 'temporal file' and not 'temperature'
        x_temp, y_temp, current_temp, current_std_temp, timestamp_temp = read_file(filename)
        for item in x_temp:
            x.append(item)
        for element in y_temp:
            y.append(element)
        for element in current_temp:
            current.append(element)
        for element in current_std_temp:
            current_std.append(element)
        for element in timestamp_temp:
            timestamp.append(element)

    x = np.array(x)
    y = np.array(y)
    current = np.array(current)
    current_std = np.array(current_std)

    return x, y, current, current_std, timestamp


# Read temperature from Climate Chamber data
def read_temperature_file(path_to_file):
    time = []                   # time in seconds from the starting point
    measured_temperature = []   # measured temperature in Celsius given via the Climate Chamber
    demanded_temperature = []   # demanded temperature in Celsius by the user (programmed)
    measured_humidity = []      # measured humidity given via the Climate Chamber
    demanded_humidity = []      # demanded humidity by the user (programmed)

    file_address = data_dir + '/{}'.format(path_to_file)
    with open(file_address) as file:
        data = csv.reader(file, delimiter=';')
        for i, row in enumerate(data):
            if i == 0:
                continue
            if i == 1:
                continue
            time.append(float(row[0]))
            measured_temperature.append(float(row[1]))
            demanded_temperature.append(float(row[2]))
            measured_humidity.append(float(row[3]))
            demanded_humidity.append(float(row[4]))

    time = np.array(time)
    measured_temperature = np.array(measured_temperature)
    demanded_temperature = np.array(demanded_temperature)
    measured_humidity = np.array(measured_humidity)
    demanded_humidity = np.array(demanded_humidity)

    return time, measured_temperature, demanded_temperature, measured_humidity, demanded_humidity


def read_cerenkov_spectra(cerenkov_file='./Data/Specification/Spectra_Ch_300-600.txt'):
    wavelength = []
    spectra = []

    file_address = cerenkov_file

    with open(file_address) as file:
        data = csv.reader(file, delimiter=' ')
        for i, row in enumerate(data):
            if i == 0:
                continue
            if i == 1:
                continue
            if i == 2:
                continue
            wavelength.append(float(row[0]))
            spectra.append(float(row[1]))

    wavelength = np.array(wavelength)
    spectra = np.array(spectra)

    return wavelength, spectra


def read_pindiode_temperature_coefficient(temperature_coefficient_file='./Data/Specification/PinDiode_Photosensitivity_temperature_characteristic.csv'):
    wavelength = []
    temperature_coefficient = []

    file_address = temperature_coefficient_file

    with open(file_address) as file:
        data = csv.reader(file, delimiter=' ')
        for i, row in enumerate(data):
            if i == 0:
                continue
            wavelength.append(float(row[0]))
            temperature_coefficient.append(float(row[1]))

    wavelength = np.array(wavelength)
    temperature_coefficient = np.array(temperature_coefficient)

    return wavelength, temperature_coefficient


def read_pindiode_photo_sensitivity(cerenkov_file='./Data/Specification/PinDiode_Spectral_response.csv'):
    wavelength = []
    photo_sensitivity = []

    file_address = cerenkov_file

    with open(file_address) as file:
        data = csv.reader(file, delimiter=' ')
        for i, row in enumerate(data):
            if i == 0:
                continue
            wavelength.append(float(row[0]))
            photo_sensitivity.append(float(row[1]))

    wavelength = np.array(wavelength)
    photo_sensitivity = np.array(photo_sensitivity)

    return wavelength, photo_sensitivity

def read_led_spectra(led_spectra_file):
    wavelength = []
    intensity = []

    file_address = led_spectra_file

    with open(file_address) as file:
        data = csv.reader(file, delimiter=' ')
        for i, row in enumerate(data):
            if i == 0:
                continue
            if i == 1:
                continue
            wavelength.append(float(row[0]))
            intensity.append(float(row[1]))

    wavelength = np.array(wavelength)
    intensity = np.array(intensity)

    return wavelength, intensity

def read_flasher_spectra(flasher_spectra='./Data/Specification/Flasher_Spectra_DC_corrected.txt'):
    wavelength = []
    spectra = []

    file_address = flasher_spectra

    with open(file_address) as file:
        data = csv.reader(file, delimiter=';')
        for i, row in enumerate(data):
            if i in range(0, 7):
                continue
            wavelength.append(float(row[0]))
            spectra.append(float(row[4]))

    wavelength = np.array(wavelength)
    spectra = np.array(spectra)

    return wavelength, spectra



