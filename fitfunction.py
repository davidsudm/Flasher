#! /anaconda3/envs/Flasher/bin/python

import numpy as np
from scipy import optimize
from scipy.optimize import curve_fit


def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x, y: height * np.exp(-(((center_x-x)/width_x)**2 + ((center_y-y)/width_y)**2)/2)


def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    xx, yy = np.indices(data.shape)
    x = (xx * data.T).sum() / total
    y = (yy * data.T).sum() / total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    height = data.max()
    return height, x, y, width_x, width_y


def fit_gaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    error_function = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data.T)
    p, success = optimize.leastsq(error_function, params)

    return p



def irradiance(x, distance, amplitude, frequency, offset):
    
    return (amplitude/distance**2) * np.cos(frequency * (np.arctan((x - offset) / distance))) ** 3



def fit_irradiance(distance, initial_params, x, y):
    """"""

    # Fit function
    # Irradiance ~ cos^3 (angle)
    fitfunc = lambda p, x: (p[0] / distance ** 2) * np.cos(p[1] * (np.arctan((x - p[2]) / distance))) ** 3
    # Distance to the target function
    errfunc = lambda p, x, y: fitfunc(p, x) - y

    params, pcov, infodict, errmsg, success = optimize.leastsq(errfunc,
                                                               initial_params[:],
                                                               args=(x, y),
                                                               full_output=1,
                                                               epsfcn=0.0001)

    if (len(y) > len(initial_params)) and pcov is not None:
        s_sq = (errfunc(params, x, y) ** 2).sum() / (len(y) - len(initial_params))
        pcov = pcov * s_sq
    else:
        pcov = np.inf

    error = []
    for i in range(len(params)):
        try:
            error.append(np.absolute(pcov[i][i]) ** 0.5)
        except:
            error.append(0.00)
    pfit_leastsq = params
    perr_leastsq = np.array(error)

    return pfit_leastsq, perr_leastsq
