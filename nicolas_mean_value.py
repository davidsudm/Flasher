from matplotlib import rc, rcParams
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import numpy as np
from digicampipe.instrument.camera import DigiCam
from ctapipe.visualization import CameraDisplay
from ctapipe.instrument.camera import CameraGeometry
from digicampipe.visualization.plot import plot_array_camera
from scipy import interpolate
import scipy as scp
import scipy.ndimage

mean_trans_cher = [68.7832578071, 78.1275274659] #computed with the other script (scan_window.py/first_window.py)

#loading pixels id and coordinates from camera config file
pixid, x_pix, y_pix = np.loadtxt('/Users/Nicolas/Desktop/UniGe/Master 2/Travail de Master/Window measurements/data/camera_config.cfg.txt', unpack =True, skiprows=47, usecols=(8,9,10))

pixid, x_pix, y_pix = zip(*sorted(zip(pixid, x_pix, y_pix)))
pixid=list(pixid)
x_pix=list(x_pix)
y_pix=list(y_pix)
pixid=[int(x) for x in pixid]

folder = ['first_window', 'PLOT']
num = ['1st', '2nd']

def hexagon(pos, pixpos):
    """
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
    #x, y = map(abs, pos)
    #return x < min(3**0.5*s + ypix + xpix/(3**0.5) - 3**0.5*y, 3**0.5*s/2 + xpix)

def polar2cartesian(outcoords, inputshape, origin):
    """Coordinate transform for converting a polar array to Cartesian coordinates.
    inputshape is a tuple containing the shape of the polar array. origin is a
    tuple containing the x and y indices of where the origin should be in the
    output array."""

    xindex, yindex = outcoords
    x0, y0 = origin
    x = xindex - x0
    y = yindex - y0

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    theta_index = np.round((theta + np.pi) * inputshape[1] / (2 * np.pi))

    return (r,theta_index)


def extract_trans_perpix(window_number):
    """
    Extract mean value per pixel of trans_norm_cher
    :param window_number: 1 or 2 to choose between the first or second window
    :return: text file with mean trans_norm_cher values per pixel
    """
    r, trans = np.loadtxt('data/trans_norm_cher_vs_r.txt', unpack=True, skiprows=1, usecols=(0, window_number)) # transmittance of the first window normalised with cherenkov spectrum
    n_radius = 516
    n_radial_bin = 2000
    p = np.linspace(0, 2 * np.pi, n_radial_bin)
    R, P = np.meshgrid(r[:n_radius], p)
    Z = np.array([trans[:n_radius]] * n_radial_bin)

    # Express the mesh in the cartesian system.

    trans_cart = scp.ndimage.geometric_transform(Z, polar2cartesian, order=0, output_shape=(Z.shape[0] * 2, Z.shape[0] * 2), extra_keywords={'inputshape': Z.shape, 'origin': (Z.shape[0], Z.shape[0])})

    print(trans_cart.shape, trans_cart)

    X, Y = R * np.cos(P), R * np.sin(P)

    print(X.shape, Y.shape, Z.shape)

    pix_trans_cher=[]
    for k in range(len(pixid)):
        pix_values=[]
        for i in range(len(X)):
            for j in range(len(X[0])):
                if hexagon((X[i][j], Y[i][j]), (x_pix[k], y_pix[k])):
                    pix_values.append(Z[i][j])
        pix_trans_cher.append(np.mean(pix_values))
        print("pixel {0}: {1}, {2}".format(pixid[k],pix_trans_cher[k],len(pix_values)))

    path="data/mean_transmittance_per_pixel_{0}_wdw.txt".format(num[window_number-1])
    output=open(path,"w")
    output.write("pixel_id\t transmittance_cher\n")
    for k in range(len(pixid)):
        output.write("{0}\t {1}\n".format(pixid[k], pix_trans_cher[k]))
    output.close()



def plot_window_pixid():
    """
    Plot of the camera drawing with pixels numbering
    :return: plot of the pixels with corresponding pixel_id
    """
    w, h = mpl.figure.figaspect(0.87)
    fig = plt.figure(figsize=(w,h))
    ax = plt.gca()
    ax.set_alpha(0)
    CameraDisplay(DigiCam.geometry, np.zeros(1296), cmap='binary', title='').highlight_pixels(range(1296), color='k', linewidth=0.2)
    for i in range(len(pixid)):
        plt.text(x_pix[i]-7, y_pix[i]-3, pixid[i], fontsize=2)
    plt.xlim(-550, 550)
    plt.ylim(-550, 550)
    fig.savefig('PLOT/pix_id.pdf')
    plt.clf()


def plot_window_trans(window_number):
    """
    Plot continuous map of trans_norm_cher superimposed with pixels drawing
    :param window_number: 1 or 2 to choose between the first or second window
    :return: plot a radial 2D-map of the transmittance (normalised with the Cherenkov spectrum) superimposed with the pixels drawing
    """
    r, trans = np.loadtxt('data/trans_norm_cher_vs_r.txt', unpack=True, skiprows=1, usecols=(0, window_number)) # transmittance of the first window normalised with cherenkov spectrum
    n_radius = 516
    n_radial_bin = 100
    p = np.linspace(0, 2 * np.pi, n_radial_bin)
    R, P = np.meshgrid(r[:n_radius], p)
    Z = np.array([trans[:n_radius]] * n_radial_bin)

    # Express the mesh in the cartesian system.

    trans_cart = scp.ndimage.geometric_transform(Z, polar2cartesian, order=0, output_shape=(Z.shape[0] * 2, Z.shape[0] * 2), extra_keywords={'inputshape': Z.shape, 'origin': (Z.shape[0], Z.shape[0])})

    print(trans_cart.shape, trans_cart)

    X, Y = R * np.cos(P), R * np.sin(P)

    print(X.shape, Y.shape, Z.shape)

    fig = plt.figure()
    ax = plt.gca()
    ax.set_alpha(0)
    CameraDisplay(DigiCam.geometry, np.zeros(1296), cmap='binary', title='').highlight_pixels(range(1296), color='k', linewidth=0.2)
    surf2 = ax.pcolormesh(X, Y, np.array(Z), cmap='viridis', alpha=0.5)
    plt.annotate("mean: {0:.2f}\%".format(mean_trans_cher[window_number-1]), xy=(-430, 490), xycoords="data", va="center", ha="center", bbox=dict(boxstyle="round", fc="w", ec="silver"))
    plt.xlim(-550, 550)
    plt.ylim(-550, 550)
    cbar2 = fig.colorbar(surf2)
    cbar2.set_label("Transmittance $\otimes$ norm\_cher", rotation=90, labelpad=10)
    fig.savefig('{0}/trans_norm_cher_pixels_{1}wdw.pdf'.format(folder[window_number-1], num[window_number-1]))
    plt.show()


def plot_window_trans_check_perpix(window_number):
    """
    Comparison plot between continuous map of trans_norm_cher and mean value per pixel
    :param window_number: 1 or 2 to choose between the first or second window
    :return: plot a radial 2D-map of the transmittance (normalised with the Cherenkov spectrum) superimposed with a colored dot for each pixel that correspond to the mean value of trans_norm_cher for the pixel
    """
    r, trans = np.loadtxt('data/trans_norm_cher_vs_r.txt', unpack=True, skiprows=1, usecols=(0, window_number))  # transmittance of the first window normalised with cherenkov spectrum
    n_radius = 516
    n_radial_bin = 100
    p = np.linspace(0, 2 * np.pi, n_radial_bin)
    R, P = np.meshgrid(r[:n_radius], p)
    Z = np.array([trans[:n_radius]] * n_radial_bin)

    # Express the mesh in the cartesian system.

    trans_cart = scp.ndimage.geometric_transform(Z, polar2cartesian, order=0, output_shape=(Z.shape[0] * 2, Z.shape[0] * 2), extra_keywords={'inputshape': Z.shape, 'origin': (Z.shape[0], Z.shape[0])})

    print(trans_cart.shape, trans_cart)

    X, Y = R * np.cos(P), R * np.sin(P)

    print(X.shape, Y.shape, Z.shape)

    pix_trans_cher = np.loadtxt("data/mean_transmittance_per_pixel_{0}_wdw.txt".format(num[window_number-1]), usecols=1, skiprows=1, unpack=True)

    fig = plt.figure()
    ax = plt.gca()
    ax.set_alpha(0)
    CameraDisplay(DigiCam.geometry, np.zeros(1296), cmap='binary', title='').highlight_pixels(range(1296), color='k', linewidth=0.2)

    min_ = min(np.min(np.array(Z)), min(pix_trans_cher))
    max_ = max(np.max(np.array(Z)), max(pix_trans_cher))
    ax.scatter(x_pix, y_pix, s=2, c=pix_trans_cher, cmap='viridis', vmin=min_, vmax=max_)

    surf2 = ax.pcolormesh(X, Y, np.array(Z), cmap='viridis', alpha=0.5, vmin=min_, vmax=max_)
    plt.xlim(-550, 550)
    plt.ylim(-550, 550)
    cbar2 = fig.colorbar(surf2)
    cbar2.set_label("Transmittance $\otimes$ norm\_cher", rotation=90, labelpad=10)
    fig.savefig('{0}/check_trans_values_{1}wdw.pdf'.format(folder[window_number-1], num[window_number-1]))
    plt.show()


def plot_window_trans_perpix(window_number):
    """
    Plot of the mean value of trans_norm_cher per pixel
    :param window_number: 1 or 2 to choose between the first or second window
    :return: plot the camera pixels with the mean value of trans_norm_cher for each of the pixels
    """
    pix_trans_cher = np.loadtxt("data/mean_transmittance_per_pixel_{0}_wdw.txt".format(num[window_number-1]), usecols=1, skiprows=1, unpack=True)

    fig = plt.figure()
    ax = plt.gca()
    ax.set_alpha(0)
    CameraDisplay(DigiCam.geometry, pix_trans_cher, cmap='viridis', title='').highlight_pixels(range(1296), color='k', linewidth=0.2)
    CameraDisplay(DigiCam.geometry, pix_trans_cher, cmap='viridis', title='').add_colorbar(label="Transmittance $\otimes$ norm\_cher")
    plt.annotate("mean: {0:.2f}\%".format(np.mean(pix_trans_cher)*100.), xy=(-430, 490), xycoords="data", va="center", ha="center", bbox=dict(boxstyle="round", fc="w", ec="silver"))
    plt.xlim(-550, 550)
    plt.ylim(-550, 550)
    fig.savefig('{0}/trans_values_perpixel_{1}wdw.pdf'.format(folder[window_number-1], num[window_number-1]))
    plt.show()


def plot_window_corr_factor(window_number):
    """
    Plot of the correction factor per pixel
    :param window_number: 1 or 2 to choose between the first or second window
    :return: plot the camera pixels with the correction factor (trans_norm_cher normalised by the overall mean trans_norm_cher) per pixel
    """

    pix_trans_cher = np.loadtxt("data/mean_transmittance_per_pixel_{0}_wdw.txt".format(num[window_number-1]), usecols=1, skiprows=1, unpack=True)

    pix_corr_factor=[x/np.mean(pix_trans_cher) for x in pix_trans_cher]#computing correction factors

    path = "data/correction_factor_per_pixel_{0}_wdw.txt".format(num[window_number - 1])
    output = open(path, "w")
    output.write("pixel_id\t correction_factor\n")
    for k in range(len(pixid)):
        output.write("{0}\t {1}\n".format(pixid[k], pix_corr_factor[k]))
    output.close()

    fig = plt.figure()
    ax = plt.gca()
    ax.set_alpha(0)
    CameraDisplay(DigiCam.geometry, pix_corr_factor, cmap='viridis', title='').highlight_pixels(range(1296), color='k', linewidth=0.2)
    CameraDisplay(DigiCam.geometry, pix_corr_factor, cmap='viridis', title='').add_colorbar(label="Transmittance $\otimes$ norm\_cher normalised to mean value")
    plt.xlim(-550, 550)
    plt.ylim(-550, 550)
    fig.savefig('{0}/corr_factor_perpixel_{1}wdw.pdf'.format(folder[window_number-1], num[window_number-1]))
    plt.show()



if __name__ == '__main__':

    #plot_window_pixid()
    plot_window_trans(1)
    plot_window_trans(2)
    plot_window_trans_perpix(1)
    plot_window_trans_perpix(2)

    '''
    extract_trans_perpix(1)
    plot_window_trans(1)
    plot_window_trans_check_perpix(1)
    plot_window_trans_perpix(1)
    plot_window_corr_factor(1)


    extract_trans_perpix(2)
    plot_window_trans(2)
    plot_window_trans_check_perpix(2)
    plot_window_trans_perpix(2)
    plot_window_corr_factor(2)
    '''
