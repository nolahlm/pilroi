import glob
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_scan_csv(path):
    ''' Imports .csv generated from spec at BL 7-2

    Args:
        path (str): path to .csv file

    Returns:
        dataframe: same columns as .csv

    '''
    data = pd.read_csv(path)
    data = data.rename(columns=lambda x: x.strip())

    # Nice to have all lowercase columns
    data.columns = data.columns.str.lower()

    return data


def read_raw(impath):
    ''' Reads a .raw pilatus image from BL 7-2

    Args:
        impath (str): path to .raw file

    Returns:
        ndarray: data of .raw image, reshaped to appropriate pilatus
        detector dimensions (195 x 487)
    '''

    image = np.fromfile(impath, dtype=np.uint32)
    image.shape = (195, 487)

    return image


def image_paths(folder):
    ''' Parses the .raw file paths from a folder containing them.  Filename of
    pilatus imagers must be 'blah_blah_####.raw'

    Args:
        folder (str): path to a folder that contains .raw pilatus images.

    Returns:
        list of str: ordered list of pilatus filenames
    '''

    imlist = glob.glob(folder + '*.raw')
    imlist.sort(key=lambda x: int(x[-8:-4]))

    return imlist


def pdi_paths(folder):
    ''' Parses the .raw.pdi file paths from a folder containing them.
    Filename of pilatus imagers must be 'blah_blah_####.raw.pdi'

    Args:
        folder (str): path to a folder that contains .raw.pdi pilatus images.

    Returns:
        list of str: ordered list of pilatus filenames
    '''
    pdilist = glob.glob(folder + '*.raw.pdi')
    pdilist.sort(key=lambda x: int(x[-8:-4]))

    return pdilist


def pdi_parse(path):
    ''' Extracts motor positions from a .raw.pdi detector scan file


    args:
        path (str): path to file

    returns:
        motors (dict): motor positions, and Calculated Detector
        Calibration Parameters for image from .pdi.raw

    '''
    file = open(path, 'r')
    lines = file.readlines()

    motor_line = _find_num(lines[3])
    calib_line = _find_num(lines[5])

    # If the .pdi file changes you're screwed...
    motors = {
        'th': motor_line[0],
        'tth': motor_line[2],
        'chi': motor_line[3],
        'phi': motor_line[4],
        'gamma': motor_line[5],
        'mu': motor_line[6],
        'PD_X': calib_line[0],
        'PD_y': calib_line[1],
        'PD_DIST': calib_line[2],
        'PD_ALPHA': calib_line[3],
        'PD_DELTA': calib_line[4],
        'LAMBDA': calib_line[5]
    }

    for key, val in motors.items():
        motors[key] = float(val.strip())
    return motors


def _find_num(string):
    ''' Uses literal black magic to find all numbers in a specific string.

    args:
        string (str)

    returns
        (list) all of the numerical sequences in the given string
    '''

    return re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", string)  # noqa


def foil_attenuation(foils, foil_insertion):
    ''' Calculates the attenuation factor due to foils

    Args:
        foils (list): attenuation factor for foils.  len(foils) = 4
        foil_insertion (ndarray): which foils are inserted

    Returns:
        ndarray: Attenuation factor based upon foils
    '''
    # Foil insertion '0011' is saved to data file as 11.  This returns the
    # full foil expression which should be 4 bools

    # Turn foils into a list of ints
    foil_insertion = [int(i) for i in str(foil_insertion)]
    # Insert the required zeros
    while len(foil_insertion) is not 4:
        foil_insertion.insert(0, 0)

    # multiply foil insertion with foil attenuation factor
    attenuation = [foil_insertion[i] * foils[i] for i in range(4)]

    return np.exp(sum(attenuation))


def create_scan(scanpath, imfolder, foils, bl=None):
    ''' Creates a scan dataframe, columns h, k, l, attenuation, monitor, raw,
    norm

    Args:
        scanpath (str): path to scan .csv
        imfolder (str): path to a folder that contains .raw pilatus images
        foils (list): attenuation factor for foils.  len = 4
        bl (str): one of '72' or '21'

    Returns:
        DataFrame: Processed h, k, l, and pilatus signal

    '''

    # First I need to get the base data
    data = read_scan_csv(scanpath)
    im_paths = image_paths(imfolder)
    if bl is not None:
        if bl is '72':
            columns = ['h', 'k', 'l', 'monitor', 'foils']
        if bl is '21':
            columns = ['twotheta', 'theta', 'monitor', 'foils', 'normalized']
    if bl is None:
        print('Please enter an appropriate beamline')
        return None

    # I need to specify .copy() at the end, otherwise adding columns to scan
    # raises "SettingWithCopyWarning"
    scan = data[columns].copy()

    # Calculate filter attenuation
    # scan.assign(attenuation=[foil_attenuation(
    #     foils, foil_insertion) for foil_insertion in scan['foils']])

    scan.loc[:, 'attenuation'] = [foil_attenuation(
        foils, foil_insertion) for foil_insertion in scan['foils']]
    scan.loc[:, 'raw'] = [read_raw(path) for path in im_paths]

    scan.loc[:, 'norm'] = scan['raw'] * scan['attenuation'] / scan['monitor']

    return scan


def crop_scan(scan, lim1, lim2):
    ''' Crops raw 2D signal in x direction based on user inputs for lim1 and lim2
    Adds columns crop, px_x, px_y for signal, max x, and max y pixels

    Args:
        scan (DataFrame): created with create_scan()
        lim1 (int): crop limit 1
        lim2 (int): crop limit 2
    '''

    scan['crop'] = [x[:, lim1:lim2] for x in scan['norm']]

    # Find max x and y pixels
    pxmax = []
    for x in scan['crop'].values:
        pxmax.append(np.unravel_index(x.argmax(), x.shape))

    scan['px_x'] = [i[1] for i in pxmax]
    scan['px_y'] = [i[0] for i in pxmax]


def find_limits(scan, window, center=False, **kwargs):
    ''' Plots a mean of all images in the dataset, useful to find lim1 and
    lim2 for crop_scan()

    This is done by finding the center and defining a symmetric window around
    it.  All rows of the detector image are summed together and a line profile
    is plotted from which the window width can be visually verified.

    An argument 'center' is included to account for instances in which the max
    intensity is not actually the center of the window

    Args:
        scan (DataFrame): created with create_scan
        window (int): width of data window
        center (False or a #): "Center" pixel location in x.

    Returns:
        lim1 (int): limit 1
        lim2 (int): limit 2
    '''

    profile = scan.mean(axis=0)['norm'].sum(axis=0)

    xmax = profile.argmax()

    # If the center pixel identified is not correct
    if center:
        xmax = center

    fig, ax = plt.subplots()
    ax.semilogy(profile, **kwargs)
    # Plot center
    ax.axvline(xmax, color='k', ls='--')

    # Is it silly to do this with an int?
    lim1 = int(xmax - window / 2)
    lim2 = int(xmax + window / 2)

    # Plot limit lines
    ax.axvline(lim1, color='r', ls='--')
    ax.axvline(lim2, color='r', ls='--')
    plt.show()
    return lim1, lim2
