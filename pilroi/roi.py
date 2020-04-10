import numpy as np


def make_roi(im_dim, cenx, ceny, height, width):
    ''' Creates an ROI ndarray, a bool array with 1's defining the ROI

    This creates an array of ones and zeros, that can be multiplied with an
    image ndarray to extract data

    Args:
        im_dim (Tuple): Dimensions of image ROI is for (use im.shape)
        cenx (int): Center of ROI
        ceny (int): center of ROI
        height (int): ODD integer, height of ROI
        width (int): ODD integer, width of ROI
    '''

    # Create mask array of zeros
    roi = np.zeros(im_dim)

    # Relative height and width + / - the center pixel
    rel_width = width // 2
    rel_height = height // 2

    # The center pixel is included in the ROI
    roi[ceny, cenx] = 1
    # Loop through coordinates in ROI and make the mask array 1
    # Need the +1 because numpy slicing is exclusive on the 2nd index!
    for x in np.arange(cenx - rel_width, cenx + rel_width + 1):
        for y in np.arange(ceny - rel_height, ceny + rel_height + 1):

            roi[y, x] = 1
    return roi


def roi_extract(scan, roi):
    '''
    Extracts intensity from a specific ROI defined with make_roi()

    Since an roi is a bool array this is just multiplying the roi * the signal

    Args:
        scan (DataFrame): created with create_scan()
        roi (ndarray OR list): created with make_roi() OR roi_track_cen

    Returns:
        signal (ndarray): intensity at each point in the scan
    '''

    # We can accept either one ndarray, or many.  If the input is a list of
    # ndarrays they correspond to individual data points in the scan

    signal = []

    if type(roi) is list:
        for index, row in scan.iterrows():
            data = row['crop'] * roi[index]
            signal.append(data.sum().sum())

    # Otherwise we only look at one ROI
    else:
        for index, row in scan.iterrows():
            data = row['crop'] * roi
            signal.append(data.sum().sum())

    return np.array(signal)


def roi_track_cen(scan, ceny, height, width):
    ''' Creates a list of ndarrays where each item is the ROI for a specific
    index in the scan.  Center of ROI tracks horizontal motion of most intense
    pixel during scan (scan.px_x)

    Args:
        scan (DataFrame): created with create_scan()
        ceny (int) center of ROI vertically
        height (int) height of ROI, odd #
        width (int) width of ROI, odd #

    Returns:
        roi (list): list of ndarrays of ROIs
    '''

    # Empty list for ROIs
    roi = []

    # iterate through scan and append roi object
    for index, row in scan.iterrows():
        roi.append(make_roi(row['crop'].shape, row['px_x'], ceny, height, width))  # noqa

    return roi


def get_idx(scan, col, val):
    ''' Gets the index where col is closest to val

    Args:
        scan (DataFrame): created with create_scan(), must be cropped
        col (str): column in scan
        val (number): value you want to find in given column

    Returns:
        idx (int): Column index that has specific value
    '''

    # Get the row closest to val
    row = scan.iloc[(scan[col] - val).abs().argsort()[:1]]

    idx = row.index[0]

    return idx
