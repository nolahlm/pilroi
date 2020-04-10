import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from ipywidgets import *

from pilroi.roi import get_idx


def show_image(scan, idx, vmin=1e2, vmax=1e7, cen_pix=False, ywidth=40, **kwargs):
    ''' Plots the given idx of a scan

    Args:
        scan (DataFrame): created with create_scan(), must be cropped
        idx (int): row index
        vmin (float): min intensity to plot
        vmax (float): max intensity to plot
        cen_pix (bool): whether or not max intensity pixel is indicated
        ywidth (int): plot y limits, offset from center pixel
    '''

    fig, ax = plt.subplots()

    ax.imshow(scan['crop'][idx], norm=LogNorm(vmax=vmax, vmin=vmin), **kwargs)

    # Added so using a BL21 scan doesn't throw an error
    if 'l' in scan.columns:
        ax.set_title('L = ' + str(scan['l'][idx])[0:5])

    # Median in y, for setting limits of plot
    y_med = scan['px_y'].median(axis=0)

    ax.set_ylim(bottom=y_med - ywidth, top=y_med + ywidth)

    if cen_pix:
        ax.plot(scan['px_x'][idx], scan['px_y']
                [idx], marker='.', color='r', ms=4)


def animate_scan(scan, **kwargs):
    ''' Creates a scan animation with a slider in a Jupyter notebook using
    ipywidgets

    Args:
        scan (DataFrame) created with analyze.create_scan(), must be a
        cropped scan
    '''

    # Create the figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)

    im = ax.imshow(scan['crop'][0], norm=LogNorm(vmax=1e7, vmin=1e2), **kwargs)

    # Plot the center pixel
    x, = ax.plot(scan['px_x'][0], scan['px_y'][0],
                 marker='.', ms=5, color='r', alpha=.6)

    y_median = scan['px_y'].median(axis=0)
    ax.set_ylim(bottom=y_median - 50, top=y_median + 50)

    # Set title, 'l=, cx, cy = '
    ax.set_title(str(scan['l'][0])[0:5] + ' cx = ' +
                 str(scan['px_x'][0]) + ' cy = ' + str(scan['px_y'][0]))

    def update(l, **kwargs):
        # Need to use an index from scan in order to find our specific l value
        idx = get_idx(scan, 'l', l)
        # Update plotted data
        im.set_data(scan['crop'][idx])
        im.set_norm(norm=LogNorm(vmax=1e7, vmin=1e2))

        x.set_data(scan['px_x'][idx], scan['px_y'][idx])

        ax.set_title('l = ' + str(scan['l'][idx])[0:5] + ' cx = ' +
                     str(scan['px_x'][idx]) + ' cy = ' + str(scan['px_y'][idx]))  # noqa

        fig.canvas.draw()

        return l

    interact(update, l=widgets.FloatSlider(value=1, min=scan['l'].min(axis=0),
                                           max=scan['l'].max(axis=0), step=0.005, continuous_update=False))
