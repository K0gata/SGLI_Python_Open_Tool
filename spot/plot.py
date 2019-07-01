import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.colors as mat_colrs
import numpy as np
import math
import warnings
import logging
from spot.config import SpotWarnings


def saveimg(data: np.ndarray, lat: np.ndarray=None, lon: np.ndarray=None, outpath: str=None,
            cmap=None, clim=None, log10: bool=False,
            cbar: bool=True, cbar_label: str=None,
            mask :np.ndarray=None,
            axes=True, xlabel :str=None, ylabel :str=None,  title :str=None,
            grid :bool=False, grid_interval=None):

    # =====================
    # Config
    # =====================
    dpi = 300
    x0 = 0
    y0 = 0
    plt.rcParams['axes.linewidth'] = 0.5
    lines, pixels = data.shape[:2]
    geo_mode = False

    # =====================c
    # Validation
    # =====================
    if (lat is not None) and (lon is not None):
        lat_shape = lat.shape
        lon_shape = lon.shape
        if lat_shape[0] != lon_shape[0] or lon_shape[1] != lat_shape[1]:
            warnings.warn('Latitude and Longitude data have different array dimensions!: Lat:{0}, Lon:{1}'.format(lat.shape, lon.shape), SpotWarnings, stacklevel=2)
            return None

        if lat_shape[0] != lines or lon_shape[1] != pixels:
            warnings.warn('Plotting data and Lat/Lon data have different array dimensions!: Data:{0}, Lat/Lon:{1}'.format(data.shape, lon.shape), SpotWarnings, stacklevel=2)
            return None

        geo_mode = True

    # =====================
    # Init
    # =====================
    img_data = data.copy()
    min_plot_val = None
    max_plot_val = None
    if clim is not None:
        min_plot_val = clim[0]
        max_plot_val = clim[1]
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings(r'ignore', 'invalid value encountered in (less|greater)')
            img_data[img_data < min_plot_val] = min_plot_val
            img_data[img_data > max_plot_val] = max_plot_val
    if log10 is True:
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', 'invalid value encountered in less_equal')
            img_data[img_data <= 0] = np.NaN
        if np.isnan(img_data).all():
            logging.warning('log10 mode has been canceled because the input data was all composed of invalid values (NaN, negative')
            log10 = False

    figsize = ((pixels / dpi), (lines / dpi))
    if axes is True:
        figsize = (figsize[0] + 1.1, figsize[1] + 1.)
        x0 = 250
        y0 = 150
    if cbar is True:
        figsize = (figsize[0], figsize[1] + 0.7)
        y0 = y0 + dpi * 0.7
    if axes is False and title is not None:
        figsize = (figsize[0], figsize[1] + 0.3)


    fig = plt.figure(dpi=dpi)
    fig.set_size_inches(figsize[0], figsize[1])
    fig_w_pxls = fig.get_figwidth() * fig.get_dpi()
    fig_h_pxls = fig.get_figheight() * fig.get_dpi()

    if cmap is not None and cmap.lower() == 'wave':
        cmap = get_cmap_wave()
    elif cmap == 'Scat':
        pass
        # cmap = get_cmap_Scat()

    if geo_mode is True:
        if grid_interval is None:
            grid_interval = 3
    else:
        if grid_interval is None:
            grid_interval = 200

    # =====================
    # Draw image
    # =====================
    if log10 is True:
        img_data[np.isnan(img_data)] = 0
        im = fig.figimage(img_data, cmap=cmap, xo=x0, yo=y0, norm=LogNorm(vmin=min_plot_val, vmax=max_plot_val),
                          interpolation='none')
    else:
        im = fig.figimage(img_data, cmap=cmap, xo=x0, yo=y0, vmin=min_plot_val, vmax=max_plot_val, interpolation='none')

    rect = im.get_extent()
    im_size_h, im_size_w = im.get_size()

    if mask is not None:
        mask_data =mask.astype(np.float32) * 0.6
        mask_data[mask_data == 0] = np.NaN
        fig.figimage(mask_data, xo=x0, yo=y0, cmap='gray', clim=(0, 1), interpolation='none')

    # =====================
    # Draw axes
    # =====================
    ax = None
    if axes is True:
        ax = plt.axes([int(rect[0]) / fig_w_pxls, int(rect[2]) / fig_h_pxls, (im_size_w + 2) / fig_w_pxls,
                       (im_size_h + 2) / fig_h_pxls])
        ax.patch.set_alpha(0)

        if geo_mode is False:
            # set label and etc.
            if xlabel is None:
                xlabel = 'Pixel'
            if ylabel is None:
                ylabel = 'Line'

            ax.set_xlim(0, im_size_w)
            ax.set_xlabel(xlabel, size=12)
            ax.set_xticks(range(0, im_size_w, grid_interval))

            ax.set_ylim(0, im_size_h)
            ax.set_ylabel(ylabel, size=12)
            ax.set_yticks(range(0, lines, grid_interval))
            ax.set_yticklabels(range(0, lines, grid_interval), rotation=90)
            ax.invert_yaxis()

            if grid is True:
                ax.grid(color='gray', linestyle='--', linewidth=0.5)

        if geo_mode is True:
            if xlabel is None:
                xlabel = 'Longitude [degree]'
            if ylabel is None:
                ylabel = 'Latitude [degree]'

            lon = lon.copy()
            lat = lat.copy()
            if (lon[0,0] > lon[0,-1]) or (lon[-1,0] > lon[-1,-1]):
                lon[lon < 0] = 360. + lon[lon < 0]

            # Daw x-grid lines
            alpha = 1 if (grid is True) else 0
            lon_grid_min = math.floor(np.nanmin(lon))
            lon_grid_max = math.ceil(np.nanmax(lon))
            lon_grid = np.arange(lon_grid_min, lon_grid_max, grid_interval)
            if lon_grid[-1] + grid_interval == lon_grid_max:
                lon_grid = np.concatenate((lon_grid, [lon_grid_max]))
            lon_con = ax.contour(lon, origin='image', levels=lon_grid, colors='gray', linestyles='--', linewidths=0.5, alpha=alpha)

            # Draw x-labels and ticks
            ax.set_xlabel(xlabel, size=12)
            lon_tick_pos = []
            lon_tick_labels = []
            for level, polygon in zip(lon_con.levels, lon_con.allsegs):
                if len(polygon) < 1:
                    continue
                if polygon[-1][:, 1][0] > 1:
                    continue
                lon_tick_labels.append(level)
                lon_tick_pos.append(polygon[-1][:, 0][0])
            ax.set_xticks(lon_tick_pos)
            lon_tick_labels = map(lambda e: -360 + e if e > 180. else e, lon_tick_labels)
            lon_tick_labels = map(lambda e: '{0}°E'.format(e) if e >= 0 else '{0}°W'.format(e*-1), lon_tick_labels)
            ax.set_xticklabels(lon_tick_labels, ha="center")

            # Daw y-grid lines
            lat_grid_min = math.floor(np.nanmin(lat))
            lat_grid_max = math.ceil(np.nanmax(lat))
            lat_grid = np.arange(lat_grid_min, lat_grid_max, grid_interval)
            if lat_grid[-1] + grid_interval == lat_grid_max:
                lat_grid = np.concatenate((lat_grid, [lat_grid_max]))
            lat_con = ax.contour(lat, origin='image', levels=lat_grid, colors='gray', linestyles='--', linewidths=0.5, alpha=alpha)

            # Draw y-labels and ticks
            ax.set_ylabel(ylabel, size=12)
            lat_tick_pos = []
            lat_tick_labels = []
            for level, polygon in zip(lat_con.levels, lat_con.allsegs):
                if len(polygon) < 1:
                    continue
                if polygon[-1][-1][0] > 1:
                    continue
                lat_tick_labels.append(level)
                lat_tick_pos.append(polygon[-1][-1][1])
            ax.set_yticks(lat_tick_pos)
            lat_tick_labels = map(lambda e: '{0}°N'.format(e) if e >= 0 else '{0}°S'.format(e*-1), lat_tick_labels)
            ax.set_yticklabels(lat_tick_labels, rotation=90, va="center")

    # =====================
    # Draw title
    # =====================
    if title is not None:
        if ax is not None:
            ax.set_title(title, size=9)
        else:
            fig.text(0.5, (im_size_h + 10 + int(rect[2])) / fig_h_pxls, title, fontsize=9, horizontalalignment='center', verticalalignment='bottom')

    # =====================
    # Draw color-bar
    # =====================
    if cbar is True:
        cax = plt.axes([(int(rect[0])/fig_w_pxls)+0.05, 150/fig_h_pxls, (im_size_w/fig_w_pxls)-0.1, 50/fig_h_pxls])
        cax.tick_params(labelsize=10)

        if img_data.dtype != np.bool:
            cb = plt.colorbar(im, cax=cax, orientation='horizontal')
            if cbar_label is not None:
                cb.set_label(cbar_label, size=12)

    # =====================
    # Save
    # =====================
    plt.savefig(outpath)

    plt.close()

def get_cmap_wave():
    norm256 = np.mod(np.arange(0, 256, dtype=np.float32), 255.) / 255. * 2. * np.pi
    blue_wave = (np.sin(norm256) + 1.) / 2.
    red_wave = 1. - blue_wave

    norm16 = np.arange(0, 16, dtype=np.float32) / 16. * 2. * np.pi
    green_wave = (np.sin(norm16) + 1.) / 2. * 0.5
    blue = []
    red = []
    green = []
    for i, (b, r, g) in enumerate(zip(blue_wave, red_wave, list(green_wave) * 16)):
        blue.append((float(i/255), b, b))
        red.append((float(i/255), r, r))
        green.append((float(i/255), g, g))

    cdict = {
        'red': tuple(red),
        'green': tuple(green),
        'blue': tuple(blue),
    }

    return mat_colrs.LinearSegmentedColormap('wave', cdict)

# EOF
