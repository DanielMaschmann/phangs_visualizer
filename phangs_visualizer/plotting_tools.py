"""
This script gathers functions to help plotting procedures
"""

import os
from pathlib import Path
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.stats import SigmaClip
from astropy.visualization.wcsaxes import SphericalCircle

from matplotlib.colors import Normalize, LogNorm
from matplotlib.colorbar import ColorbarBase
from matplotlib import patheffects

import decimal

from photutils.segmentation import make_2dgaussian_kernel
from astropy.convolution import convolve

import numpy as np

from phangs_data_access import helper_func, phys_params, sample_access
import dust_tools


nuvb_label_dict = {
    1: {'offsets': [0.25, -0.1], 'ha': 'center', 'va': 'bottom', 'label': r'1,2,3 Myr'},
    5: {'offsets': [0.05, 0.1], 'ha': 'right', 'va': 'top', 'label': r'5 Myr'},
    10: {'offsets': [0.1, -0.2], 'ha': 'left', 'va': 'bottom', 'label': r'10 Myr'},
    100: {'offsets': [-0.1, 0.0], 'ha': 'right', 'va': 'center', 'label': r'100 Myr'}
}
ub_label_dict = {
    1: {'offsets': [0.2, -0.1], 'ha': 'center', 'va': 'bottom', 'label': r'1,2,3 Myr'},
    5: {'offsets': [0.05, 0.1], 'ha': 'right', 'va': 'top', 'label': r'5 Myr'},
    10: {'offsets': [0.1, -0.2], 'ha': 'left', 'va': 'bottom', 'label': r'10 Myr'},
    100: {'offsets': [-0.1, 0.0], 'ha': 'right', 'va': 'center', 'label': r'100 Myr'}
}
bv_label_dict = {
    1: {'offsets': [0.2, -0.1], 'ha': 'center', 'va': 'bottom', 'label': r'1,2,3 Myr'},
    5: {'offsets': [0.05, 0.1], 'ha': 'right', 'va': 'top', 'label': r'5 Myr'},
    10: {'offsets': [0.1, -0.1], 'ha': 'left', 'va': 'bottom', 'label': r'10 Myr'},
    100: {'offsets': [-0.1, 0.1], 'ha': 'right', 'va': 'center', 'label': r'100 Myr'}
}

nuvb_annotation_dict = {
    500: {'offset': [-0.5, +0.0], 'label': '500 Myr', 'ha': 'right', 'va': 'center'},
    1000: {'offset': [-0.7, +0.5], 'label': '1 Gyr', 'ha': 'right', 'va': 'center'},
    13750: {'offset': [+0.05, -0.9], 'label': '13.8 Gyr', 'ha': 'left', 'va': 'center'}
}
ub_annotation_dict = {
    500: {'offset': [-0.5, +0.0], 'label': '500 Myr', 'ha': 'right', 'va': 'center'},
    1000: {'offset': [-0.5, +0.5], 'label': '1 Gyr', 'ha': 'right', 'va': 'center'},
    13750: {'offset': [-0.0, -0.7], 'label': '13.8 Gyr', 'ha': 'left', 'va': 'center'}
}
bv_annotation_dict = {
    500: {'offset': [-0.5, +0.3], 'label': '500 Myr', 'ha': 'right', 'va': 'center'},
    1000: {'offset': [-0.5, +0.5], 'label': '1 Gyr', 'ha': 'right', 'va': 'center'},
    13750: {'offset': [-0.0, -0.7], 'label': '13.8 Gyr', 'ha': 'left', 'va': 'center'}
}


class WCSPlottingTools:
    """
    All functions related to WCS coordinate projects
    """
    @staticmethod
    def draw_box(ax, wcs, coord, box_size, color='k', line_width=2, line_style='-'):
        """
        function to draw a box around a coordinate on an axis with a WCS projection

        Parameters
        ----------
        ax : ``astropy.visualization.wcsaxes.core.WCSAxes``
        wcs : ``astropy.wcs.WCS``
        coord : ``astropy.coordinates.SkyCoord``
        box_size : tuple of float
            box size in arcsec
        color : str
        line_width : float
        line_style : str

        Returns
        -------
        None
        """
        if isinstance(box_size, tuple):
            box_size = box_size * u.arcsec
        elif isinstance(box_size, float) | isinstance(box_size, int):
            box_size = (box_size, box_size) * u.arcsec
        else:
            raise KeyError('cutout_size must be float or tuple')

        top_left_pix = wcs.world_to_pixel(SkyCoord(ra=coord.ra + (box_size[1] / 2) / np.cos(coord.dec.degree * np.pi / 180),
                                                   dec=coord.dec + (box_size[0] / 2)))
        top_right_pix = wcs.world_to_pixel(
            SkyCoord(ra=coord.ra - (box_size[1] / 2) / np.cos(coord.dec.degree * np.pi / 180),
                     dec=coord.dec + (box_size[0] / 2)))
        bottom_left_pix = wcs.world_to_pixel(
            SkyCoord(ra=coord.ra + (box_size[1] / 2) / np.cos(coord.dec.degree * np.pi / 180),
                     dec=coord.dec - (box_size[0] / 2)))
        bottom_right_pix = wcs.world_to_pixel(
            SkyCoord(ra=coord.ra - (box_size[1] / 2) / np.cos(coord.dec.degree * np.pi / 180),
                     dec=coord.dec - (box_size[0] / 2)))

        ax.plot([top_left_pix[0], top_right_pix[0]], [top_left_pix[1], top_right_pix[1]], color=color,
                linewidth=line_width, linestyle=line_style)
        ax.plot([bottom_left_pix[0], bottom_right_pix[0]], [bottom_left_pix[1], bottom_right_pix[1]], color=color,
                linewidth=line_width, linestyle=line_style)
        ax.plot([top_left_pix[0], bottom_left_pix[0]], [top_left_pix[1], bottom_left_pix[1]], color=color,
                linewidth=line_width, linestyle=line_style)
        ax.plot([top_right_pix[0], bottom_right_pix[0]], [top_right_pix[1], bottom_right_pix[1]], color=color,
                linewidth=line_width, linestyle=line_style)

    @staticmethod
    def plot_coord_circle(ax, pos, rad, color, line_style='-', line_width=3, alpha=1., fill=False):
        """
        function to draw circles around a coordinate on an axis with a WCS projection

        Parameters
        ----------
        ax : ``astropy.visualization.wcsaxes.core.WCSAxes``
        pos : ``astropy.coordinates.SkyCoord``
        rad : float
            circle_radius in arcsec
        color : str
        line_style: str
        line_width: float
        alpha: float
        fill: bool



        Returns
        -------
        None
        """

        if fill:
            face_color = color
        else:
            face_color = 'none'

        if isinstance(pos, list):
            if not isinstance(rad, list):
                rad = [rad] * len(pos)
            if not isinstance(color, list):
                color = [color] * len(pos)
            if not isinstance(line_style, list):
                line_style = [line_style] * len(pos)
            if not isinstance(line_width, list):
                line_width = [line_width] * len(pos)
            if not isinstance(alpha, list):
                alpha = [alpha] * len(pos)
            for pos_i, rad_i, color_i, line_style_i, line_width_i, alpha_i in zip(pos, rad, color, line_style, line_width,
                                                                                  alpha):
                circle = SphericalCircle(pos_i, rad_i * u.arcsec, edgecolor=color_i, facecolor=face_color,
                                         linewidth=line_width_i,
                                         linestyle=line_style_i, alpha=alpha_i, transform=ax.get_transform('fk5'))
                ax.add_patch(circle)
        else:
            circle = SphericalCircle(pos, rad * u.arcsec, edgecolor=color, facecolor=face_color, linewidth=line_width,
                                     linestyle=line_style, alpha=alpha, transform=ax.get_transform('fk5'))
            ax.add_patch(circle)

    @staticmethod
    def plot_coord_crosshair(ax, pos, wcs, rad, hair_length, color, line_style='-', line_width=3, alpha=1.):
        """
        function to draw crosshair around a coordinate on an axis with a WCS projection

        Parameters
        ----------
        ax : ``astropy.visualization.wcsaxes.core.WCSAxes``
        pos : ``astropy.coordinates.SkyCoord``
        wcs : ``astropy.wcs.WCS``
        hair_length : float
            length in arcseconds
        rad : float
            circle_radius in arcsec

        color : str
        line_style: str
        line_width: float
        alpha: float

        Returns
        -------
        None
        """

        pos_pix = wcs.world_to_pixel(pos)
        horizontal_rad = helper_func.CoordTools.transform_world2pix_scale(length_in_arcsec=rad, wcs=wcs)
        vertical_rad = helper_func.CoordTools.transform_world2pix_scale(length_in_arcsec=rad, wcs=wcs, dim=1)
        horizontal_hair_length = helper_func.CoordTools.transform_world2pix_scale(length_in_arcsec=hair_length, wcs=wcs)
        vertical_hair_length = helper_func.CoordTools.transform_world2pix_scale(length_in_arcsec=hair_length, wcs=wcs,
                                                                                dim=1)

        ax.plot([pos_pix[0] + horizontal_rad, pos_pix[0] + horizontal_rad + horizontal_hair_length],
                [pos_pix[1], pos_pix[1]], color=color, linestyle=line_style, linewidth=line_width, alpha=alpha)
        ax.plot([pos_pix[0] - horizontal_rad, pos_pix[0] - horizontal_rad - horizontal_hair_length],
                [pos_pix[1], pos_pix[1]], color=color, linestyle=line_style, linewidth=line_width, alpha=alpha)
        ax.plot([pos_pix[0], pos_pix[0]], [pos_pix[1] + vertical_rad, pos_pix[1] + vertical_rad + vertical_hair_length],
                color=color, linestyle=line_style, linewidth=line_width, alpha=alpha)
        ax.plot([pos_pix[0], pos_pix[0]], [pos_pix[1] - vertical_rad, pos_pix[1] - vertical_rad - vertical_hair_length],
                color=color, linestyle=line_style, linewidth=line_width, alpha=alpha)

    @staticmethod
    def plot_img_scale_bar(ax, img_shape, wcs, bar_length=1, length_unit='kpc', target_dist_mpc=None,
                           phangs_target=None, bar_color='white', text_color='white', line_width=4, fontsize=30,
                           va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01):
        """
        function to display a scale bar on an WCS image

        Parameters
        ----------
        ax : ``astropy.visualization.wcsaxes.core.WCSAxes``
        img_shape : tuple
        wcs : ``astropy.wcs.WCS``
        bar_length : int or float
        length_unit : str
        target_dist_mpc : float
            default None
        phangs_target : str
            default None
        bar_color, text_color : str
            colors for the plotted objects
        line_width : int or float
        fontsize : int or float
        va, ha : str
            specifying the position of the bar
        x_offset, y_offset : float
            specifying the percentual offset to the image frame
        text_y_offset_diff : float
            percentual offset for the text above the bar

        Returns
        -------
        None
        """

        # get distance to target
        if (((phangs_target is None) & (target_dist_mpc is None)) |
                ((phangs_target is not None) & (target_dist_mpc is not None))):
            raise KeyError('To get the distance you need either to provide the target distance or a PHANGS target '
                           'name. It is also not possible to provide both due to ambiguity. ')
        if phangs_target is not None:
            phangs_sample = sample_access.SampleAccess()
            target_dist_mpc = phangs_sample.get_target_dist(target=phangs_target)

        if length_unit == 'pc':
            bar_length_in_arcsec = helper_func.CoordTools.kpc2arcsec(diameter_kpc=bar_length * 1e-3,
                                                                     target_dist_mpc=target_dist_mpc)
        elif length_unit == 'kpc':
            bar_length_in_arcsec = helper_func.CoordTools.kpc2arcsec(diameter_kpc=bar_length,
                                                                     target_dist_mpc=target_dist_mpc)
        elif length_unit == 'Mpc':
            bar_length_in_arcsec = helper_func.CoordTools.kpc2arcsec(diameter_kpc=bar_length * 1e3,
                                                                     target_dist_mpc=target_dist_mpc)
        else:
            raise KeyError('length_unit must be either pc, kpc or Mpc but was given to be: ', length_unit)
        bar_length_in_pixel = helper_func.CoordTools.transform_world2pix_scale(length_in_arcsec=bar_length_in_arcsec,
                                                                               wcs=wcs, dim=0)

        if isinstance(bar_length, int):
            length_str = str(bar_length) + ' ' + length_unit
        elif isinstance(bar_length, float):
            length_str = StrTools.float2str(f=bar_length) + ' ' + length_unit
        else:
            raise KeyError(' bar_length must be either int or float!')

        # position ing the bar
        assert va in ['bottom', 'top']
        assert ha in ['left', 'right']

        if ha == 'left':
            pos_left = x_offset * img_shape[0]
        else:
            pos_left = img_shape[0] - (x_offset * img_shape[0] + bar_length_in_pixel)
        # text position is just relative to left bar position
        text_pos_x = pos_left + bar_length_in_pixel/2

        if va == 'bottom':
            pos_bottom = y_offset * img_shape[1]
        else:
            pos_bottom = img_shape[1] - (y_offset * img_shape[1])

        text_pos_y = pos_bottom + text_y_offset_diff * img_shape[1]

        ax.plot([pos_left, pos_left+bar_length_in_pixel], [pos_bottom, pos_bottom], linewidth=line_width,
                color=bar_color)
        ax.text(text_pos_x, text_pos_y, length_str, horizontalalignment='center', verticalalignment='bottom',
                color=text_color, fontsize=fontsize)

    @staticmethod
    def arr_axis_params(ax, ra_tick_label=True, dec_tick_label=True,
                        ra_axis_label='R.A. (2000.0)', dec_axis_label='DEC. (2000.0)',
                        ra_minpad=0.8, dec_minpad=0.8, tick_color='k', label_color='k',
                        fontsize=15., labelsize=14., ra_tick_num=None, dec_tick_num=None,
                        ra_minor_ticks=True, dec_minor_ticks=True):
        """
        plots circle on image using coordinates and WCS to orientate

        Parameters
        ----------
        ax : ``astropy.visualization.wcsaxes.core.WCSAxes``
            axis for plotting
        ra_tick_label, dec_tick_label : bool
        ra_axis_label, dec_axis_label : str
        ra_minpad, dec_minpad : float
        tick_color, label_color : str
        fontsize, labelsize : float
        ra_tick_num, dec_tick_num : int
        ra_minor_ticks, dec_minor_ticks : bool

        Returns
        -------
        None
        """
        ax.tick_params(which='both', width=1.5, length=7, direction='in', color=tick_color, labelsize=labelsize)

        if not ra_tick_label:
            ax.coords['ra'].set_ticklabel_visible(False)
            ax.coords['ra'].set_axislabel(' ', color=label_color)
        else:
            ax.coords['ra'].set_ticklabel(rotation=0, color=label_color)
            ax.coords['ra'].set_axislabel(ra_axis_label, minpad=ra_minpad, color=label_color, fontsize=fontsize)

        if not dec_tick_label:
            ax.coords['dec'].set_ticklabel_visible(False)
            ax.coords['dec'].set_axislabel(' ', color=label_color)
        else:
            ax.coords['dec'].set_ticklabel(rotation=90, color=label_color)
            ax.coords['dec'].set_axislabel(dec_axis_label, minpad=dec_minpad, color=label_color, fontsize=fontsize)

        if ra_tick_num is not None:
            ax.coords['ra'].set_ticks(number=ra_tick_num)
        if ra_minor_ticks:
            ax.coords['ra'].display_minor_ticks(True)
        if dec_tick_num is not None:
            ax.coords['dec'].set_ticks(number=dec_tick_num)
        if dec_minor_ticks:
            ax.coords['dec'].display_minor_ticks(True)



class CCDPlottingTools:
    """
    all functions for Color-color diagram visualization
    """
    @staticmethod
    def gauss2d(x, y, x0, y0, sig_x, sig_y):
        """
        2D Gaussian function
        """
        expo = -(((x - x0)**2)/(2 * sig_x**2) + ((y - y0)**2)/(2 * sig_y**2))
        norm_amp = 1 / (2 * np.pi * sig_x * sig_y)
        return norm_amp * np.exp(expo)

    @staticmethod
    def calc_gauss_weight_map(x_data, y_data, x_data_err, y_data_err, x_lim, y_lim, n_x_bins, n_y_bins, norm_map=True,
                              gauss_conv=True, kernel_size=9, kernel_std=4.0):
        """
        calculate Gaussian weighted 2D map of data. the uncertainties are used as the std of the Gaussian.
        """
        # bins
        x_bins_gauss = np.linspace(x_lim[0], x_lim[1], n_x_bins)
        y_bins_gauss = np.linspace(y_lim[0], y_lim[1], n_y_bins)
        # get a mesh
        x_mesh, y_mesh = np.meshgrid(x_bins_gauss, y_bins_gauss)
        gauss_map = np.zeros((len(x_bins_gauss), len(y_bins_gauss)))

        for idx in range(len(x_data)):
            x_err = np.sqrt(x_data_err[idx]**2 + 0.01**2)
            y_err = np.sqrt(y_data_err[idx]**2 + 0.01**2)
            gauss = CCDPlottingTools.gauss2d(x=x_mesh, y=y_mesh, x0=x_data[idx], y0=y_data[idx],
                                             sig_x=x_err, sig_y=y_err)
            gauss_map += gauss

        if gauss_conv:
            kernel = make_2dgaussian_kernel(kernel_std, size=kernel_size)
            conv_gauss_map = convolve(gauss_map, kernel)
            if norm_map:
                return conv_gauss_map / np.sum(conv_gauss_map)
            else:
                return conv_gauss_map
        else:
            if norm_map:
                return gauss_map / np.sum(norm_map)
            else:
                return gauss_map

    @staticmethod
    def display_models(ax, y_color='ub',
                       x_color='vi',
                   age_cut_sol50=5e2,
                   age_dots_sol=None,
                   age_dots_sol50=None,
                   age_labels=False,
                   age_label_color='red',
                   age_label_fontsize=30,
                   color_sol='tab:cyan', linewidth_sol=4, linestyle_sol='-',
                   color_sol50='m', linewidth_sol50=4, linestyle_sol50='-',
                   label_sol=None, label_sol50=None):

        package_dir_path = os.path.dirname(os.path.dirname(__file__))
        x_model_sol = np.load(Path(package_dir_path) / 'data' / ('model_%s_sol.npy' % x_color))
        x_model_sol50 = np.load(Path(package_dir_path) / 'data' / ('model_%s_sol50.npy' % x_color))

        y_model_sol = np.load(Path(package_dir_path) / 'data' / ('model_%s_sol.npy' % y_color))
        y_model_sol50 = np.load(Path(package_dir_path) / 'data' / ('model_%s_sol50.npy' % y_color))

        age_mod_sol = np.load(Path(package_dir_path) / 'data' / 'age_mod_sol.npy')
        age_mod_sol50 = np.load(Path(package_dir_path) / 'data' / 'age_mod_sol50.npy')

        ax.plot(x_model_sol, y_model_sol, color=color_sol, linewidth=linewidth_sol, linestyle=linestyle_sol, zorder=10,
                label=label_sol)
        ax.plot(x_model_sol50[age_mod_sol50 > age_cut_sol50], y_model_sol50[age_mod_sol50 > age_cut_sol50],
                color=color_sol50, linewidth=linewidth_sol50, linestyle=linestyle_sol50, zorder=10, label=label_sol50)

        if age_dots_sol is None:
            age_dots_sol = [1, 5, 10, 100, 500, 1000, 13750]
        for age in age_dots_sol:
            ax.scatter(x_model_sol[age_mod_sol == age], y_model_sol[age_mod_sol == age], color='b', s=80, zorder=20)

        if age_dots_sol50 is None:
            age_dots_sol50 = [500, 1000, 13750]
        for age in age_dots_sol50:
            ax.scatter(x_model_sol50[age_mod_sol50 == age], y_model_sol50[age_mod_sol50 == age], color='tab:pink', s=80, zorder=20)

        if age_labels:
            label_dict = globals()['%s_label_dict' % y_color]
            pe = [patheffects.withStroke(linewidth=3, foreground="w")]
            for age in label_dict.keys():

                ax.text(x_model_sol[age_mod_sol == age]+label_dict[age]['offsets'][0],
                        y_model_sol[age_mod_sol == age]+label_dict[age]['offsets'][1],
                        label_dict[age]['label'], horizontalalignment=label_dict[age]['ha'], verticalalignment=label_dict[age]['va'],
                        color=age_label_color, fontsize=age_label_fontsize,
                        path_effects=pe)

            annotation_dict = globals()['%s_annotation_dict' % y_color]
            for age in annotation_dict.keys():

                txt_sol = ax.annotate(' ', #annotation_dict[age]['label'],
                            xy=(x_model_sol[age_mod_sol == age], y_model_sol[age_mod_sol == age]),
                            xytext=(x_model_sol[age_mod_sol == age]+annotation_dict[age]['offset'][0],
                                    y_model_sol[age_mod_sol == age]+annotation_dict[age]['offset'][1]),
                            fontsize=age_label_fontsize, xycoords='data', textcoords='data', color=age_label_color,
                            ha=annotation_dict[age]['ha'], va=annotation_dict[age]['va'], zorder=30,
                                  arrowprops=dict(arrowstyle='-|>', color='darkcyan', lw=3, ls='-'),
                            path_effects=[patheffects.withStroke(linewidth=3,
                                                            foreground="w")])
                txt_sol.arrow_patch.set_path_effects([patheffects.Stroke(linewidth=5, foreground="w"),
                                                      patheffects.Normal()])
                txt_sol50 = ax.annotate(' ',
                            xy=(x_model_sol50[age_mod_sol50 == age], y_model_sol50[age_mod_sol50 == age]),
                            xytext=(x_model_sol[age_mod_sol == age]+annotation_dict[age]['offset'][0],
                                    y_model_sol[age_mod_sol == age]+annotation_dict[age]['offset'][1]),
                            fontsize=age_label_fontsize, xycoords='data', textcoords='data',
                            ha=annotation_dict[age]['ha'], va=annotation_dict[age]['va'], zorder=30,
                                  arrowprops=dict(arrowstyle='-|>', color='darkviolet', lw=3, ls='-'),
                            path_effects=[patheffects.withStroke(linewidth=3,
                                                            foreground="w")])
                txt_sol50.arrow_patch.set_path_effects([patheffects.Stroke(linewidth=5, foreground="w"),
                                                      patheffects.Normal()])
                ax.text(x_model_sol[age_mod_sol == age]+annotation_dict[age]['offset'][0],
                        y_model_sol[age_mod_sol == age]+annotation_dict[age]['offset'][1],
                        annotation_dict[age]['label'],
                        horizontalalignment=annotation_dict[age]['ha'], verticalalignment=annotation_dict[age]['va'],
                        color=age_label_color, fontsize=age_label_fontsize, zorder=40, path_effects=pe)

    @staticmethod
    def plot_reddening_vect(ax, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                        x_color_int=0, y_color_int=0, av_val=1,
                        linewidth=2, line_color='k',
                        text=False, fontsize=20, text_color='k', x_text_offset=0.1, y_text_offset=-0.3):

        nuv_wave = phys_params.hst_wfc3_uvis1_bands_wave['F275W']['mean_wave']*1e-4
        u_wave = phys_params.hst_wfc3_uvis1_bands_wave['F336W']['mean_wave']*1e-4
        b_wave = phys_params.hst_wfc3_uvis1_bands_wave['F438W']['mean_wave']*1e-4
        v_wave = phys_params.hst_wfc3_uvis1_bands_wave['F555W']['mean_wave']*1e-4
        i_wave = phys_params.hst_wfc3_uvis1_bands_wave['F814W']['mean_wave']*1e-4

        x_wave_1 = locals()[x_color_1 + '_wave']
        x_wave_2 = locals()[x_color_2 + '_wave']
        y_wave_1 = locals()[y_color_1 + '_wave']
        y_wave_2 = locals()[y_color_2 + '_wave']

        color_ext_x = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=x_wave_1, wave2=x_wave_2, av=av_val)
        color_ext_y = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=y_wave_1, wave2=y_wave_2, av=av_val)

        slope_av_vector = ((y_color_int + color_ext_y) - y_color_int) / ((x_color_int + color_ext_x) - x_color_int)

        angle_av_vector = np.arctan(color_ext_y/color_ext_x) * 180/np.pi

        ax.annotate('', xy=(x_color_int + color_ext_x, y_color_int + color_ext_y), xycoords='data',
                    xytext=(x_color_int, y_color_int), fontsize=fontsize,
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color=line_color, lw=linewidth, ls='-'))

        if text:
            if isinstance(av_val, int):
                arrow_text = r'A$_{\rm V}$=%i mag' % av_val
            else:
                arrow_text = r'A$_{\rm V}$=%.1f mag' % av_val
            ax.text(x_color_int + x_text_offset, y_color_int + y_text_offset, arrow_text,
                    horizontalalignment='left', verticalalignment='bottom',
                    transform_rotates_text=True, rotation_mode='anchor',
                    rotation=angle_av_vector, fontsize=fontsize, color=text_color)


class StrTools:
    """
    basic class to gather handling of strings and other things for displaying text
    """
    @staticmethod
    def float2str(f, max_digits=20):
        """
        Convert the given float to a string,
        without resorting to scientific notation

        Parameters
        ----------

        f : float
        max_digits : int
        Returns
        -------
        float_in_str: str
        """

        # create a new context for this task
        ctx = decimal.Context()

        ctx.prec = max_digits


        d1 = ctx.create_decimal(repr(f))
        return format(d1, 'f')

    @staticmethod
    def display_text_in_corner(ax, text, fontsize, text_color='k', x_frac=0.02, y_frac=0.98, horizontalalignment='left',
                               verticalalignment='top', path_eff=True, path_err_linewidth=3, path_eff_color='white'):
        if path_eff:
            pe = [patheffects.withStroke(linewidth=path_err_linewidth, foreground=path_eff_color)]
        else:
            pe = None
        ax.text(x_frac, y_frac, text, horizontalalignment=horizontalalignment, verticalalignment=verticalalignment,
                fontsize=fontsize, color=text_color, transform=ax.transAxes, path_effects=pe)

