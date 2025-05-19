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
import matplotlib.pyplot as plt


import decimal

from photutils.segmentation import make_2dgaussian_kernel
from photutils.background import Background2D, MedianBackground
from astropy.stats import sigma_clipped_stats
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import SqrtStretch

from astropy.convolution import convolve

from regions import PixCoord, RectanglePixelRegion

import numpy as np
from setuptools.command.rotate import rotate

import phangs_visualizer.multicolorfits as mcf

from phangs_data_access import helper_func, phys_params, sample_access
import dust_tools
from phangs_data_access.helper_func import ObsTools
from phangs_data_access import spec_tools
from phangs_data_access.spec_tools import SpecTools


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


# define a color seelction
color_list_dark2 = plt.get_cmap('Dark2')(range(8))
color_list_set2 = plt.get_cmap('Set2')(range(8))
color_list_set3 = plt.get_cmap('Set3')(range(12))
color_list_paired = plt.get_cmap('Paired')(range(12))
color_list_tab10 = plt.get_cmap('tab10')(range(10))
color_list_pastel2 = plt.get_cmap('Pastel2')(range(8))

color_list_rainbow = plt.get_cmap('rainbow')





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
                           va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01,
                           path_eff=True, path_eff_color='white'):
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
        if ((((phangs_target is None) & (target_dist_mpc is None)) |
                ((phangs_target is not None) & (target_dist_mpc is not None))) &
                (length_unit in ['pc', 'kpc', 'Mpc'])):
            raise KeyError('To get the distance you need either to provide the target distance or a PHANGS target '
                           'name. It is also not possible to provide both due to ambiguity. ')
        if phangs_target is not None:
            phangs_sample = sample_access.SampleAccess()
            target_dist_mpc = phangs_sample.get_target_dist(
                target=helper_func.FileTools.target_name_no_directions(target=phangs_target))

        if length_unit == 'pc':
            bar_length_in_arcsec = helper_func.CoordTools.kpc2arcsec(diameter_kpc=bar_length * 1e-3,
                                                                     target_dist_mpc=target_dist_mpc)
        elif length_unit == 'kpc':
            bar_length_in_arcsec = helper_func.CoordTools.kpc2arcsec(diameter_kpc=bar_length,
                                                                     target_dist_mpc=target_dist_mpc)
        elif length_unit == 'Mpc':
            bar_length_in_arcsec = helper_func.CoordTools.kpc2arcsec(diameter_kpc=bar_length * 1e3,
                                                                     target_dist_mpc=target_dist_mpc)
        elif length_unit == 'arcsec':
            bar_length_in_arcsec = bar_length
        elif length_unit == 'arcmin':
            bar_length_in_arcsec = bar_length * 60
        elif length_unit == 'deg':
            bar_length_in_arcsec = bar_length * 3600
        else:
            raise KeyError('length_unit must be either pc, kpc or Mpc but was given to be: ', length_unit)


        bar_length_in_pixel = helper_func.CoordTools.transform_world2pix_scale(length_in_arcsec=bar_length_in_arcsec,
                                                                               wcs=wcs, dim=0)

        if length_unit in ['pc', 'kpc', 'Mpc']:
            length_unit_str = length_unit
        elif length_unit in ['arcsec', 'arcmin', 'deg']:
            if length_unit == 'arcsec':
                length_unit_str = '\"'
            elif length_unit == 'arcmin':
                length_unit_str = '\''
            elif length_unit == 'deg':
                length_unit_str = r'$^{\rm o}$'
            else:
                length_unit_str = None
        else:
            raise KeyError('length unit musst be: pc, kpc, Mpc, arcsec, arcmin, deg  but got ', length_unit)

        if isinstance(bar_length, int):
            length_str = str(bar_length) + length_unit_str
        elif isinstance(bar_length, float):
            length_str = StrTools.float2str(f=bar_length) + length_unit_str
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
        StrTools.display_text_on_data_point(ax=ax, text=length_str,
                                   x_data_point=text_pos_x, y_data_point=text_pos_y,
                                   x_axis_frac_offset=0., y_axis_frac_offset=0.0,
                                              x_scale_log=False, y_scale_log=False,
                                   fontsize=fontsize, text_color=text_color,
                                   horizontal_alignment='center',
                                   vertical_alignment='bottom',
                                   path_eff=path_eff, path_err_linewidth=2, path_eff_color=path_eff_color, rotation=0,
                                            rotation_mode='anchor', transform_rotates_text=True)
        #
        # ax.text(text_pos_x, text_pos_y, length_str, horizontalalignment='center', verticalalignment='bottom',
        #          color=text_color, fontsize=fontsize)

    @staticmethod
    def arr_axis_params(ax, ra_tick_label=True, dec_tick_label=True,
                        ra_axis_label='R.A. (2000.0)', dec_axis_label='DEC. (2000.0)',
                        ra_label_ha='center', dec_label_va='center',
                        ra_minpad=0.8, dec_minpad=0.8, ra_tick_color='k', ra_label_color='k',
                        dec_tick_color='k', dec_label_color='k',
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
        ax.tick_params(which='ra', width=1.5, length=7, direction='in', color=ra_tick_color, labelsize=labelsize)
        ax.tick_params(which='dec', width=1.5, length=7, direction='in', color=dec_tick_color, labelsize=labelsize)

        if not ra_tick_label:
            ax.coords['ra'].set_ticklabel_visible(False)
            ax.coords['ra'].set_axislabel(' ', color=ra_label_color)
        else:
            ax.coords['ra'].set_ticklabel(rotation=0, color=ra_label_color)
            ax.coords['ra'].set_axislabel(ra_axis_label, minpad=ra_minpad, color=ra_label_color, ha=ra_label_ha, fontsize=fontsize)
        if not dec_tick_label:
            ax.coords['dec'].set_ticklabel_visible(False)
            ax.coords['dec'].set_axislabel(' ', color=dec_label_color)
        else:
            ax.coords['dec'].set_ticklabel(rotation=90, color=dec_label_color)
            ax.coords['dec'].set_axislabel(dec_axis_label, minpad=dec_minpad, color=dec_label_color, va=dec_label_va, fontsize=fontsize)

        if ra_tick_num is not None:
            ax.coords['ra'].set_ticks(number=ra_tick_num)
        if ra_minor_ticks:
            ax.coords['ra'].display_minor_ticks(True)
        if dec_tick_num is not None:
            ax.coords['dec'].set_ticks(number=dec_tick_num)
        if dec_minor_ticks:
            ax.coords['dec'].display_minor_ticks(True)

    @staticmethod
    def plot_slit(ax, coords_slit_pix, slit_length, slit_width, slit_pa, plot_scatter=True, color='red', lw=2,
                  linestyle='-', offset=90):
        x_cent = float(coords_slit_pix[0])
        y_cent = float(coords_slit_pix[1])

        reg = RectanglePixelRegion(PixCoord(x=x_cent, y=y_cent), width=slit_length,
                                   height=slit_width, angle=(slit_pa + offset) * u.deg)
        reg.plot(ax=ax, edgecolor=color, linewidth=lw, linestyle=linestyle)
        if plot_scatter:
            ax.scatter(coords_slit_pix[0], coords_slit_pix[1], c='r', s=120)


class AxisTools:
    """
    Simple function to create figures and add axis etc
    """
    @staticmethod
    def init_fig(fig_dict):
        return plt.figure(figsize=fig_dict['fig_size'])

    @staticmethod
    def add_panel_axis(fig, left_align, bottom_align, width, height, space_vertical, space_horizontal,
                       row_idx, col_idx, projection=None):
        return fig.add_axes([left_align + (width + space_vertical)*col_idx,
                             bottom_align + (height + space_horizontal)*row_idx,
                             width, height],
                            projection=projection)


class ImgTools:
    """
    Tools to plot all kind of images like RGB or scaled images
    """

    @staticmethod
    def get_rgb_img(data_r=None, data_g=None, data_b=None, color_r='#FF4433', color_g='#0FFF50', color_b='#1F51FF',
                    min_max_r=None, min_max_g=None, min_max_b=None,
                    rescalefn='asinh',
                    scaletype_r='perc', scaletype_g='perc', scaletype_b='perc',
                    gamma_r=2.2, gamma_g=2.2, gamma_b=2.2,
                    gamma_corr_r=2.2, gamma_corr_g=2.2, gamma_corr_b=2.2, combined_gamma=2.2,
                    inverse=False):
        """
        Function to create an RGB image

        Parameters
        ----------
        data_r, data_g, data_b : ``numpy.ndarray``or array
            color images. Must be all same shape
        color_r, color_g, color_b: str
            hex code for color
        min_max_r, min_max_g, min_max_b : tuple or None
            denotes the percentages till where the data is used
        rescalefn : str
            scale function can be linear sqrt squared log power sinh asinh
        scaletype_r, scaletype_g, scaletype_b : str
        'abs' for absolute values, 'perc' for percentiles
        gamma_r, gamma_g, gamma_b : float
            gamma factor for each individual color band
        gamma_corr_r, gamma_corr_g, gamma_corr_b : float
            gamma correction factor for each grey scale image
        combined_gamma : float
            gamma factor of resulting rgb image

        Returns
        -------
        rgb image : ``numpy.ndarray``
            of shape (N,N, 3)
        """
        if min_max_r is None:
            min_max_r = [0., 99.7]
        if min_max_g is None:
            min_max_g = [0., 99.7]
        if min_max_b is None:
            min_max_b = [0., 99.7]

        color_list = []
        if data_r is not None:
            grey_r = mcf.greyRGBize_image(data_r, rescalefn=rescalefn, scaletype=scaletype_r, min_max=min_max_r,
                                          gamma=gamma_r)
            color_list.append(mcf.colorize_image(grey_r, color_r, colorintype='hex', gammacorr_color=gamma_corr_r))
        if data_g is not None:
            grey_g = mcf.greyRGBize_image(data_g, rescalefn=rescalefn, scaletype=scaletype_g, min_max=min_max_g,
                                          gamma=gamma_g)
            color_list.append(mcf.colorize_image(grey_g, color_g, colorintype='hex', gammacorr_color=gamma_corr_g))
        if data_b is not None:
            grey_b = mcf.greyRGBize_image(data_b, rescalefn=rescalefn, scaletype=scaletype_b, min_max=min_max_b,
                                          gamma=gamma_b)
            color_list.append(mcf.colorize_image(grey_b, color_b, colorintype='hex', gammacorr_color=gamma_corr_b))


        return mcf.combine_multicolor(color_list, gamma=combined_gamma, inverse=inverse)

    @staticmethod
    def get_4color_img(data1=None, data2=None, data3=None, data4=None,
                       color1='#FF4433', color2='#0FFF50', color3='#1F51FF', color4='#1F51FF',
                    min_max1=None, min_max2=None, min_max3=None, min_max4=None,
                    rescalefn='asinh',
                    scaletype1='perc', scaletype2='perc', scaletype3='perc', scaletype4='perc',
                    gamma1=2.2, gamma2=2.2, gamma3=2.2, gamma4=2.2,
                    gamma_corr1=2.2, gamma_corr2=2.2, gamma_corr3=2.2, gamma_corr4=2.2,
                       combined_gamma=2.2,
                    inverse=False):
        """
        Function to create an RGB image

        Parameters
        ----------
        data1, data2, data3 : ``numpy.ndarray``or array
            color images. Must be all same shape
        color1, color2, color3: str
            hex code for color
        min_max1, min_max2, min_max3 : tuple or None
            denotes the percentages till where the data is used
        rescalefn : str
            scale function can be linear sqrt squared log power sinh asinh
        scaletype1, scaletype2, scaletype3 : str
        'abs' for absolute values, 'perc' for percentiles
        gamma1, gamma2, gamma3 : float
            gamma factor for each individual color band
        gamma_corr1, gamma_corr2, gamma_corr3 : float
            gamma correction factor for each grey scale image
        combined_gamma : float
            gamma factor of resulting rgb image

        Returns
        -------
        rgb image : ``numpy.ndarray``
            of shape (N,N, 3)
        """
        if min_max1 is None:
            min_max1 = [0., 99.7]
        if min_max2 is None:
            min_max2 = [0., 99.7]
        if min_max3 is None:
            min_max3 = [0., 99.7]
        if min_max4 is None:
            min_max4 = [0., 99.7]

        color_list = []
        if data1 is not None:
            grey1 = mcf.greyRGBize_image(data1, rescalefn=rescalefn, scaletype=scaletype1, min_max=min_max1,
                                          gamma=gamma1)
            color_list.append(mcf.colorize_image(grey1, color1, colorintype='hex', gammacorr_color=gamma_corr1))
        if data2 is not None:
            grey2 = mcf.greyRGBize_image(data2, rescalefn=rescalefn, scaletype=scaletype2, min_max=min_max2,
                                          gamma=gamma2)
            color_list.append(mcf.colorize_image(grey2, color2, colorintype='hex', gammacorr_color=gamma_corr2))
        if data3 is not None:
            grey3 = mcf.greyRGBize_image(data3, rescalefn=rescalefn, scaletype=scaletype3, min_max=min_max3,
                                          gamma=gamma3)
            color_list.append(mcf.colorize_image(grey3, color3, colorintype='hex', gammacorr_color=gamma_corr3))

        if data4 is not None:
            grey4 = mcf.greyRGBize_image(data4, rescalefn=rescalefn, scaletype=scaletype4, min_max=min_max4,
                                          gamma=gamma4)
            color_list.append(mcf.colorize_image(grey4, color4, colorintype='hex', gammacorr_color=gamma_corr4))

        return mcf.combine_multicolor(color_list, gamma=combined_gamma, inverse=inverse)


class CCDTools:
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
        x_bins_gauss_center = (x_bins_gauss[1:] + x_bins_gauss[:-1]) / 2
        y_bins_gauss_center = (y_bins_gauss[1:] + y_bins_gauss[:-1]) / 2

        # get a mesh
        x_mesh, y_mesh = np.meshgrid(x_bins_gauss_center, y_bins_gauss_center)

        gauss_map = np.zeros((len(x_bins_gauss_center), len(y_bins_gauss_center)))

        for idx in range(len(x_data)):
            x_err = np.sqrt(x_data_err[idx]**2 + 0.01**2)
            y_err = np.sqrt(y_data_err[idx]**2 + 0.01**2)
            gauss = CCDTools.gauss2d(x=x_mesh, y=y_mesh, x0=x_data[idx], y0=y_data[idx],
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
    def display_ccd_model(ax, x_model, y_model, age_model=None, model_color='darkred', model_line_style='-',
                          model_line_width=5, model_label=None,
                          mark_age=None, mark_age_color='darkred', marker_size=200,
                          label_age=None, label_offset=None, text_color_label='k', font_size_label=30, ha_label=None, va_label=None,

                          label_age_arrow=None, arrow_pos_offset=None, arrow_head_offset=None, arrow_color='k', text_color_arrow='k', arrow_line_width=3, fontsize_label_arrow=30, ha_label_arrow=None, va_label_arrow=None,):
        # plot
        ax.plot(x_model, y_model, color=model_color, linestyle=model_line_style, linewidth=model_line_width, label=model_label)
        if mark_age is not None:
            for age in mark_age:
                ax.scatter(x_model[age_model == age], y_model[age_model == age], color=mark_age_color, s=marker_size)
        if label_age is not None:
            if ha_label is None:
                ha_label = ['center'] * len(label_age)
            if va_label is None:
                va_label = ['center'] * len(label_age)
            if label_offset is None:
                label_offset = [(0, 0)] * len(label_age)

            for age, offset, ha, va in zip(label_age, label_offset, ha_label, va_label):
                age_str = StrTools.age2label(age=age)
                StrTools.display_text_on_data_point(ax=ax, text=age_str, x_data_point=x_model[age_model == age],
                                                    y_data_point=y_model[age_model == age],
                                                    x_axis_frac_offset=offset[0], y_axis_frac_offset=offset[1],
                                                    fontsize=font_size_label, text_color=text_color_label,
                                                    horizontal_alignment=ha, vertical_alignment=va,
                                                    path_eff=True, path_err_linewidth=3, path_eff_color='white', )

        if label_age_arrow is not None:
            if arrow_head_offset is None:
                arrow_head_offset = [(0, 0)] * len(label_age_arrow)
            for age, pos_offset, head_offset, ha, va in zip(label_age_arrow, arrow_pos_offset, arrow_head_offset,
                                                            ha_label_arrow, va_label_arrow):
                age_str = StrTools.age2label(age=age)
                ax.annotate(age_str, xy=(x_model[age_model == age] + head_offset[0],
                                         y_model[age_model == age] + head_offset[1]),
                            xytext=(x_model[age_model == age] + pos_offset[0],
                                    y_model[age_model == age] + pos_offset[1]),
                            fontsize=fontsize_label_arrow, xycoords='data', textcoords='data', color=arrow_color,
                            ha=ha, va=va, zorder=30, arrowprops=dict(arrowstyle='-|>', color=text_color_arrow,
                                                                     lw=arrow_line_width, ls='-'),
                            path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])

                # txt_sol.arrow_patch.set_path_effects([patheffects.Stroke(linewidth=5, foreground="w"),
                #                                       patheffects.Normal()])
                # ax.text(x_model_sol[age_mod_sol == age]+annotation_dict[age]['offset'][0],
                #                         y_model_sol[age_mod_sol == age]+annotation_dict[age]['offset'][1],
                #                         annotation_dict[age]['label'],
                #                         horizontalalignment=annotation_dict[age]['ha'], verticalalignment=annotation_dict[age]['va'],
                #                         color=age_label_color, fontsize=age_label_fontsize, zorder=40, path_effects=pe)

    @staticmethod
    def plot_reddening_vect(ax, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                        x_color_int=0, y_color_int=0, av_val=1,
                        linewidth=2, line_color='k',
                        text=False, fontsize=20, text_color='k', x_text_offset=0.01, y_text_offset=-0.01):

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

            StrTools.display_text_on_data_point(ax=ax, text=arrow_text,
                                                x_data_point=x_color_int + color_ext_x/2,
                                                y_data_point= y_color_int + color_ext_y/2,
                                                x_axis_frac_offset=x_text_offset, y_axis_frac_offset=y_text_offset,
                                              x_scale_log=False, y_scale_log=False,
                                   fontsize=fontsize, text_color=text_color,
                                   horizontal_alignment='center',
                                   vertical_alignment='bottom',
                                   path_eff=False, path_err_linewidth=3, path_eff_color='white', rotation=angle_av_vector)

            # ax.text(x_color_int + x_text_offset, y_color_int + y_text_offset, arrow_text,
            #         horizontalalignment='left', verticalalignment='bottom',
            #         transform_rotates_text=True, rotation_mode='anchor',
            #         rotation=angle_av_vector, fontsize=fontsize, color=text_color)

    @staticmethod
    def plot_reddening_vect_wave_av(ax, x_wave_1, x_wave_2, y_wave_1, y_wave_2,
                                 x_color_int=0, y_color_int=0, av_val=1,
                        linewidth=2, line_color='k',
                        text=True, fontsize=20, text_color='k', x_text_offset=0.01, y_text_offset=-0.01):

        color_ext_x = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=x_wave_1, wave2=x_wave_2, av=av_val)
        color_ext_y = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=y_wave_1, wave2=y_wave_2, av=av_val)
        print(color_ext_x)
        print(color_ext_y)

        # slope_av_vector = ((y_color_int + color_ext_y) - y_color_int) / ((x_color_int + color_ext_x) - x_color_int)

        angle_av_vector = np.arctan(color_ext_y/color_ext_x) * 180/np.pi

        ax.annotate('', xy=(x_color_int + color_ext_x, y_color_int + color_ext_y), xycoords='data',
                    xytext=(x_color_int, y_color_int), fontsize=fontsize,
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color=line_color, lw=linewidth, ls='-'))

        if text:
            if isinstance(av_val, int):
                arrow_text = r'A$_{\rm V}$=%i mag' % av_val
            else:
                arrow_text = r'A$_{\rm V}$=%.1f mag' % av_val

            StrTools.display_text_on_data_point(ax=ax, text=arrow_text,
                                                x_data_point=x_color_int + color_ext_x/2,
                                                y_data_point= y_color_int + color_ext_y/2,
                                                x_axis_frac_offset=x_text_offset, y_axis_frac_offset=y_text_offset,
                                              x_scale_log=False, y_scale_log=False,
                                   fontsize=fontsize, text_color=text_color,
                                   horizontal_alignment='center',
                                   vertical_alignment='bottom',
                                   path_eff=False, path_err_linewidth=3, path_eff_color='white', rotation=angle_av_vector)

            # ax.text(x_color_int + x_text_offset, y_color_int + y_text_offset, arrow_text,
            #         horizontalalignment='left', verticalalignment='bottom',
            #         transform_rotates_text=True, rotation_mode='anchor',
            #         rotation=angle_av_vector, fontsize=fontsize, color=text_color)


    @staticmethod
    def plot_reddening_vect_wave_ebv(ax, x_wave_1, x_wave_2, y_wave_1, y_wave_2,
                                 x_color_int=0, y_color_int=0, ebv_val=1,
                        linewidth=2, line_color='k',
                        text=True, fontsize=20, text_color='k', x_text_offset=0.01, y_text_offset=-0.01):

        color_ext_x = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=x_wave_1, wave2=x_wave_2, av=dust_tools.extinction_tools.ExtinctionTools.ebv2av(ebv=ebv_val))
        color_ext_y = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=y_wave_1, wave2=y_wave_2, av=dust_tools.extinction_tools.ExtinctionTools.ebv2av(ebv=ebv_val))
        print(color_ext_x)
        print(color_ext_y)

        # slope_av_vector = ((y_color_int + color_ext_y) - y_color_int) / ((x_color_int + color_ext_x) - x_color_int)

        angle_av_vector = np.arctan(color_ext_y/color_ext_x) * 180/np.pi

        ax.annotate('', xy=(x_color_int + color_ext_x, y_color_int + color_ext_y), xycoords='data',
                    xytext=(x_color_int, y_color_int), fontsize=fontsize,
                    textcoords='data', arrowprops=dict(arrowstyle='-|>', color=line_color, lw=linewidth, ls='-'))

        if text:
            if isinstance(ebv_val, int):
                arrow_text = r'E(B-V)=%i mag' % ebv_val
            else:
                arrow_text = r'E(B-V)=%.1f mag' % ebv_val

            StrTools.display_text_on_data_point(ax=ax, text=arrow_text,
                                                x_data_point=x_color_int + color_ext_x/2,
                                                y_data_point= y_color_int + color_ext_y/2,
                                                x_axis_frac_offset=x_text_offset, y_axis_frac_offset=y_text_offset,
                                              x_scale_log=False, y_scale_log=False,
                                   fontsize=fontsize, text_color=text_color,
                                   horizontal_alignment='center',
                                   vertical_alignment='bottom',
                                   path_eff=False, path_err_linewidth=3, path_eff_color='white', rotation=angle_av_vector)

            # ax.text(x_color_int + x_text_offset, y_color_int + y_text_offset, arrow_text,
            #         horizontalalignment='left', verticalalignment='bottom',
            #         transform_rotates_text=True, rotation_mode='anchor',
            #         rotation=angle_av_vector, fontsize=fontsize, color=text_color)


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
    def float2mag_str(f, f_err=None, n_digits=2):
        order_of_mag = int(np.log10(f))
        str_value = StrTools.float2str(f=f/(10**(order_of_mag)), max_digits=n_digits)
        if f_err is not None:
            str_value_f_err = f"{f_err/(10**(order_of_mag)):.{n_digits}f}"

            str_value += (r' $\pm$ ' + str_value_f_err)
        str_value += (r' $10^{%i}$' % order_of_mag)
        return str_value

    @staticmethod
    def age2label(age):
        if np.isnan(age):
            return 'NaN'
        if age < 1000: return str(age) + ' Myr'
        # convert to Gyr
        else:
            if age % 1000 == 0: return '%i Gyr' % (age / 1000)
            else: return '%.1f Gyr' % (age / 1000)

    @staticmethod
    def mstar2label(mstar):
        if np.isnan(mstar):
            return 'NaN'
        if mstar == -999:
            return '-999'
        print(mstar)
        order_of_mag = int(np.log10(mstar))
        print(order_of_mag)
        return r'%.1f 10$^{%i}$ M$_{\odot}$' % ((mstar / 10**(order_of_mag)), order_of_mag)


    @staticmethod
    def display_text_in_corner(ax, text, fontsize, text_color='k', x_frac=0.02, y_frac=0.98, horizontal_alignment='left',
                               vertical_alignment='top', path_eff=True, path_err_linewidth=3, path_eff_color='white', rotation=0):
        if path_eff:
            pe = [patheffects.withStroke(linewidth=path_err_linewidth, foreground=path_eff_color)]
        else:
            pe = None
        ax.text(x_frac, y_frac, text, horizontalalignment=horizontal_alignment, verticalalignment=vertical_alignment,
                fontsize=fontsize, color=text_color, transform=ax.transAxes, path_effects=pe, rotation=rotation)

    @staticmethod
    def display_text_on_data_point(ax, text,
                                   x_data_point, y_data_point,
                                   x_axis_frac_offset=0., y_axis_frac_offset=0.01,
                                              x_scale_log=False, y_scale_log=False,
                                   fontsize=None, text_color='k',
                                   horizontal_alignment='center',
                                   vertical_alignment='bottom',
                                       path_eff=True, path_err_linewidth=3, path_eff_color='white', rotation=0, rotation_mode='anchor', transform_rotates_text=True):
        if path_eff:
            pe = [patheffects.withStroke(linewidth=path_err_linewidth, foreground=path_eff_color)]
        else:
            pe = None

        if x_scale_log:
            x_diff = np.log10(ax.get_xlim())[1] - np.log10(ax.get_xlim())[0]
            x_pos = 10**(np.log10(x_data_point) + x_diff*x_axis_frac_offset)
        else:
            x_pos = x_data_point + (ax.get_xlim()[1] - ax.get_xlim()[0])*x_axis_frac_offset

        if y_scale_log:
            y_diff = np.log10(ax.get_ylim())[1] - np.log10(ax.get_ylim())[0]
            y_pos = 10**(np.log10(y_data_point) + y_diff*y_axis_frac_offset)
        else:
            y_pos = y_data_point + (ax.get_ylim()[1] - ax.get_ylim()[0])*y_axis_frac_offset
        ax.text(x_pos, y_pos, text, horizontalalignment=horizontal_alignment, verticalalignment=vertical_alignment,
                fontsize=fontsize, color=text_color, path_effects=pe, rotation=rotation, rotation_mode=rotation_mode, transform_rotates_text=transform_rotates_text)

    @staticmethod
    def display_text_on_fig(fig, text,x_pos, y_pos, fontsize, text_color=None, horizontal_alignment='left', vertical_alignment='top',
                            with_box=True, box_frame_color='k', box_frame_line_width=3, box_facecolor='grey',
                            box_style='round', box_edge_pad=0.5, box_alpha=0.4):
        if with_box:
            bbox = dict(facecolor=box_facecolor, edgecolor=box_frame_color, boxstyle=box_style + ',pad=' + str(box_edge_pad), linewidth=box_frame_line_width,
                        alpha=box_alpha)
        else:
            bbox=None
        fig.text(x_pos, y_pos, text, color=text_color, fontsize=fontsize, horizontalalignment=horizontal_alignment,
                 verticalalignment=vertical_alignment,
                 bbox=bbox)


class ColorBarTools:

    @staticmethod
    def create_cbar(ax_cbar, cmap, norm, cbar_label, fontsize, ticks=None, labelpad=2, tick_width=2,
                    orientation='vertical', top_lable=True, label_color='k',
                    extend='neither'):
        """

        Parameters
        ----------
        ax_cbar : ``matplotlib.pylab.axis``
        cmap : str
            same as name parameter of ``matplotlib.colors.Colormap.name``
        norm : ``matplotlib.colors.Normalize``  or ``matplotlib.colors.LogNorm``
        cbar_label : str
        fontsize : int or float
        ticks : list
        labelpad : int or float
        tick_width : int or float
        orientation : str
            default is `vertical`
        extend : str
            default is 'neither'
            can be 'neither', 'min' , 'max' or 'both'
        """
        ColorbarBase(ax_cbar, orientation=orientation, cmap=cmap, norm=norm, extend=extend, ticks=ticks)
        if orientation == 'vertical':
            ax_cbar.set_ylabel(cbar_label, labelpad=labelpad, fontsize=fontsize, color=label_color)
            ax_cbar.tick_params(axis='both', which='both', width=tick_width, direction='in', top=True,
                                labelbottom=False,
                                labeltop=True, labelsize=fontsize, colors=label_color)
        elif orientation == 'horizontal':
            if top_lable:
                # ax_cbar.set_xlabel(cbar_label, labelpad=labelpad, fontsize=fontsize)
                ax_cbar.tick_params(width=tick_width, direction='in', top=True, labeltop=True, bottom=False,
                                    labelbottom=False,
                                    labelsize=fontsize, colors=label_color)
                # also put the minor ticks to the top
                ax_cbar.tick_params(which='minor', width=tick_width, direction='in',
                                    top=True, labeltop=True, bottom=False, labelbottom=False,
                                    labelsize=fontsize / 1.5)
                ax_cbar.set_title(cbar_label, fontsize=fontsize, color=label_color)
            else:
                # ax_cbar.set_xlabel(cbar_label, labelpad=labelpad, fontsize=fontsize)
                ax_cbar.tick_params(width=tick_width, direction='in', top=False, labeltop=False, bottom=True,
                                    labelbottom=True,
                                    labelsize=fontsize, colors=label_color)
                # also put the minor ticks to the top
                ax_cbar.tick_params(which='minor', width=tick_width, direction='in',
                                    top=False, labeltop=False, bottom=True, labelbottom=True,
                                    labelsize=fontsize / 1.5, colors=label_color)
                ax_cbar.set_xlabel(cbar_label, fontsize=fontsize, color=label_color)
    @staticmethod
    def compute_cbar_norm(vmin_vmax=None, cutout_list=None, log_scale=False):
        """
        Computing the color bar scale for a single or multiple cutouts.

        Parameters
        ----------
        vmin_vmax : tuple
        cutout_list : list
            This list should include all cutouts
        log_scale : bool

        Returns
        -------
        norm : ``matplotlib.colors.Normalize``  or ``matplotlib.colors.LogNorm``
        """
        if (vmin_vmax is None) & (cutout_list is None):
            raise KeyError('either vmin_vmax or cutout_list must be not None')

        # get maximal value
        # vmin_vmax

        if vmin_vmax is None:
            vmin = None
            vmax = None
            for cutout in cutout_list:
                sigma_clip = SigmaClip(sigma=3)
                mask_zeros = np.invert(cutout == 0)
                if len(sigma_clip(cutout[mask_zeros])) == 0:
                    return None
                min = np.nanmin(sigma_clip(cutout[mask_zeros]))
                max = np.nanmax(sigma_clip(cutout[mask_zeros]))
                if vmin is None:
                    vmin = min
                if vmax is None:
                    vmax = max
                if min < vmin:
                    vmin = min
                if max > vmax:
                    vmax = max

            # list_of_means = [np.nanmean(cutout) for cutout in cutout_list]
            # list_of_stds = [np.nanstd(cutout) for cutout in cutout_list]
            # mean, std = (np.nanmean(list_of_means), np.nanstd(list_of_stds))
            #
            # vmin = mean - 5 * std
            # vmax = mean + 20 * std


        else:
            vmin, vmax = vmin_vmax[0], vmin_vmax[1]
        if log_scale:

            if vmax < 0:
                vmax = 0.000001
            if vmin < 0:
                vmin = vmax / 100
            norm = LogNorm(vmin, vmax)
        else:
            norm = Normalize(vmin, vmax)
        return norm

    @staticmethod
    def get_cutout_norm(img_data, min_std_scale=3,  max_std_scale=10):
        """
        Function calculate scaling in image by taking median background of image and the maximum value inside a circle

        Parameters
        ----------
        img_data : ``numpy.ndarray``
            data image
        min_std_scale, max_std_scale : float
            multiplicative faftor for max and min


        Returns
        -------
        (bkg_median, max_value) : tuple
            median background and maximum value inside the circle
        """

        mean, median, std = sigma_clipped_stats(img_data, sigma=3.0)
        return ImageNormalize(stretch=SqrtStretch(), vmin=median - min_std_scale * std, vmax=median + max_std_scale * std)

    @staticmethod
    def get_image_scale_with_circle(img_data, img_wcs, ra, dec, circle_rad=0.16, box_scaling=1/10, filter_scaling=1/10):
        """
        Function calculate scaling in image by taking median background of image and the maximum value inside a circle

        Parameters
        ----------
        img_data : ``numpy.ndarray``
            data image
        img_wcs : ``astropy.wcs.WCS``
            world coordinate system
        ra, dec : float
            coordinates
        circle_rad : float
            radius of circle
        box_scaling : float
            factor by how much the box estimation for the Background2D should be relative to the input image
        filter_scaling : float
            factor by how much the filter for the Background2D should be relative to the input image


        Returns
        -------
        (bkg_median, max_value) : tuple
            median background and maximum value inside the circle
        """
        # get background value as minimum
        sigma_clip = SigmaClip()
        bkg_estimator = MedianBackground()
        box_size = list([int(img_data.shape[0]*box_scaling), int(img_data.shape[0]*box_scaling)])
        filter_size = list([int(img_data.shape[0]*filter_scaling), int(img_data.shape[0]*filter_scaling)])
        # assure that filter has an odd size
        if filter_size[0] % 2 == 0:
            filter_size[0] += 1
        if filter_size[1] % 2 == 0:
            filter_size[1] += 1

        # get background estimation
        bkg = Background2D(img_data, box_size=box_size, filter_size=filter_size,
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        bkg_median = bkg.background_median

        # get coordinates and radius in pixel scale
        central_pos_world = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        central_pos_pixel = img_wcs.world_to_pixel(central_pos_world)
        circle_rad_pix = helper_func.CoordTools.transform_world2pix_scale(length_in_arcsec=circle_rad, wcs=img_wcs)
        # get meshgrid of image
        mesh_x, mesh_y = np.meshgrid(np.linspace(0, img_data.shape[1]-1, img_data.shape[1]),
                                     np.linspace(0, img_data.shape[0]-1, img_data.shape[0]))
        # mask pixels inside the radius
        mask_inside_circle = np.sqrt((mesh_x - central_pos_pixel[0]) ** 2 +
                                     (mesh_y - central_pos_pixel[1])**2) < circle_rad_pix
        max_value = np.nanmax(img_data[mask_inside_circle])

        # what if the background is higher than the maximal value in the circle ?
        if bkg_median > max_value:
            return max_value/10, max_value
        else:
            return bkg_median, max_value


class SEDTools:
    """
    helper to plot SEDs
    """
    @staticmethod
    def display_hst_filter(ax, band, instrument, display_filter_name=True, filter_name_ypos=1, filter_name_color='k',
                           filter_name_fontsize=None,
                           ymin=None, ymax=None, filter_color='tab:grey', alpha=0.5, wave_unit='mu'):
        if (instrument == 'uvis') | (instrument == 'uvis1'):
            xmin = phys_params.hst_wfc3_uvis1_bands_wave[band]['min_wave']
            xmax = phys_params.hst_wfc3_uvis1_bands_wave[band]['max_wave']
            xmean = phys_params.hst_wfc3_uvis1_bands_wave[band]['mean_wave']
        elif instrument == 'uvis2':
            xmin = phys_params.hst_wfc3_uvis2_bands_wave[band]['min_wave']
            xmax = phys_params.hst_wfc3_uvis2_bands_wave[band]['max_wave']
            xmean = phys_params.hst_wfc3_uvis2_bands_wave[band]['mean_wave']

        elif instrument == 'acs':
            xmin = phys_params.hst_acs_wfc1_bands_wave[band]['min_wave']
            xmax = phys_params.hst_acs_wfc1_bands_wave[band]['max_wave']
            xmean = phys_params.hst_acs_wfc1_bands_wave[band]['mean_wave']

        else:
            raise KeyError('instrument not understood')

        xmin = helper_func.UnitTools.angstrom2unit(xmin, unit=wave_unit)
        xmax = helper_func.UnitTools.angstrom2unit(xmax, unit=wave_unit)
        xmean = helper_func.UnitTools.angstrom2unit(xmean, unit=wave_unit)

        if ymin is None:
            ymin = ax.get_ylim()[0]
        if ymax is None:
            ymax = ax.get_ylim()[1]

        ax.axvspan(xmin, xmax, alpha=alpha, ymin=ymin, ymax=ymax, color=filter_color)

        if display_filter_name:
            StrTools.display_text_on_data_point(ax=ax, text=helper_func.ObsTools.hst_band2filter_name(band=band),
                                   x_data_point=xmean, y_data_point=filter_name_ypos,
                                   x_axis_frac_offset=0., y_axis_frac_offset=0.0,
                                   fontsize=filter_name_fontsize, text_color=filter_name_color,
                                   horizontal_alignment='center',
                                   vertical_alignment='bottom',
                                   path_eff=True, path_err_linewidth=3, path_eff_color='white', rotation=0)

    @staticmethod
    def display_all_hst_filter(ax, target=None, alpha=0.5, filter_name_ypos=1, filter_name_fontsize=None, ymin=None, ymax=None,
                               wave_unit='mu'):
        if target is None:
            band_list = ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']
            instrument_list = ['uvis', 'uvis', 'uvis', 'uvis', 'uvis']
            for band, instrument in zip(band_list, instrument_list):
                SEDTools.display_hst_filter(ax=ax, band=band, instrument=instrument,
                                            display_filter_name=True, filter_name_ypos=filter_name_ypos,
                                            filter_name_color='k',
                           filter_name_fontsize=filter_name_fontsize,
                           ymin=ymin, ymax=ymax, filter_color='tab:grey', alpha=alpha, wave_unit=wave_unit)
        else:
            band_list = ObsTools.get_hst_obs_broad_band_list(target=target)
            for band in band_list:
                SEDTools.display_hst_filter(ax=ax, band=band, instrument=helper_func.ObsTools.get_hst_instrument(target=target, band=band),
                                            display_filter_name=True, filter_name_ypos=filter_name_ypos,
                                            filter_name_color='k',
                                            filter_name_fontsize=filter_name_fontsize,
                                            ymin=ymin, ymax=ymax, filter_color='tab:grey', alpha=alpha, wave_unit=wave_unit)

    @staticmethod
    def display_nircam_filter(ax, band, display_filter_name=1, filter_name_ypos=None, filter_name_color='k',
                              filter_name_fontsize=None,
                              ymin=None, ymax=None, filter_color='tab:grey', alpha=0.5, wave_unit='mu'):
        xmin = phys_params.nircam_bands_wave[band]['min_wave']
        xmax = phys_params.nircam_bands_wave[band]['max_wave']
        xmean = phys_params.nircam_bands_wave[band]['mean_wave']

        xmin = helper_func.UnitTools.angstrom2unit(xmin, unit=wave_unit)
        xmax = helper_func.UnitTools.angstrom2unit(xmax, unit=wave_unit)
        xmean = helper_func.UnitTools.angstrom2unit(xmean, unit=wave_unit)

        if ymin is None:
            ymin = ax.get_ylim()[0]
        if ymax is None:
            ymax = ax.get_ylim()[1]

        ax.axvspan(xmin, xmax, alpha=alpha, ymin=ymin, ymax=ymax, color=filter_color)

        if display_filter_name:
            StrTools.display_text_on_data_point(ax=ax, text=band,
                                   x_data_point=xmean, y_data_point=filter_name_ypos,
                                   x_axis_frac_offset=0., y_axis_frac_offset=0.0,
                                   fontsize=filter_name_fontsize, text_color=filter_name_color,
                                   horizontal_alignment='center',
                                   vertical_alignment='center',
                                   path_eff=True, path_err_linewidth=3, path_eff_color='white', rotation=90)

    @staticmethod
    def display_all_nircam_filter(ax, target=None, alpha=0.5, filter_name_ypos=1, filter_name_fontsize=None, ymin=None,
                               ymax=None, wave_unit='mu'):

        if filter_name_ypos is None:
            filter_name_ypos = 1
        if target is None:
            band_list = ['F200W', 'F300M', 'F335M', 'F360M']
        else:
            band_list = ObsTools.get_nircam_obs_band_list(target=target)

        for band in band_list:
            SEDTools.display_nircam_filter(ax=ax, band=band,
                                        display_filter_name=True, filter_name_ypos=filter_name_ypos,
                                        filter_name_color='k',
                                        filter_name_fontsize=filter_name_fontsize,
                                        ymin=ymin, ymax=ymax, filter_color='tab:grey', alpha=alpha,
                                        wave_unit=wave_unit)

    @staticmethod
    def display_miri_filter(ax, band, display_filter_name=True, filter_name_ypos=1, filter_name_color='k',
                           filter_name_fontsize=None,
                           ymin=None, ymax=None, filter_color='tab:grey', alpha=0.5, wave_unit='mu'):
        xmin = phys_params.miri_bands_wave[band]['min_wave']
        xmax = phys_params.miri_bands_wave[band]['max_wave']
        xmean = phys_params.miri_bands_wave[band]['mean_wave']

        xmin = helper_func.UnitTools.angstrom2unit(xmin, unit=wave_unit)
        xmax = helper_func.UnitTools.angstrom2unit(xmax, unit=wave_unit)
        xmean = helper_func.UnitTools.angstrom2unit(xmean, unit=wave_unit)

        if ymin is None:
            ymin = ax.get_ylim()[0]
        if ymax is None:
            ymax = ax.get_ylim()[1]

        ax.axvspan(xmin, xmax, alpha=alpha, ymin=ymin, ymax=ymax, color=filter_color)

        if display_filter_name:
            StrTools.display_text_on_data_point(ax=ax, text=band,
                                   x_data_point=xmean, y_data_point=filter_name_ypos,
                                   x_axis_frac_offset=0., y_axis_frac_offset=0.0,
                                   fontsize=filter_name_fontsize, text_color=filter_name_color,
                                   horizontal_alignment='center',
                                   vertical_alignment='center',
                                   path_eff=True, path_err_linewidth=3, path_eff_color='white', rotation=90)

    @staticmethod
    def display_all_miri_filter(ax, target=None, alpha=0.5, filter_name_ypos=1, filter_name_fontsize=None, ymin=None,
                               ymax=None,
                               wave_unit='mu'):
        if filter_name_ypos is None:
            filter_name_ypos = 1
        if target is None:
            band_list = ['F770W', 'F1000W', 'F1130W', 'F2100W']
        else:
            band_list = ObsTools.get_miri_obs_band_list(target=target)

        for band in band_list:
            SEDTools.display_miri_filter(ax=ax, band=band,
                                        display_filter_name=True, filter_name_ypos=filter_name_ypos,
                                        filter_name_color='k',
                                        filter_name_fontsize=filter_name_fontsize,
                                        ymin=ymin, ymax=ymax, filter_color='tab:grey', alpha=alpha,
                                        wave_unit=wave_unit)


class SpecPlotTools:

    @staticmethod
    def get_em_comp_colors(idx=0, n_comps=1, line_type='nl'):
        # can go up to five gaussian components (I think everything above this is totally nonsense)
        color_list_nl_1 = [color_list_tab10[2]]
        color_list_nl_2 = [color_list_tab10[0], color_list_tab10[3]]
        color_list_nl_3 = [color_list_tab10[0], color_list_tab10[1], color_list_tab10[3]]
        color_list_nl_4 = [color_list_tab10[0], color_list_tab10[1], color_list_tab10[3], color_list_tab10[4]]
        color_list_nl_5 = [color_list_tab10[0], color_list_tab10[1], color_list_tab10[3], color_list_tab10[4], color_list_tab10[6]]
        # somehow I allow two BL...
        color_list_bl =[color_list_tab10[5], color_list_tab10[7]]

        if line_type == 'nl':
            return locals()['color_list_nl_%i' % n_comps][idx]
        if line_type == 'bl':
            return color_list_bl[idx]
        if line_type == 'total':
            return color_list_tab10[2]

    @staticmethod
    def plot_em_line_spec(ax, em_line_fit_dict, line_list, ax_res=None, left_offset=15, right_offset=15,
                          y_axis_offset_frac_bottom = 0.1, y_axis_offset_frac_top = 0.05,
                          instrument='muse', display_legend=False, display_line_names=True,
                          label_color='k', spec_line_color='k',
                          font_size_label=20, font_size_title=30,
                          display_y_label=True, y_label_pos='left',
                          display_x_label=True, y_axis_scale=1e16,
                          data_lw=3, model_lw=4):


        # get spec dimensions
        left_obs_wave = SpecTools.get_line_pos(line=np.min(line_list), vel_kmps=em_line_fit_dict['sys_vel'],
                                               instrument=instrument)
        right_obs_wave = SpecTools.get_line_pos(line=np.max(line_list), vel_kmps=em_line_fit_dict['sys_vel'],
                                                instrument=instrument)
        min_wave = left_obs_wave - left_offset
        max_wave = right_obs_wave + right_offset
        mask_select_wave = ((em_line_fit_dict['wave'] > min_wave) & (em_line_fit_dict['wave'] < max_wave))
        spec_min = np.nanmin(em_line_fit_dict['em_flux'][mask_select_wave])
        spec_max = np.nanmax(em_line_fit_dict['em_flux'][mask_select_wave])
        spec_width = spec_max - spec_min

        # set limits
        ax.set_xlim(min_wave, max_wave)
        ax.set_ylim((spec_min - y_axis_offset_frac_bottom * spec_width) * y_axis_scale,
                    (spec_max + y_axis_offset_frac_top * spec_width) * y_axis_scale)
        # plot all the data
        ax.step(em_line_fit_dict['wave'], em_line_fit_dict['em_flux'] * y_axis_scale, where='mid', linewidth=data_lw, color=spec_line_color,
                label='Cont. sub. Spec.')

        # plot all individual models
        dummy_wave = np.linspace(min_wave, max_wave, sum(mask_select_wave) * 10)
        dummy_total_model = np.zeros(len(dummy_wave))

        # plot narrow gaussian lines
        for gauss_idx in range(em_line_fit_dict['n_nl_gauss']):
            dummy_gaus_comp = np.zeros(len(dummy_wave))

            for line in line_list:
                dummy_gaus_comp += SpecTools.get_obs_gauss_from_fit_output(x_data=dummy_wave,
                                                                           em_line_fit_dict=em_line_fit_dict,
                                                                           line=line, gauss_index=gauss_idx,
                                                                           line_type='nl', vel_unit='kmps',
                                                                           instrument=instrument)
            ax.plot(dummy_wave, dummy_gaus_comp * y_axis_scale, linewidth=model_lw,
                    color=SpecPlotTools.get_em_comp_colors(
                        idx=gauss_idx, n_comps=em_line_fit_dict['n_nl_gauss'], line_type='nl'),
                    label='Comp %i' % (gauss_idx + 1))
            dummy_total_model += dummy_gaus_comp

        # plot broad gaussian lines
        for gauss_idx in range(em_line_fit_dict['n_bl_gauss']):
            dummy_gaus_comp = np.zeros(len(dummy_wave))

            for line in line_list:
                if line in [4863, 6565]:
                    dummy_gaus_comp += SpecTools.get_obs_gauss_from_fit_output(x_data=dummy_wave,
                                                                               em_line_fit_dict=em_line_fit_dict,
                                                                               line=line, gauss_index=gauss_idx,
                                                                               line_type='bl', vel_unit='kmps',
                                                                               instrument=instrument)
            ax.plot(dummy_wave, dummy_gaus_comp * y_axis_scale, linewidth=model_lw,
                    color=SpecPlotTools.get_em_comp_colors(
                        idx=gauss_idx, n_comps=em_line_fit_dict['n_bl_gauss'], line_type='bl'),
                    label='BRL Comp %i' % (gauss_idx + 1))
            dummy_total_model += dummy_gaus_comp

        ax.plot(dummy_wave, dummy_total_model * y_axis_scale, linewidth=model_lw,
                color=SpecPlotTools.get_em_comp_colors(line_type='total'), label='Best fit')

        if display_legend:
            ax.legend(frameon=False, fontsize=font_size_title)

        # display line names
        if display_line_names:
            for line in line_list:
                StrTools.display_text_on_data_point(
                    ax=ax, text=phys_params.opt_line_wave[line]['plot_name'],
                    x_data_point=SpecTools.get_line_pos(line=line, vel_kmps=em_line_fit_dict['sys_vel'], instrument=instrument),
                    y_data_point = 0, x_axis_frac_offset=0., y_axis_frac_offset=-0.02,
                    x_scale_log=False, y_scale_log=False, fontsize=font_size_title, text_color='k',
                    horizontal_alignment='center', vertical_alignment='top', path_eff=True,
                    path_err_linewidth=3, path_eff_color='white', rotation=0,
                    rotation_mode='anchor', transform_rotates_text=True)
                ax.plot([SpecTools.get_line_pos(line=line, vel_kmps=em_line_fit_dict['sys_vel'], instrument=instrument),
                         SpecTools.get_line_pos(line=line, vel_kmps=em_line_fit_dict['sys_vel'], instrument=instrument)],
                        [0, SpecTools.estimate_line_amp(
                            line=line, wave=em_line_fit_dict['wave'], em_flux=em_line_fit_dict['em_flux'],
                            vel=em_line_fit_dict['sys_vel'], instrument=instrument, bin_rad=4) * y_axis_scale * 0.2],
                        color='k', linestyle='--')
        ax.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', colors=label_color,
                           labelsize=font_size_label)
        # put X in labels
        if ax_res is not None:
            best_fit = np.zeros(len(em_line_fit_dict['wave']))
            best_fit[em_line_fit_dict['ln_mask']] = em_line_fit_dict['best_fit']
            ax_res.step(em_line_fit_dict['wave'][mask_select_wave],
                        (em_line_fit_dict['em_flux'] - best_fit)[mask_select_wave] * y_axis_scale, where='mid', linewidth=2,
                        color='k')
            ax_res.set_xlim(min_wave, max_wave)
            ax_res.plot([min_wave, max_wave], [0, 0], linewidth=2,
                        color=SpecPlotTools.get_em_comp_colors(line_type='total'))
            ax.set_xticklabels([])
            ax_res.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', colors=label_color,
                               labelsize=font_size_label)
            if display_x_label:
                ax_res.set_xlabel(r'Wavelength [${\rm \AA}$]', fontsize=font_size_label, color=label_color)
        else:
            if display_x_label:
                ax.set_xlabel(r'Wavelength [${\rm \AA}$]', fontsize=font_size_label)

        # put in Y labels
        if y_label_pos == 'left':

            if display_y_label:
                ax.set_ylabel(r'$\phi$ [10$^{-%i}$ erg cm$^{-2}$ s$^{-1}$ ${\rm \AA^{-1}}$]' % int(np.log10(y_axis_scale)),
                              fontsize=font_size_label, color=label_color)
        else:
            ax.yaxis.set_label_position('right')
            ax.yaxis.tick_right()
            if display_y_label:
                ax.set_ylabel(r'$\phi$ [10$^{-%i}$ erg cm$^{-2}$ s$^{-1}$ ${\rm \AA^{-1}}$]' % int(np.log10(y_axis_scale)),
                              fontsize=font_size_label, color=label_color)


    @staticmethod
    def plot_red_bump(ax, ppxf_fit_dict, y_axis_scale=1e16, font_size_label=20, display_label=True,
                          font_size_title=30, display_y_label=False, y_label_pos='left', display_x_label=False,
                      plot_continuum=True,
                      min_rest_wave_red_bump=5650, max_rest_wave_red_bump=5850,
        y_axis_offset_frac_bottom = 0.1, y_axis_offset_frac_top = 0.05, data_lw=3, model_lw=4
    ):


        min_obs_wave_red_bump = SpecTools.conv_rest_wave2obs_wave(rest_wave=min_rest_wave_red_bump, vel_kmps=ppxf_fit_dict['sys_vel'])
        max_obs_wave_red_bump = SpecTools.conv_rest_wave2obs_wave(rest_wave=max_rest_wave_red_bump, vel_kmps=ppxf_fit_dict['sys_vel'])
        mask = ((ppxf_fit_dict['wave'] >  min_obs_wave_red_bump) &
                (ppxf_fit_dict['wave'] <  max_obs_wave_red_bump))


        spec_min = np.nanmin(ppxf_fit_dict['total_flux'][mask])
        spec_max = np.nanmax(ppxf_fit_dict['total_flux'][mask])
        spec_width = spec_max - spec_min

        # set limits
        ax.set_xlim(min_obs_wave_red_bump, max_obs_wave_red_bump)
        ax.set_ylim((spec_min - y_axis_offset_frac_bottom * spec_width) * y_axis_scale,
                    (spec_max + y_axis_offset_frac_top * spec_width) * y_axis_scale)


        ax.step(ppxf_fit_dict['wave'][mask], ppxf_fit_dict['total_flux'][mask] * y_axis_scale, where='mid',
                linewidth=data_lw, color='k', label='Obs. Spectrum')

        if plot_continuum:
            ax.plot(ppxf_fit_dict['wave'][mask], ppxf_fit_dict['continuum_best_fit'][mask] * 1e16, linewidth=model_lw,
                    color='tab:orange', label='Stellar Continuum fit')
        if display_label:
            StrTools.display_text_in_corner(ax=ax, text='Red Bump', fontsize=font_size_title, text_color='k',
                                            x_frac=0.97, y_frac=0.97, horizontal_alignment='right',
                               vertical_alignment='top', path_eff=True, path_err_linewidth=3, path_eff_color='white', rotation=0)

        ax.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in',
                       labelsize=font_size_label)
        if display_x_label:
            ax.set_xlabel(r'Wavelength [${\rm \AA}$]', fontsize=font_size_label)

        # put in Y labels
        if y_label_pos == 'left':

            if display_y_label:
                ax.set_ylabel(r'$\phi$ [10$^{-%i}$ erg cm$^{-2}$ s$^{-1}$ ${\rm \AA^{-1}}$]' % int(np.log10(y_axis_scale)),
                              fontsize=font_size_label)
        else:
            ax.yaxis.set_label_position('right')
            ax.yaxis.tick_right()
            if display_y_label:
                ax.set_ylabel(r'$\phi$ [10$^{-%i}$ erg cm$^{-2}$ s$^{-1}$ ${\rm \AA^{-1}}$]' % int(np.log10(y_axis_scale)),
                              fontsize=font_size_label)



    @staticmethod
    def plot_stellar_hei6680(ax, ppxf_fit_dict, y_axis_scale=1e16, font_size_label=20, display_label=True,
                          font_size_title=30, display_y_label=False, display_x_label=False):
        min_rest_wave_he1 = phys_params.opt_line_wave[6680]['vac_wave'] - 20
        max_rest_wave_he1 = phys_params.opt_line_wave[6680]['vac_wave'] + 20

        min_obs_wave_he1 = SpecTools.conv_rest_wave2obs_wave(rest_wave=min_rest_wave_he1, vel_kmps=ppxf_fit_dict['sys_vel'])
        max_obs_wave_he1 = SpecTools.conv_rest_wave2obs_wave(rest_wave=max_rest_wave_he1, vel_kmps=ppxf_fit_dict['sys_vel'])
        mask = ((ppxf_fit_dict['wave'] >  min_obs_wave_he1) &
                (ppxf_fit_dict['wave'] <  max_obs_wave_he1))
        print(ppxf_fit_dict.keys())


        ax.step(ppxf_fit_dict['wave'][mask], ppxf_fit_dict['total_flux'][mask] * y_axis_scale, where='mid', linewidth=2, color='k',
                label='Obs. Spectrum')
        ax.plot(ppxf_fit_dict['wave'][mask], ppxf_fit_dict['best_fit'][mask] * y_axis_scale, linewidth=3, color='tab:red',
                          label='Best total fit')
        ax.plot(ppxf_fit_dict['wave'][mask], ppxf_fit_dict['continuum_best_fit'][mask] * 1e16, linewidth=3,
                color='tab:orange', label='Stellar Continuum fit')
        if display_label:
            StrTools.display_text_in_corner(ax=ax, text='HeI 6680', fontsize=font_size_title, text_color='k',
                                            x_frac=0.97, y_frac=0.97, horizontal_alignment='right',
                               vertical_alignment='top', path_eff=True, path_err_linewidth=3, path_eff_color='white', rotation=0)

        ax.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in',
                       labelsize=font_size_label)
        if display_x_label:
            ax.set_xlabel(r'Wavelength [${\rm \AA}$]', fontsize=font_size_label)
        if display_y_label:
            ax.set_ylabel(r'$\phi$ [10$^{-%i}$ erg cm$^{-2}$ s$^{-1}$ ${\rm \AA^{-1}}$]' % int(np.log10(y_axis_scale)),
                          fontsize=font_size_label)

from matplotlib import patches
from matplotlib import text as mtext
import math

class CurvedText(mtext.Text):
    """
    A text object that follows an arbitrary curve.
    """
    def __init__(self, x, y, text, axes, **kwargs):
        super(CurvedText, self).__init__(x[0],y[0],' ', **kwargs)

        axes.add_artist(self)

        ##saving the curve:
        self.__x = x
        self.__y = y
        self.__zorder = self.get_zorder()

        ##creating the text objects
        self.__Characters = []
        for c in text:
            if c == ' ':
                ##make this an invisible 'a':
                t = mtext.Text(0,0,'a')
                t.set_alpha(0.0)
            else:
                t = mtext.Text(0,0,c, **kwargs)

            #resetting unnecessary arguments
            t.set_ha('center')
            t.set_rotation(0)
            t.set_zorder(self.__zorder +1)

            self.__Characters.append((c,t))
            axes.add_artist(t)


    ##overloading some member functions, to assure correct functionality
    ##on update
    def set_zorder(self, zorder):
        super(CurvedText, self).set_zorder(zorder)
        self.__zorder = self.get_zorder()
        for c,t in self.__Characters:
            t.set_zorder(self.__zorder+1)

    def draw(self, renderer, *args, **kwargs):
        """
        Overload of the Text.draw() function. Do not do
        do any drawing, but update the positions and rotation
        angles of self.__Characters.
        """
        self.update_positions(renderer)

    def update_positions(self,renderer):
        """
        Update positions and rotations of the individual text elements.
        """

        #preparations

        ##determining the aspect ratio:
        ##from https://stackoverflow.com/a/42014041/2454357

        ##data limits
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        ## Axis size on figure
        figW, figH = self.axes.get_figure().get_size_inches()
        ## Ratio of display units
        _, _, w, h = self.axes.get_position().bounds
        ##final aspect ratio
        aspect = ((figW * w)/(figH * h))*(ylim[1]-ylim[0])/(xlim[1]-xlim[0])

        #points of the curve in figure coordinates:
        x_fig,y_fig = (
            np.array(l) for l in zip(*self.axes.transData.transform([
            (i,j) for i,j in zip(self.__x,self.__y)
            ]))
        )

        #point distances in figure coordinates
        x_fig_dist = (x_fig[1:]-x_fig[:-1])
        y_fig_dist = (y_fig[1:]-y_fig[:-1])
        r_fig_dist = np.sqrt(x_fig_dist**2+y_fig_dist**2)

        #arc length in figure coordinates
        l_fig = np.insert(np.cumsum(r_fig_dist),0,0)

        #angles in figure coordinates
        rads = np.arctan2((y_fig[1:] - y_fig[:-1]),(x_fig[1:] - x_fig[:-1]))
        degs = np.rad2deg(rads)


        rel_pos = 10
        for c,t in self.__Characters:
            #finding the width of c:
            t.set_rotation(0)
            t.set_va('center')
            bbox1  = t.get_window_extent(renderer=renderer)
            w = bbox1.width
            h = bbox1.height

            #ignore all letters that don't fit:
            if rel_pos+w/2 > l_fig[-1]:
                t.set_alpha(0.0)
                rel_pos += w
                continue

            elif c != ' ':
                t.set_alpha(1.0)

            #finding the two data points between which the horizontal
            #center point of the character will be situated
            #left and right indices:
            il = np.where(rel_pos+w/2 >= l_fig)[0][-1]
            ir = np.where(rel_pos+w/2 <= l_fig)[0][0]

            #if we exactly hit a data point:
            if ir == il:
                ir += 1

            #how much of the letter width was needed to find il:
            used = l_fig[il]-rel_pos
            rel_pos = l_fig[il]

            #relative distance between il and ir where the center
            #of the character will be
            fraction = (w/2-used)/r_fig_dist[il]

            ##setting the character position in data coordinates:
            ##interpolate between the two points:
            x = self.__x[il]+fraction*(self.__x[ir]-self.__x[il])
            y = self.__y[il]+fraction*(self.__y[ir]-self.__y[il])

            #getting the offset when setting correct vertical alignment
            #in data coordinates
            t.set_va(self.get_va())
            bbox2  = t.get_window_extent(renderer=renderer)

            bbox1d = self.axes.transData.inverted().transform(bbox1)
            bbox2d = self.axes.transData.inverted().transform(bbox2)
            dr = np.array(bbox2d[0]-bbox1d[0])

            #the rotation/stretch matrix
            rad = rads[il]
            rot_mat = np.array([
                [math.cos(rad), math.sin(rad)*aspect],
                [-math.sin(rad)/aspect, math.cos(rad)]
            ])

            ##computing the offset vector of the rotated character
            drp = np.dot(dr,rot_mat)

            #setting final position and rotation:
            t.set_position(np.array([x,y])+drp)
            t.set_rotation(degs[il])

            t.set_va('center')
            t.set_ha('center')

            #updating rel_pos to right edge of character
            rel_pos += w-used



