"""
Tool to visualize PHANGS imaging data
"""


import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from astropy.visualization import SqrtStretch, LogStretch

from astropy.visualization.mpl_normalize import ImageNormalize


from matplotlib.colors import Normalize, LogNorm

import matplotlib.pyplot as plt
from phangs_data_access import helper_func, phangs_info, phys_params

from phangs_data_access.phot_access import PhotAccess
from phangs_data_access.gas_access import GasAccess
from phangs_data_access.spec_access import SpecAccess
from phangs_data_access.cluster_cat_access import ClusterCatAccess
from phangs_data_access import phot_tools

from phangs_visualizer import plotting_tools

from dust_tools import extinction_tools

from astropy import constants as const
speed_of_light_kmps = const.c.to('km/s').value
from astropy.table import QTable, Table, Column


class PhotVisualizer(PhotAccess, GasAccess, SpecAccess):
    """
    Class to plot cutouts in multiple bands
    """

    def __init__(self, target_name=None, phot_hst_target_name=None, phot_hst_ha_cont_sub_target_name=None,
                 phot_nircam_target_name=None, phot_miri_target_name=None, phot_astrosat_target_name=None,
                 nircam_data_ver='v1p1p1', miri_data_ver='v1p1p1', astrosat_data_ver='v1p0'):
        PhotAccess.__init__(self,
                            phot_target_name=target_name,
                            phot_hst_target_name=phot_hst_target_name,
                            phot_hst_ha_cont_sub_target_name=phot_hst_ha_cont_sub_target_name,
                            phot_nircam_target_name=phot_nircam_target_name,
                            phot_miri_target_name=phot_miri_target_name,
                            phot_astrosat_target_name=phot_astrosat_target_name,
                            nircam_data_ver=nircam_data_ver,
                            miri_data_ver=miri_data_ver,
                            astrosat_data_ver=astrosat_data_ver)
        GasAccess.__init__(self, gas_target_name=target_name)
        SpecAccess.__init__(self, spec_target_name=target_name)

    def plot_hst_overview_panel(self, fig, fig_dict, ra_box=None, dec_box=None):
        """
        """

        # get hst overview image
        band_list = [helper_func.ObsTools.filter_name2hst_band(target=self.phot_hst_target_name,
                                                                filter_name=fig_dict['overview_red_band']),
                     helper_func.ObsTools.filter_name2hst_band(target=self.phot_hst_target_name,
                                                                filter_name=fig_dict['overview_green_band']),
                     helper_func.ObsTools.filter_name2hst_band(target=self.phot_hst_target_name,
                                                                filter_name=fig_dict['overview_blue_band'])]

        # get hst_bvi_zoom_in
        img_overview, wcs_overview = self.get_target_overview_rgb_img(
            red_band=band_list[0], green_band=band_list[1], blue_band=band_list[2],
            overview_img_size=fig_dict['overview_img_pixel_size'], **fig_dict['overview_img_params'])

        # plot the overview image
        # create overview panel axis
        ax_img_overview = fig.add_axes([fig_dict['overview_left_align'], fig_dict['overview_bottom_align'],
                                        fig_dict['overview_width'], fig_dict['overview_height']],
                                       projection=wcs_overview)
        # plot rgb image
        ax_img_overview.imshow(img_overview)
        # add text
        plotting_tools.StrTools.display_text_in_corner(ax=ax_img_overview, text='HST',
                                                       fontsize=fig_dict['overview_title_font_size'],
                                                       text_color='white', x_frac=0.02, y_frac=0.98,
                                                       horizontal_alignment='left', vertical_alignment='top',
                                                       path_eff=True, path_err_linewidth=3, path_eff_color='white')
        plotting_tools.StrTools.display_text_in_corner(ax=ax_img_overview, text=fig_dict['overview_red_band'],
                                                       fontsize=fig_dict['overview_title_font_size'],
                                                       text_color='red', x_frac=0.02, y_frac=0.93,
                                                       horizontal_alignment='left', vertical_alignment='top',
                                                       path_eff=True, path_err_linewidth=3, path_eff_color='white')
        plotting_tools.StrTools.display_text_in_corner(ax=ax_img_overview, text=fig_dict['overview_green_band'],
                                                       fontsize=fig_dict['overview_title_font_size'],
                                                       text_color='green', x_frac=0.02, y_frac=0.88,
                                                       horizontal_alignment='left', vertical_alignment='top',
                                                       path_eff=True, path_err_linewidth=3, path_eff_color='white')
        plotting_tools.StrTools.display_text_in_corner(ax=ax_img_overview, text=fig_dict['overview_blue_band'],
                                                       fontsize=fig_dict['overview_title_font_size'],
                                                       text_color='blue', x_frac=0.02, y_frac=0.83,
                                                       horizontal_alignment='left', vertical_alignment='top',
                                                       path_eff=True, path_err_linewidth=3, path_eff_color='white')

        ax_img_overview.set_title(self.phot_hst_target_name.upper(), fontsize=fig_dict['overview_title_font_size'])
        plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_img_overview, ra_tick_label=True, dec_tick_label=True,
                        ra_axis_label='R.A. (2000.0)', dec_axis_label='DEC. (2000.0)',
                        ra_minpad=0.8, dec_minpad=0.8, ra_tick_color='white', dec_tick_color='white', ra_label_color='k', dec_label_color='k',
                        fontsize=fig_dict['overview_label_size'], labelsize=fig_dict['overview_label_size'],
                        ra_minor_ticks=True, dec_minor_ticks=True)
        if (ra_box is not None) & (dec_box is not None):
            plotting_tools.WCSPlottingTools.draw_box(ax=ax_img_overview, wcs=wcs_overview,
                                                     coord=SkyCoord(ra=ra_box*u.deg, dec=dec_box*u.deg),
                                                     box_size=fig_dict['env_cutout_size'], color='red', line_style='--')

        plotting_tools.WCSPlottingTools.plot_img_scale_bar(
            ax=ax_img_overview, img_shape=img_overview.shape, wcs=wcs_overview,
            bar_length=fig_dict['overview_scale_bar_length'], length_unit='kpc',
            phangs_target=self.phot_target_name,
            bar_color='white', text_color='white',
            line_width=4, fontsize=fig_dict['overview_label_size'],
            va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)

    def get_target_overview_rgb_img(self,
                                    red_band, green_band, blue_band,
                                    red_obs='hst', green_obs='hst', blue_obs='hst',

                                    ref_filter='red',
                                    overview_img_size=(500, 500), **rgb_img_kwargs):
        """
        Function to create an overview RGB image of PHANGS HST observations

        Parameters
        ----------
        red_band, green_band, blue_band : str
            Can be specified to any hst band
        ref_filter: str
            Band which is used for the image limits. can be red green or blue
        overview_img_size : tuple
            denotes the shape of the new image

        Returns
        -------
        rgb_image: ``numpy.ndarray``
        wcs: ``astropy.wcs.WCS``
        """


        # band list need to be loaded
        self.load_phangs_bands(band_list=[red_band, green_band, blue_band], flux_unit='MJy/sr', load_err=False)
        # get overview image

        non_zero_elements = np.where(getattr(self, '%s_bands_data' % eval('%s_obs' % ref_filter))['%s_data_img' % eval('%s_band' % ref_filter)] != 0)

        min_index_ra_axis_x_val = int(np.mean(non_zero_elements[1][non_zero_elements[1] ==
                                                                   np.min(non_zero_elements[1])]))
        min_index_ra_axis_y_val = int(np.mean(non_zero_elements[0][non_zero_elements[1] ==
                                                                   np.min(non_zero_elements[1])]))
        max_index_ra_axis_x_val = int(np.mean(non_zero_elements[1][non_zero_elements[1] ==
                                                                   np.max(non_zero_elements[1])]))
        max_index_ra_axis_y_val = int(np.mean(non_zero_elements[0][non_zero_elements[1] ==
                                                                   np.max(non_zero_elements[1])]))
        min_index_dec_axis_x_val = int(np.mean(non_zero_elements[1][non_zero_elements[0] ==
                                                                    np.min(non_zero_elements[0])]))
        min_index_dec_axis_y_val = int(np.mean(non_zero_elements[0][non_zero_elements[0] ==
                                                                    np.min(non_zero_elements[0])]))
        max_index_dec_axis_x_val = int(np.mean(non_zero_elements[1][non_zero_elements[0] ==
                                                                    np.max(non_zero_elements[0])]))
        max_index_dec_axis_y_val = int(np.mean(non_zero_elements[0][non_zero_elements[0] ==
                                                                    np.max(non_zero_elements[0])]))

        ra_max = getattr(self, '%s_bands_data' % eval('%s_obs' % ref_filter))['%s_wcs_img' % eval('%s_band' % ref_filter)].pixel_to_world(
            min_index_ra_axis_x_val, min_index_ra_axis_y_val).ra.value
        ra_min = getattr(self, '%s_bands_data' % eval('%s_obs' % ref_filter))['%s_wcs_img' % eval('%s_band' % ref_filter)].pixel_to_world(
            max_index_ra_axis_x_val, max_index_ra_axis_y_val).ra.value

        dec_min = getattr(self, '%s_bands_data' % eval('%s_obs' % ref_filter))['%s_wcs_img' % eval('%s_band' % ref_filter)].pixel_to_world(
            min_index_dec_axis_x_val, min_index_dec_axis_y_val).dec.value
        dec_max = getattr(self, '%s_bands_data' % eval('%s_obs' % ref_filter))['%s_wcs_img' % eval('%s_band' % ref_filter)].pixel_to_world(
            max_index_dec_axis_x_val, max_index_dec_axis_y_val).dec.value

        new_wcs = helper_func.CoordTools.construct_wcs(ra_min=ra_min, ra_max=ra_max, dec_min=dec_max, dec_max=dec_min,
                                                       img_shape=overview_img_size, quadratic_image=True)

        img_data_red = helper_func.CoordTools.reproject_image(data=getattr(self, '%s_bands_data' % red_obs)['%s_data_img' % red_band],
                                                              wcs=getattr(self, '%s_bands_data' % red_obs)['%s_wcs_img' % red_band],
                                                              new_wcs=new_wcs, new_shape=overview_img_size)
        img_data_green = helper_func.CoordTools.reproject_image(data=getattr(self, '%s_bands_data' % green_obs)['%s_data_img' % green_band],
                                                                wcs=getattr(self, '%s_bands_data' % green_obs)['%s_wcs_img' % green_band],
                                                                new_wcs=new_wcs, new_shape=overview_img_size)
        img_data_blue = helper_func.CoordTools.reproject_image(data=getattr(self, '%s_bands_data' % blue_obs)['%s_data_img' % blue_band],
                                                               wcs=getattr(self, '%s_bands_data' % blue_obs)['%s_wcs_img' % blue_band],
                                                               new_wcs=new_wcs, new_shape=overview_img_size)

        img_data_red[img_data_red == 0] = np.nan
        img_data_green[img_data_green == 0] = np.nan
        img_data_blue[img_data_blue == 0] = np.nan

        hst_rgb = plotting_tools.ImgTools.get_rgb_img(data_r=img_data_red, data_g=img_data_green, data_b=img_data_blue,
                                                      **rgb_img_kwargs)

        return hst_rgb, new_wcs

    def get_target_non_quadratic_rgb_img(self, ra_min, ra_max,  dec_max, dec_min,
                                         red_band, green_band, blue_band,
                                         red_obs='hst', green_obs='hst', blue_obs='hst',
                                         overview_img_size=(500, 500), **rgb_img_kwargs):
        """
        Function to create an overview RGB image of PHANGS HST observations

        Parameters
        ----------
        red_band, green_band, blue_band : str
            Can be specified to any hst band
        overview_img_size : tuple
            denotes the shape of the new image

        Returns
        -------
        rgb_image: ``numpy.ndarray``
        wcs: ``astropy.wcs.WCS``
        """


        # band list need to be loaded
        self.load_phangs_bands(band_list=[red_band, green_band, blue_band], flux_unit='MJy/sr', load_err=False)
        # get overview image



        new_wcs = helper_func.CoordTools.construct_wcs(ra_min=ra_min, ra_max=ra_max, dec_min=dec_max, dec_max=dec_min,
                                                       img_shape=overview_img_size, quadratic_image=False)

        img_data_red = helper_func.CoordTools.reproject_image(data=getattr(self, '%s_bands_data' % red_obs)['%s_data_img' % red_band],
                                                              wcs=getattr(self, '%s_bands_data' % red_obs)['%s_wcs_img' % red_band],
                                                              new_wcs=new_wcs, new_shape=overview_img_size)
        img_data_green = helper_func.CoordTools.reproject_image(data=getattr(self, '%s_bands_data' % green_obs)['%s_data_img' % green_band],
                                                                wcs=getattr(self, '%s_bands_data' % green_obs)['%s_wcs_img' % green_band],
                                                                new_wcs=new_wcs, new_shape=overview_img_size)
        img_data_blue = helper_func.CoordTools.reproject_image(data=getattr(self, '%s_bands_data' % blue_obs)['%s_data_img' % blue_band],
                                                               wcs=getattr(self, '%s_bands_data' % blue_obs)['%s_wcs_img' % blue_band],
                                                               new_wcs=new_wcs, new_shape=overview_img_size)

        img_data_red[img_data_red == 0] = np.nan
        img_data_green[img_data_green == 0] = np.nan
        img_data_blue[img_data_blue == 0] = np.nan

        hst_rgb = plotting_tools.ImgTools.get_rgb_img(data_r=img_data_red, data_g=img_data_green, data_b=img_data_blue,
                                                      **rgb_img_kwargs)

        return hst_rgb, new_wcs

    def plot_zoom_in_panel_group(self, fig, fig_dict, ra, dec, obs_list=None, nrows=2, ncols=3):

        if obs_list is None:
            obs_list = ['hst_broad_band', 'hst_ha', 'astrosat', 'nircam', 'miri', 'alma']

        row_idx, col_idx = nrows-1, 0
        for idx, obs_type in enumerate(obs_list):

            if obs_type in ['hst_broad_band', 'hst_ha', 'nircam', 'miri']:

                obs = obs_type.split('_')[0]
                if obs == 'hst':
                    # check if object has hst Ha observation
                    if (fig_dict['%s_red_band' % obs_type] == 'Ha') & (not helper_func.ObsTools.check_hst_ha_cont_sub_obs(target=self.phot_hst_target_name)):
                        continue
                    band_list = [helper_func.ObsTools.filter_name2hst_band(target=self.phot_hst_target_name,
                                                                                     filter_name=fig_dict[
                                                                                         '%s_red_band' % obs_type]),
                                          helper_func.ObsTools.filter_name2hst_band(target=self.phot_hst_target_name,
                                                                                     filter_name=fig_dict[
                                                                                         '%s_green_band' % obs_type]),
                                          helper_func.ObsTools.filter_name2hst_band(target=self.phot_hst_target_name,
                                                                                     filter_name=fig_dict[
                                                                                         '%s_blue_band' % obs_type])]
                else:
                    band_list = [fig_dict['%s_red_band' % obs_type], fig_dict['%s_green_band' % obs_type],
                                 fig_dict['%s_blue_band' % obs_type]]

                # check if object is covered
                if (self.check_coords_covered_by_band(
                        telescope=obs, ra=ra, dec=dec, band=band_list[0], max_dist_dist2hull_arcsec=2) &
                        self.check_coords_covered_by_band(
                            telescope=obs, ra=ra, dec=dec, band=band_list[1], max_dist_dist2hull_arcsec=2) &
                        self.check_coords_covered_by_band(
                            telescope=obs, ra=ra, dec=dec, band=band_list[2], max_dist_dist2hull_arcsec=2)):
                    self.load_phangs_bands(band_list=band_list, flux_unit='MJy/sr', load_err=False)

                    img_zoom_in, wcs_zoom_in = self.get_rgb_zoom_in(ra=ra, dec=dec, cutout_size=fig_dict['env_cutout_size'],
                                                                    band_red=band_list[0],
                                                                    band_green=band_list[1], band_blue=band_list[2])

                    ax_zoom_in = plotting_tools.AxisTools.add_panel_axis(
                        fig=fig,
                        left_align=fig_dict['zoom_in_left_align'],
                        bottom_align=fig_dict['zoom_in_bottom_align'],
                        width=fig_dict['zoom_in_width'],
                        height=fig_dict['zoom_in_height'],
                        space_vertical=fig_dict['zoom_in_space_vertical'],
                        space_horizontal=fig_dict['zoom_in_space_horizontal'],
                           row_idx=row_idx, col_idx=col_idx, projection=wcs_zoom_in)

                    ax_zoom_in.imshow(img_zoom_in)
                    # arrange axis
                    # add text
                    plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in, text=obs.upper(),
                                                                   fontsize=fig_dict['zoom_in_title_font_size'],
                                                                   text_color='white', x_frac=0.02, y_frac=0.98,
                                                                   horizontal_alignment='left',
                                                                   vertical_alignment='top',
                                                                   path_eff=True, path_err_linewidth=3,
                                                                   path_eff_color='white')
                    plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in,
                                                                   text=fig_dict['%s_red_band' % obs_type],
                                                                   fontsize=fig_dict['zoom_in_title_font_size'],
                                                                   text_color='red', x_frac=0.02, y_frac=0.91,
                                                                   horizontal_alignment='left',
                                                                   vertical_alignment='top',
                                                                   path_eff=True, path_err_linewidth=3,
                                                                   path_eff_color='white')
                    plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in,
                                                                   text=fig_dict['%s_green_band' % obs_type],
                                                                   fontsize=fig_dict['zoom_in_title_font_size'],
                                                                   text_color='green', x_frac=0.02, y_frac=0.84,
                                                                   horizontal_alignment='left',
                                                                   vertical_alignment='top',
                                                                   path_eff=True, path_err_linewidth=3,
                                                                   path_eff_color='white')
                    plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in,
                                                                   text=fig_dict['%s_blue_band' % obs_type],
                                                                   fontsize=fig_dict['zoom_in_title_font_size'],
                                                                   text_color='blue', x_frac=0.02, y_frac=0.77,
                                                                   horizontal_alignment='left',
                                                                   vertical_alignment='top',
                                                                   path_eff=True, path_err_linewidth=3,
                                                                   path_eff_color='white')

                    plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_zoom_in, ra_tick_label=False,
                                                                    dec_tick_label=False,
                                                                    ra_axis_label=' ',
                                                                    dec_axis_label=' ',
                                                                    ra_minpad=0.8, dec_minpad=0.8, ra_tick_color='white', dec_tick_color='white',
                                                                    ra_label_color='k', dec_label_color='k',
                                                                    fontsize=fig_dict['zoom_in_label_size'],
                                                                    labelsize=fig_dict['zoom_in_label_size'],
                                                                    ra_minor_ticks=True, dec_minor_ticks=True)

                    if (col_idx == 0) & (row_idx == nrows-1):
                        plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                            ax=ax_zoom_in, img_shape=img_zoom_in.shape, wcs=wcs_zoom_in,
                            bar_length=fig_dict['zoom_in_scale_bar_length_1'], length_unit='pc',
                            phangs_target=self.phot_target_name,
                            bar_color='tab:red', text_color='tab:red',
                            line_width=4, fontsize=fig_dict['zoom_in_label_size'],
                            va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)

                        plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                            ax=ax_zoom_in, img_shape=img_zoom_in.shape, wcs=wcs_zoom_in,
                            bar_length=fig_dict['zoom_in_scale_bar_length_2'], length_unit='arcsec',
                            phangs_target=self.phot_target_name,
                            bar_color='tab:red', text_color='tab:red',
                            line_width=4, fontsize=fig_dict['zoom_in_label_size'],
                            va='top', ha='right', x_offset=0.05, y_offset=0.15, text_y_offset_diff=0.01)

            elif obs_type == 'astrosat':
                # check if object is covered
                if helper_func.ObsTools.check_astrosat_obs(target=self.phot_astrosat_target_name):
                    astrosat_band_list = helper_func.ObsTools.get_astrosat_obs_band_list(target=self.phot_astrosat_target_name)
                    if fig_dict['astrosat_band'] in astrosat_band_list:
                        astrosat_band = fig_dict['astrosat_band']
                    else:
                        astrosat_band = astrosat_band_list[0]
                else:
                    continue

                if self.check_coords_covered_by_band(telescope="astrosat", ra=ra, dec=dec, band=astrosat_band,
                                                     max_dist_dist2hull_arcsec=2):
                    self.load_phangs_bands(band_list=astrosat_band, flux_unit='erg A-1 cm-2 s-1', load_err=False)
                    cutout_dict = self.get_band_cutout_dict(ra_cutout=ra, dec_cutout=dec,
                                                            cutout_size=fig_dict['env_cutout_size'],
                                                            band_list=[astrosat_band])
                    ax_zoom_in = plotting_tools.AxisTools.add_panel_axis(
                        fig=fig,
                        left_align=fig_dict['zoom_in_left_align'],
                        bottom_align=fig_dict['zoom_in_bottom_align'],
                        width=fig_dict['zoom_in_width'],
                        height=fig_dict['zoom_in_height'],
                        space_vertical=fig_dict['zoom_in_space_vertical'],
                        space_horizontal=fig_dict['zoom_in_space_horizontal'],
                        row_idx=row_idx, col_idx=col_idx,
                        projection=cutout_dict['%s_img_cutout' % astrosat_band].wcs)

                    min_astrosat_value = np.nanmin(cutout_dict['%s_img_cutout' % astrosat_band].data)
                    max_astrosat_value = np.nanmax(cutout_dict['%s_img_cutout' % astrosat_band].data)
                    if min_astrosat_value <= 0:
                        min_astrosat_value = max_astrosat_value / 100
                    if fig_dict['astrosat_norm'] == 'log':
                        norm = LogNorm(min_astrosat_value, max_astrosat_value)
                    else:
                        norm = Normalize(min_astrosat_value, max_astrosat_value)

                    ax_zoom_in.imshow(cutout_dict['%s_img_cutout' % astrosat_band].data, norm=norm,
                                      cmap=fig_dict['astrosat_cmap'])

                    plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in, text='ASTROSAT',
                                                                   fontsize=fig_dict['zoom_in_title_font_size'],
                                                                   text_color='white', x_frac=0.02, y_frac=0.98,
                                                                   horizontal_alignment='left',
                                                                   vertical_alignment='top',
                                                                   path_eff=True, path_err_linewidth=3,
                                                                   path_eff_color='red')
                    plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in,
                                                                   text=astrosat_band,
                                                                   fontsize=fig_dict['zoom_in_title_font_size'],
                                                                   text_color='red', x_frac=0.02, y_frac=0.91,
                                                                   horizontal_alignment='left',
                                                                   vertical_alignment='top',
                                                                   path_eff=True, path_err_linewidth=3,
                                                                   path_eff_color='red')
                    plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_zoom_in, ra_tick_label=False,
                                                                    dec_tick_label=False,
                                                                    ra_axis_label=' ',
                                                                    dec_axis_label=' ',
                                                                    ra_minpad=0.8, dec_minpad=0.8, ra_tick_color='white', dec_tick_color='white',
                                                                    ra_label_color='k', dec_label_color='k',
                                                                    fontsize=fig_dict['zoom_in_label_size'],
                                                                    labelsize=fig_dict['zoom_in_label_size'],
                                                                    ra_minor_ticks=True, dec_minor_ticks=True)
                    ax_cbar_astrosat = fig.add_axes([fig_dict['astrosat_cbar_left_align'],
                                                         fig_dict['astrosat_cbar_bottom_align'],
                                                         fig_dict['astrosat_cbar_width'],
                                                         fig_dict['astrosat_cbar_height']])
                    plotting_tools.ColorBarTools.create_cbar(ax_cbar=ax_cbar_astrosat, cmap=fig_dict['astrosat_cmap'], norm=norm,
                                                cbar_label=r'$\phi$ [erg $\AA^{-1}$ cm$^{-2}$ s$^{-2}$}',
                                                fontsize=fig_dict['zoom_in_label_size'],
                                                ticks=None, labelpad=2, tick_width=2, orientation='vertical',
                                                extend='neither')

            elif obs_type == 'alma':

                # check if target is observed
                if self.gas_target_name in phangs_info.full_alma_galaxy_list:

                    # check if object is covered
                    if self.check_coords_covered_by_alma(ra=ra, dec=dec, res=fig_dict['alma_res'], max_dist_dist2hull_arcsec=2):

                        alma_h2_map_data, alma_h2_map_wcs = self.get_alma_h2_map(
                            res=fig_dict['alma_res'], alpha_co_method=fig_dict['alma_alpha_co_method'])
                        # get cutout
                        alma_h2_map_cutout = helper_func.CoordTools.get_img_cutout(
                            img=alma_h2_map_data, wcs=alma_h2_map_wcs, coord=SkyCoord(ra=ra*u.deg, dec=dec*u.deg),
                            cutout_size=fig_dict['env_cutout_size'])
                        # make sure the alma data is covered:
                        if (np.all(alma_h2_map_cutout.data == 0) | np.all(np.isnan(alma_h2_map_cutout.data)) |
                                np.all(np.isnan(alma_h2_map_cutout.data) + (alma_h2_map_cutout.data == 0))):
                            print('no alma coverage')
                            break
                        ax_zoom_in = plotting_tools.AxisTools.add_panel_axis(
                            fig=fig,
                            left_align=fig_dict['zoom_in_left_align'],
                            bottom_align=fig_dict['zoom_in_bottom_align'],
                            width=fig_dict['zoom_in_width'],
                            height=fig_dict['zoom_in_height'],
                            space_vertical=fig_dict['zoom_in_space_vertical'],
                            space_horizontal=fig_dict['zoom_in_space_horizontal'],
                            row_idx=row_idx, col_idx=col_idx,
                            projection=alma_h2_map_cutout.wcs)

                        min_alma_value = np.nanmin(alma_h2_map_cutout.data)
                        max_alma_value = np.nanmax(alma_h2_map_cutout.data)

                        if min_alma_value <= 0:
                            min_alma_value = max_alma_value / 100
                        if fig_dict['alma_norm'] == 'log':
                            norm = LogNorm(min_alma_value, max_alma_value)
                        else:
                            norm = Normalize(min_alma_value, max_alma_value)

                        ax_zoom_in.imshow(alma_h2_map_cutout.data, norm=norm, cmap=fig_dict['alma_cmap'])
                        plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in, text='ALMA',
                                                                       fontsize=fig_dict['zoom_in_title_font_size'],
                                                                       text_color='white', x_frac=0.02, y_frac=0.98,
                                                                       horizontal_alignment='left',
                                                                       vertical_alignment='top',
                                                                       path_eff=True, path_err_linewidth=3,
                                                                       path_eff_color='red')
                        plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_zoom_in, ra_tick_label=False,
                                                                        dec_tick_label=False,
                                                                        ra_axis_label=' ',
                                                                        dec_axis_label=' ',
                                                                        ra_minpad=0.8, dec_minpad=0.8,
                                                                        ra_tick_color='white', dec_tick_color='white',
                                                                        ra_label_color='k', dec_label_color='k',
                                                                        fontsize=fig_dict['zoom_in_label_size'],
                                                                        labelsize=fig_dict['zoom_in_label_size'],
                                                                        ra_minor_ticks=True, dec_minor_ticks=True)
                        ax_cbar_alma = fig.add_axes([fig_dict['alma_cbar_left_align'],
                                                         fig_dict['alma_cbar_bottom_align'],
                                                         fig_dict['alma_cbar_width'],
                                                         fig_dict['alma_cbar_height']])
                        plotting_tools.ColorBarTools.create_cbar(ax_cbar=ax_cbar_alma, cmap=fig_dict['alma_cmap'],
                                                                 norm=norm,
                                                                 cbar_label=r'log($\Sigma_{\rm H2}$/[M$_{\odot}$ kpc$^{-2}$])',
                                                                 fontsize=fig_dict['zoom_in_label_size'],
                                                                 ticks=None, labelpad=2, tick_width=2, orientation='vertical',
                                                                 extend='neither')

            col_idx +=1
            if col_idx == ncols:
                col_idx =0
                row_idx -= 1


    def plot_zoom_in_panel_group_extra_nircam(self, fig, fig_dict, ra, dec, obs_list=None, nrows=2, ncols=3):

        if obs_list is None:
            obs_list = ['hst_broad_band', 'hst_ha', 'astrosat', 'nircam', 'nircam2', 'nircam3', 'miri', 'alma']

        row_idx, col_idx = nrows-1, 0
        for idx, obs_type in enumerate(obs_list):

            if obs_type in ['hst_broad_band', 'hst_ha', 'nircam', 'nircam2', 'nircam3', 'miri']:

                if obs_type in ['nircam2', 'nircam3']:
                    obs = 'nircam'
                else:
                    obs = obs_type.split('_')[0]

                if obs == 'hst':
                    # check if object has hst Ha observation
                    if (fig_dict['%s_red_band' % obs_type] == 'Ha') & (not helper_func.ObsTools.check_hst_ha_cont_sub_obs(target=self.phot_hst_target_name)):
                        continue
                    band_list = [helper_func.ObsTools.filter_name2hst_band(target=self.phot_hst_target_name,
                                                                                     filter_name=fig_dict[
                                                                                         '%s_red_band' % obs_type]),
                                          helper_func.ObsTools.filter_name2hst_band(target=self.phot_hst_target_name,
                                                                                     filter_name=fig_dict[
                                                                                         '%s_green_band' % obs_type]),
                                          helper_func.ObsTools.filter_name2hst_band(target=self.phot_hst_target_name,
                                                                                     filter_name=fig_dict[
                                                                                         '%s_blue_band' % obs_type])]
                else:
                    band_list = [fig_dict['%s_red_band' % obs_type], fig_dict['%s_green_band' % obs_type],
                                 fig_dict['%s_blue_band' % obs_type]]
                # check if object is covered
                if (self.check_coords_covered_by_band(
                        telescope=obs, ra=ra, dec=dec, band=band_list[0], max_dist_dist2hull_arcsec=2) &
                        self.check_coords_covered_by_band(
                            telescope=obs, ra=ra, dec=dec, band=band_list[1], max_dist_dist2hull_arcsec=2) &
                        self.check_coords_covered_by_band(
                            telescope=obs, ra=ra, dec=dec, band=band_list[2], max_dist_dist2hull_arcsec=2)):
                    self.load_phangs_bands(band_list=band_list, flux_unit='MJy/sr', load_err=False)

                    img_zoom_in, wcs_zoom_in = self.get_rgb_zoom_in(ra=ra, dec=dec, cutout_size=fig_dict['env_cutout_size'],
                                                                    band_red=band_list[0],
                                                                    band_green=band_list[1], band_blue=band_list[2])

                    ax_zoom_in = plotting_tools.AxisTools.add_panel_axis(
                        fig=fig,
                        left_align=fig_dict['zoom_in_left_align'],
                        bottom_align=fig_dict['zoom_in_bottom_align'],
                        width=fig_dict['zoom_in_width'],
                        height=fig_dict['zoom_in_height'],
                        space_vertical=fig_dict['zoom_in_space_vertical'],
                        space_horizontal=fig_dict['zoom_in_space_horizontal'],
                           row_idx=row_idx, col_idx=col_idx, projection=wcs_zoom_in)

                    ax_zoom_in.imshow(img_zoom_in)
                    # arrange axis
                    # add text
                    plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in, text=obs.upper(),
                                                                   fontsize=fig_dict['zoom_in_title_font_size'],
                                                                   text_color='white', x_frac=0.02, y_frac=0.98,
                                                                   horizontal_alignment='left',
                                                                   vertical_alignment='top',
                                                                   path_eff=True, path_err_linewidth=3,
                                                                   path_eff_color='white')
                    plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in,
                                                                   text=fig_dict['%s_red_band' % obs_type],
                                                                   fontsize=fig_dict['zoom_in_title_font_size'],
                                                                   text_color='red', x_frac=0.02, y_frac=0.91,
                                                                   horizontal_alignment='left',
                                                                   vertical_alignment='top',
                                                                   path_eff=True, path_err_linewidth=3,
                                                                   path_eff_color='white')
                    plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in,
                                                                   text=fig_dict['%s_green_band' % obs_type],
                                                                   fontsize=fig_dict['zoom_in_title_font_size'],
                                                                   text_color='green', x_frac=0.02, y_frac=0.84,
                                                                   horizontal_alignment='left',
                                                                   vertical_alignment='top',
                                                                   path_eff=True, path_err_linewidth=3,
                                                                   path_eff_color='white')
                    plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in,
                                                                   text=fig_dict['%s_blue_band' % obs_type],
                                                                   fontsize=fig_dict['zoom_in_title_font_size'],
                                                                   text_color='blue', x_frac=0.02, y_frac=0.77,
                                                                   horizontal_alignment='left',
                                                                   vertical_alignment='top',
                                                                   path_eff=True, path_err_linewidth=3,
                                                                   path_eff_color='white')

                    plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_zoom_in, ra_tick_label=False,
                                                                    dec_tick_label=False,
                                                                    ra_axis_label=' ',
                                                                    dec_axis_label=' ',
                                                                    ra_minpad=0.8, dec_minpad=0.8, ra_tick_color='white', dec_tick_color='white',
                                                                    ra_label_color='k', dec_label_color='k',
                                                                    fontsize=fig_dict['zoom_in_label_size'],
                                                                    labelsize=fig_dict['zoom_in_label_size'],
                                                                    ra_minor_ticks=True, dec_minor_ticks=True)

                    if (col_idx == 0) & (row_idx == nrows-1):
                        plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                            ax=ax_zoom_in, img_shape=img_zoom_in.shape, wcs=wcs_zoom_in,
                            bar_length=fig_dict['zoom_in_scale_bar_length_1'], length_unit='pc',
                            phangs_target=self.phot_target_name,
                            bar_color='tab:red', text_color='tab:red',
                            line_width=4, fontsize=fig_dict['zoom_in_label_size'],
                            va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)

                        plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                            ax=ax_zoom_in, img_shape=img_zoom_in.shape, wcs=wcs_zoom_in,
                            bar_length=fig_dict['zoom_in_scale_bar_length_2'], length_unit='arcsec',
                            phangs_target=self.phot_target_name,
                            bar_color='tab:red', text_color='tab:red',
                            line_width=4, fontsize=fig_dict['zoom_in_label_size'],
                            va='top', ha='right', x_offset=0.05, y_offset=0.15, text_y_offset_diff=0.01)

            elif obs_type == 'astrosat':
                # check if object is covered
                if helper_func.ObsTools.check_astrosat_obs(target=self.phot_astrosat_target_name):
                    astrosat_band_list = helper_func.ObsTools.get_astrosat_obs_band_list(target=self.phot_astrosat_target_name)
                    if fig_dict['astrosat_band'] in astrosat_band_list:
                        astrosat_band = fig_dict['astrosat_band']
                    else:
                        astrosat_band = astrosat_band_list[0]
                else:
                    continue

                if self.check_coords_covered_by_band(telescope="astrosat", ra=ra, dec=dec, band=astrosat_band,
                                                     max_dist_dist2hull_arcsec=2):
                    self.load_phangs_bands(band_list=astrosat_band, flux_unit='erg A-1 cm-2 s-1', load_err=False)
                    cutout_dict = self.get_band_cutout_dict(ra_cutout=ra, dec_cutout=dec,
                                                            cutout_size=fig_dict['env_cutout_size'],
                                                            band_list=[astrosat_band])
                    ax_zoom_in = plotting_tools.AxisTools.add_panel_axis(
                        fig=fig,
                        left_align=fig_dict['zoom_in_left_align'],
                        bottom_align=fig_dict['zoom_in_bottom_align'],
                        width=fig_dict['zoom_in_width'],
                        height=fig_dict['zoom_in_height'],
                        space_vertical=fig_dict['zoom_in_space_vertical'],
                        space_horizontal=fig_dict['zoom_in_space_horizontal'],
                        row_idx=row_idx, col_idx=col_idx,
                        projection=cutout_dict['%s_img_cutout' % astrosat_band].wcs)

                    min_astrosat_value = np.nanmin(cutout_dict['%s_img_cutout' % astrosat_band].data)
                    max_astrosat_value = np.nanmax(cutout_dict['%s_img_cutout' % astrosat_band].data)
                    if min_astrosat_value <= 0:
                        min_astrosat_value = max_astrosat_value / 100
                    if fig_dict['astrosat_norm'] == 'log':
                        norm = LogNorm(min_astrosat_value, max_astrosat_value)
                    else:
                        norm = Normalize(min_astrosat_value, max_astrosat_value)

                    ax_zoom_in.imshow(cutout_dict['%s_img_cutout' % astrosat_band].data, norm=norm,
                                      cmap=fig_dict['astrosat_cmap'])

                    plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in, text='ASTROSAT',
                                                                   fontsize=fig_dict['zoom_in_title_font_size'],
                                                                   text_color='white', x_frac=0.02, y_frac=0.98,
                                                                   horizontal_alignment='left',
                                                                   vertical_alignment='top',
                                                                   path_eff=True, path_err_linewidth=3,
                                                                   path_eff_color='red')
                    plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in,
                                                                   text=astrosat_band,
                                                                   fontsize=fig_dict['zoom_in_title_font_size'],
                                                                   text_color='red', x_frac=0.02, y_frac=0.91,
                                                                   horizontal_alignment='left',
                                                                   vertical_alignment='top',
                                                                   path_eff=True, path_err_linewidth=3,
                                                                   path_eff_color='red')
                    plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_zoom_in, ra_tick_label=False,
                                                                    dec_tick_label=False,
                                                                    ra_axis_label=' ',
                                                                    dec_axis_label=' ',
                                                                    ra_minpad=0.8, dec_minpad=0.8, ra_tick_color='white', dec_tick_color='white',
                                                                    ra_label_color='k', dec_label_color='k',
                                                                    fontsize=fig_dict['zoom_in_label_size'],
                                                                    labelsize=fig_dict['zoom_in_label_size'],
                                                                    ra_minor_ticks=True, dec_minor_ticks=True)
                    ax_cbar_astrosat = fig.add_axes([fig_dict['astrosat_cbar_left_align'],
                                                         fig_dict['astrosat_cbar_bottom_align'],
                                                         fig_dict['astrosat_cbar_width'],
                                                         fig_dict['astrosat_cbar_height']])
                    plotting_tools.ColorBarTools.create_cbar(ax_cbar=ax_cbar_astrosat, cmap=fig_dict['astrosat_cmap'], norm=norm,
                                                cbar_label=r'$\phi$ [erg $\AA^{-1}$ cm$^{-2}$ s$^{-2}$}',
                                                fontsize=fig_dict['zoom_in_label_size'],
                                                ticks=None, labelpad=2, tick_width=2, orientation='vertical',
                                                extend='neither')

            elif obs_type == 'alma':

                # check if target is observed
                if self.gas_target_name in phangs_info.full_alma_galaxy_list:

                    # check if object is covered
                    if self.check_coords_covered_by_alma(ra=ra, dec=dec, res=fig_dict['alma_res'], max_dist_dist2hull_arcsec=2):

                        alma_h2_map_data, alma_h2_map_wcs = self.get_alma_h2_map(
                            res=fig_dict['alma_res'], alpha_co_method=fig_dict['alma_alpha_co_method'])
                        # get cutout
                        alma_h2_map_cutout = helper_func.CoordTools.get_img_cutout(
                            img=alma_h2_map_data, wcs=alma_h2_map_wcs, coord=SkyCoord(ra=ra*u.deg, dec=dec*u.deg),
                            cutout_size=fig_dict['env_cutout_size'])
                        # make sure the alma data is covered:
                        if (np.all(alma_h2_map_cutout.data == 0) | np.all(np.isnan(alma_h2_map_cutout.data)) |
                                np.all(np.isnan(alma_h2_map_cutout.data) + (alma_h2_map_cutout.data == 0))):
                            print('no alma coverage')
                            break
                        ax_zoom_in = plotting_tools.AxisTools.add_panel_axis(
                            fig=fig,
                            left_align=fig_dict['zoom_in_left_align'],
                            bottom_align=fig_dict['zoom_in_bottom_align'],
                            width=fig_dict['zoom_in_width'],
                            height=fig_dict['zoom_in_height'],
                            space_vertical=fig_dict['zoom_in_space_vertical'],
                            space_horizontal=fig_dict['zoom_in_space_horizontal'],
                            row_idx=row_idx, col_idx=col_idx,
                            projection=alma_h2_map_cutout.wcs)

                        min_alma_value = np.nanmin(alma_h2_map_cutout.data)
                        max_alma_value = np.nanmax(alma_h2_map_cutout.data)

                        if min_alma_value <= 0:
                            min_alma_value = max_alma_value / 100
                        if fig_dict['alma_norm'] == 'log':
                            norm = LogNorm(min_alma_value, max_alma_value)
                        else:
                            norm = Normalize(min_alma_value, max_alma_value)

                        ax_zoom_in.imshow(alma_h2_map_cutout.data, norm=norm, cmap=fig_dict['alma_cmap'])
                        plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in, text='ALMA',
                                                                       fontsize=fig_dict['zoom_in_title_font_size'],
                                                                       text_color='white', x_frac=0.02, y_frac=0.98,
                                                                       horizontal_alignment='left',
                                                                       vertical_alignment='top',
                                                                       path_eff=True, path_err_linewidth=3,
                                                                       path_eff_color='red')
                        plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_zoom_in, ra_tick_label=False,
                                                                        dec_tick_label=False,
                                                                        ra_axis_label=' ',
                                                                        dec_axis_label=' ',
                                                                        ra_minpad=0.8, dec_minpad=0.8,
                                                                        ra_tick_color='white', dec_tick_color='white',
                                                                        ra_label_color='k', dec_label_color='k',
                                                                        fontsize=fig_dict['zoom_in_label_size'],
                                                                        labelsize=fig_dict['zoom_in_label_size'],
                                                                        ra_minor_ticks=True, dec_minor_ticks=True)
                        ax_cbar_alma = fig.add_axes([fig_dict['alma_cbar_left_align'],
                                                         fig_dict['alma_cbar_bottom_align'],
                                                         fig_dict['alma_cbar_width'],
                                                         fig_dict['alma_cbar_height']])
                        plotting_tools.ColorBarTools.create_cbar(ax_cbar=ax_cbar_alma, cmap=fig_dict['alma_cmap'],
                                                                 norm=norm,
                                                                 cbar_label=r'log($\Sigma_{\rm H2}$/[M$_{\odot}$ kpc$^{-2}$])',
                                                                 fontsize=fig_dict['zoom_in_label_size'],
                                                                 ticks=None, labelpad=2, tick_width=2, orientation='vertical',
                                                                 extend='neither')

            col_idx +=1
            if col_idx == ncols:
                col_idx = 0
                row_idx -= 1


    def get_rgb_zoom_in(self, ra, dec, cutout_size, band_red, band_green, band_blue, ref_filter='blue', **kwargs):
        """
        Function to create an RGB image of a zoom in region of PHANGS observations

        Parameters
        ----------
        ra, dec : float
            coordinates in degree
        cutout_size: tuple
            cutout size in arcsec
        circle_rad : float
            radius of circle in which the object of interest is located
        band_red, band_green, band_blue : str
            filter names
        ref_filter : str
            can be blue, green or red. In case the images are not the same size they get reprojected to one frame

        Returns
        -------
        rgb_image, wcs : ``numpy.ndarray``, ``astropy.wcs.WCS``
        """

        self.load_phangs_bands(band_list=[band_red, band_green, band_blue],
                               flux_unit='MJy/sr', load_err=False)

        cutout = self.get_band_cutout_dict(ra_cutout=ra, dec_cutout=dec, cutout_size=cutout_size,
                                           band_list=[band_red, band_green, band_blue])

        if ((cutout['%s_img_cutout' % band_red].data is None) or
                (cutout['%s_img_cutout' % band_green].data is None) or
                (cutout['%s_img_cutout' % band_blue].data is None)):
            return None, None

        ref_wcs = cutout['%s_img_cutout' % eval('band_%s' % ref_filter)].wcs

        if not (cutout['%s_img_cutout' % band_red].data.shape ==
                cutout['%s_img_cutout' % band_green].data.shape ==
                cutout['%s_img_cutout' % band_blue].data.shape):
            new_shape = cutout['%s_img_cutout' % eval('band_%s' % ref_filter)].data.shape
            if ref_filter == 'red':
                cutout_data_red = cutout['%s_img_cutout' % band_red].data
            else:
                cutout_data_red = helper_func.CoordTools.reproject_image(data=cutout['%s_img_cutout' % band_red].data,
                                                                         wcs=cutout['%s_img_cutout' % band_red].wcs,
                                                                         new_wcs=ref_wcs, new_shape=new_shape)
            if ref_filter == 'green':
                cutout_data_green = cutout['%s_img_cutout' % band_green].data
            else:
                cutout_data_green = helper_func.CoordTools.reproject_image(data=cutout['%s_img_cutout' % band_green].data,
                                                                           wcs=cutout['%s_img_cutout' % band_green].wcs,
                                                                           new_wcs=ref_wcs, new_shape=new_shape)
            if ref_filter == 'blue':
                cutout_data_blue = cutout['%s_img_cutout' % band_blue].data
            else:
                cutout_data_blue = helper_func.CoordTools.reproject_image(data=cutout['%s_img_cutout' % band_blue].data,
                                                                          wcs=cutout['%s_img_cutout' % band_blue].wcs,
                                                                          new_wcs=ref_wcs, new_shape=new_shape)
        else:
            cutout_data_red = cutout['%s_img_cutout' % band_red].data
            cutout_data_green = cutout['%s_img_cutout' % band_green].data
            cutout_data_blue = cutout['%s_img_cutout' % band_blue].data

        cutout_rgb_img = plotting_tools.ImgTools.get_rgb_img(data_r=cutout_data_red,
                                          data_g=cutout_data_green,
                                          data_b=cutout_data_blue, **kwargs)

        return cutout_rgb_img, ref_wcs

    def plot_img_stamps(self, fig, fig_dict, ra, dec, plot_rad_profile=True):


        hst_broad_band_list = self.get_covered_hst_broad_band_list(ra=ra, dec=dec)
        hst_ha_band = self.get_covered_hst_ha_band(ra=ra, dec=dec)
        nircam_band_list = self.get_covered_nircam_band_list(ra=ra, dec=dec)
        miri_band_list = self.get_covered_miri_band_list(ra=ra, dec=dec)

        band_list = self.get_covered_hst_broad_band_list(ra=ra, dec=dec)
        # check if H-alpha is available
        if helper_func.ObsTools.check_hst_ha_cont_sub_obs(target=self.phot_hst_ha_cont_sub_target_name):
            if self.check_coords_covered_by_band(telescope="hst", ra=ra, dec=dec, band=helper_func.ObsTools.get_hst_ha_band(target=self.phot_hst_ha_cont_sub_target_name), max_dist_dist2hull_arcsec=2):
                band_list += [hst_ha_band]
                # check if Ha- continuum subtracted image is available
                if self.phot_hst_ha_cont_sub_target_name in phangs_info.hst_ha_cont_sub_dict.keys():
                    band_list += [hst_ha_band + '_cont_sub']
        band_list += nircam_band_list + miri_band_list

        # load data
        self.load_phangs_bands(band_list=band_list, flux_unit='MJy/sr', load_err=True, load_hst=True, load_hst_ha=True,
                              load_nircam=True, load_miri=True, load_astrosat=False)
        # load cutout stamps
        cutout_dict_stamp = self.get_band_cutout_dict(ra_cutout=ra, dec_cutout=dec, cutout_size=fig_dict['stamp_size'],
                                                      band_list=band_list, include_err=True)

        hst_col_index = 0
        for hst_col_index, hst_stamp_band in enumerate(hst_broad_band_list):
            if plot_rad_profile:
                ax_rad_profile = plotting_tools.AxisTools.add_panel_axis(
                            fig=fig,
                            left_align=fig_dict['rad_pro_left_align'],
                            bottom_align=fig_dict['rad_pro_bottom_align'],
                            width=fig_dict['rad_pro_width'],
                            height=fig_dict['rad_pro_height'],
                            space_vertical=fig_dict['rad_pro_space_vertical'],
                            space_horizontal=fig_dict['rad_pro_space_horizontal'],
                               row_idx=1, col_idx=hst_col_index)

                radius, profile, error = phot_tools.ProfileTools.get_rad_profile_from_img(
                    img=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].data,
                    wcs=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].wcs,
                    ra=ra, dec=dec, max_rad_arcsec=0.5, img_err=cutout_dict_stamp['%s_err_cutout' % hst_stamp_band].data)

                psf_dict = phot_tools.PSFTools.load_obs_psf_dict(
                    band=hst_stamp_band, instrument=helper_func.ObsTools.get_hst_instrument(
                        target=self.phot_hst_target_name, band=hst_stamp_band))
                mask_psf_rad = psf_dict['radius_arcsec'] < np.max(radius)
                ax_rad_profile.fill_between(radius, profile-error, profile+error, color='gray', alpha=0.7)
                ax_rad_profile.plot(radius, profile, linewidth=4, color='k')
                ax_rad_profile.plot(psf_dict['radius_arcsec'][mask_psf_rad], psf_dict['psf_profile'][mask_psf_rad] / np.max(psf_dict['psf_profile']), linewidth=4, color='red')

                ax_rad_profile.set_yticklabels([])
                ax_rad_profile.tick_params(axis='both', which='both', width=2, direction='in', labelsize=fig_dict['stamp_label_size'])
                ax_rad_profile.set_title(hst_stamp_band.upper(), fontsize=fig_dict['stamp_title_font_size'],
                                         color=fig_dict['hst_broad_band_color'])
                if hst_col_index == 0:
                    ax_rad_profile.set_xlabel('rad. [\"]', fontsize=fig_dict['stamp_label_size'])

            ax_stamp = plotting_tools.AxisTools.add_panel_axis(
                        fig=fig,
                        left_align=fig_dict['stamp_left_align'],
                        bottom_align=fig_dict['stamp_bottom_align'],
                        width=fig_dict['stamp_width'],
                        height=fig_dict['stamp_height'],
                        space_vertical=fig_dict['stamp_space_vertical'],
                        space_horizontal=fig_dict['stamp_space_horizontal'],
                           row_idx=1, col_idx=hst_col_index, projection=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].wcs)
            norm_hst_stamp = plotting_tools.ColorBarTools.compute_cbar_norm(
                cutout_list=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].data,
                log_scale=True)
            ax_stamp.imshow(
                cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].data, norm=norm_hst_stamp,
                cmap='Greys')
            plotting_tools.WCSPlottingTools.plot_coord_crosshair(ax=ax_stamp,
                                                            pos=SkyCoord(ra=ra * u.deg,
                                                                         dec=dec * u.deg),
                                                            wcs=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].wcs,
                                                            rad=0.3, hair_length=0.3,
                                                            color='red', line_width=2)
            plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_stamp, ra_tick_label=False,
                                 dec_tick_label=False, ra_axis_label=' ', dec_axis_label=' ',
                                 fontsize=fig_dict['stamp_label_size'], labelsize=fig_dict['stamp_label_size'])
            if hst_col_index == 0:
                plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                    ax=ax_stamp, img_shape=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].data.shape,
                    wcs=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].wcs,
                    bar_length=fig_dict['stamp_scale_bar_length_1'], length_unit='arcsec',
                    bar_color='tab:red', text_color='tab:red',
                    line_width=4, fontsize=fig_dict['stamp_label_size'],
                               va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)
            if hst_col_index == 1:
                plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                    ax=ax_stamp, img_shape=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].data.shape,
                    wcs=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].wcs,
                    bar_length=fig_dict['stamp_scale_bar_length_2'], length_unit='pc',
                    phangs_target=self.phot_hst_target_name,
                    bar_color='tab:red', text_color='tab:red',
                    line_width=4, fontsize=fig_dict['stamp_label_size'],
                               va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)


        # add h alpha
        if helper_func.ObsTools.check_hst_ha_cont_sub_obs(target=self.phot_hst_target_name):
            if self.check_coords_covered_by_band(
                    telescope="hst", ra=ra, dec=dec, band=helper_func.ObsTools.get_hst_ha_band(target=self.phot_hst_target_name),
                    max_dist_dist2hull_arcsec=2):
                hst_ha_band = helper_func.ObsTools.get_hst_ha_band(target=self.phot_hst_target_name)
                if plot_rad_profile:
                    ax_rad_profile = plotting_tools.AxisTools.add_panel_axis(
                                fig=fig,
                                left_align=fig_dict['rad_pro_left_align'],
                                bottom_align=fig_dict['rad_pro_bottom_align'],
                                width=fig_dict['rad_pro_width'],
                                height=fig_dict['rad_pro_height'],
                                space_vertical=fig_dict['rad_pro_space_vertical'],
                                space_horizontal=fig_dict['rad_pro_space_horizontal'],
                                   row_idx=1, col_idx=hst_col_index+1)

                    radius, profile, error = phot_tools.ProfileTools.get_rad_profile_from_img(
                        img=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].data,
                        wcs=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].wcs,
                        ra=ra, dec=dec, max_rad_arcsec=0.5, img_err=cutout_dict_stamp['%s_err_cutout' % hst_ha_band].data)
                    psf_dict = phot_tools.PSFTools.load_obs_psf_dict(
                        band=hst_ha_band, instrument=helper_func.ObsTools.get_hst_instrument(
                            target=self.phot_hst_target_name, band=hst_ha_band))
                    ax_rad_profile.fill_between(radius, profile-error, profile+error, color='gray', alpha=0.7)
                    ax_rad_profile.plot(radius, profile, linewidth=4, color='k')
                    mask_psf_rad = psf_dict['radius_arcsec'] < np.max(radius)

                    ax_rad_profile.plot(psf_dict['radius_arcsec'][mask_psf_rad],
                                        psf_dict['psf_profile'][mask_psf_rad] / np.max(psf_dict['psf_profile']),
                                        linewidth=4, color='red')

                    ax_rad_profile.set_yticklabels([])
                    ax_rad_profile.tick_params(axis='both', which='both', width=2, direction='in', labelsize=fig_dict['stamp_label_size'])
                    ax_rad_profile.set_title(hst_ha_band.upper(), fontsize=fig_dict['stamp_title_font_size'],
                                             color=fig_dict['hst_ha_color'])
                    if hst_col_index == 0:
                        ax_rad_profile.set_xlabel('rad. [\"]', fontsize=fig_dict['stamp_label_size'])

                ax_stamp = plotting_tools.AxisTools.add_panel_axis(
                            fig=fig,
                            left_align=fig_dict['stamp_left_align'],
                            bottom_align=fig_dict['stamp_bottom_align'],
                            width=fig_dict['stamp_width'],
                            height=fig_dict['stamp_height'],
                            space_vertical=fig_dict['stamp_space_vertical'],
                            space_horizontal=fig_dict['stamp_space_horizontal'],
                               row_idx=1, col_idx=hst_col_index+1, projection=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].wcs)
                norm_hst_stamp = plotting_tools.ColorBarTools.compute_cbar_norm(
                    cutout_list=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].data,
                    log_scale=True)
                ax_stamp.imshow(
                    cutout_dict_stamp['%s_img_cutout' % hst_ha_band].data, norm=norm_hst_stamp,
                    cmap='Greys')
                plotting_tools.WCSPlottingTools.plot_coord_crosshair(ax=ax_stamp,
                                                                pos=SkyCoord(ra=ra * u.deg,
                                                                             dec=dec * u.deg),
                                                                wcs=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].wcs,
                                                                rad=0.3, hair_length=0.3,
                                                                color='red', line_width=2)
                plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_stamp, ra_tick_label=False,
                                     dec_tick_label=False, ra_axis_label=' ', dec_axis_label=' ',
                                     fontsize=fig_dict['stamp_label_size'], labelsize=fig_dict['stamp_label_size'])
                if hst_col_index == 0:
                    plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                        ax=ax_stamp, img_shape=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].data.shape,
                        wcs=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].wcs,
                        bar_length=fig_dict['stamp_scale_bar_length_1'], length_unit='arcsec',
                        bar_color='tab:red', text_color='tab:red',
                        line_width=4, fontsize=fig_dict['stamp_label_size'],
                                   va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)

                # same with continuum subtracted H-alpha
                hst_ha_cont_sub_band = hst_ha_band + '_cont_sub'
                if hst_ha_cont_sub_band in band_list:
                    # check if data is not corrupt
                    if (not (np.all(np.isnan(cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].data)) |
                             np.all(cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].data == 0))):


                        if plot_rad_profile:
                            ax_rad_profile = plotting_tools.AxisTools.add_panel_axis(
                                        fig=fig,
                                        left_align=fig_dict['rad_pro_left_align'],
                                        bottom_align=fig_dict['rad_pro_bottom_align'],
                                        width=fig_dict['rad_pro_width'],
                                        height=fig_dict['rad_pro_height'],
                                        space_vertical=fig_dict['rad_pro_space_vertical'],
                                        space_horizontal=fig_dict['rad_pro_space_horizontal'],
                                           row_idx=1, col_idx=hst_col_index+2)

                            radius, profile, error = phot_tools.ProfileTools.get_rad_profile_from_img(
                                img=cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].data,
                                wcs=cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].wcs,
                                ra=ra, dec=dec, max_rad_arcsec=0.5, img_err=cutout_dict_stamp['%s_err_cutout' % hst_ha_cont_sub_band].data)
                            psf_dict = phot_tools.PSFTools.load_obs_psf_dict(
                                band=hst_ha_band, instrument=helper_func.ObsTools.get_hst_instrument(
                                    target=self.phot_hst_target_name, band=hst_ha_band))
                            ax_rad_profile.fill_between(radius, profile-error, profile+error, color='gray', alpha=0.7)
                            ax_rad_profile.plot(radius, profile, linewidth=4, color='k')
                            mask_psf_rad = psf_dict['radius_arcsec'] < np.max(radius)

                            ax_rad_profile.plot(psf_dict['radius_arcsec'][mask_psf_rad],
                                                psf_dict['psf_profile'][mask_psf_rad] / np.max(psf_dict['psf_profile']),
                                                linewidth=4, color='red')

                            ax_rad_profile.set_yticklabels([])
                            ax_rad_profile.tick_params(axis='both', which='both', width=2, direction='in', labelsize=fig_dict['stamp_label_size'])
                            ax_rad_profile.set_title(hst_ha_cont_sub_band.upper(), fontsize=fig_dict['stamp_title_font_size'],
                                                     color=fig_dict['hst_ha_color'])
                            if hst_col_index == 0:
                                ax_rad_profile.set_xlabel('rad. [\"]', fontsize=fig_dict['stamp_label_size'])

                        ax_stamp = plotting_tools.AxisTools.add_panel_axis(
                                    fig=fig,
                                    left_align=fig_dict['stamp_left_align'],
                                    bottom_align=fig_dict['stamp_bottom_align'],
                                    width=fig_dict['stamp_width'],
                                    height=fig_dict['stamp_height'],
                                    space_vertical=fig_dict['stamp_space_vertical'],
                                    space_horizontal=fig_dict['stamp_space_horizontal'],
                                       row_idx=1, col_idx=hst_col_index+2, projection=cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].wcs)
                        norm_hst_stamp = plotting_tools.ColorBarTools.compute_cbar_norm(
                            cutout_list=cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].data,
                            log_scale=True)
                        ax_stamp.imshow(
                            cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].data, norm=norm_hst_stamp,
                            cmap='Greys')
                        plotting_tools.WCSPlottingTools.plot_coord_crosshair(ax=ax_stamp,
                                                                        pos=SkyCoord(ra=ra * u.deg,
                                                                                     dec=dec * u.deg),
                                                                        wcs=cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].wcs,
                                                                        rad=0.3, hair_length=0.3,
                                                                        color='red', line_width=2)
                        plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_stamp, ra_tick_label=False,
                                             dec_tick_label=False, ra_axis_label=' ', dec_axis_label=' ',
                                             fontsize=fig_dict['stamp_label_size'], labelsize=fig_dict['stamp_label_size'])
                        if hst_col_index == 0:
                            plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                                ax=ax_stamp, img_shape=cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].data.shape,
                                wcs=cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].wcs,
                                bar_length=fig_dict['stamp_scale_bar_length_1'], length_unit='arcsec',
                                bar_color='tab:red', text_color='tab:red',
                                line_width=4, fontsize=fig_dict['stamp_label_size'],
                                           va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)


        nircam_col_index = 0
        for nircam_col_index, nircam_stamp_band in enumerate(nircam_band_list):
            if plot_rad_profile:
                ax_rad_profile = plotting_tools.AxisTools.add_panel_axis(
                            fig=fig,
                            left_align=fig_dict['rad_pro_left_align'],
                            bottom_align=fig_dict['rad_pro_bottom_align'],
                            width=fig_dict['rad_pro_width'],
                            height=fig_dict['rad_pro_height'],
                            space_vertical=fig_dict['rad_pro_space_vertical'],
                            space_horizontal=fig_dict['rad_pro_space_horizontal'],
                               row_idx=0, col_idx=nircam_col_index)

                radius, profile, error = phot_tools.ProfileTools.get_rad_profile_from_img(
                    img=cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].data,
                    wcs=cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].wcs,
                    ra=ra, dec=dec, max_rad_arcsec=0.5, img_err=cutout_dict_stamp['%s_err_cutout' % nircam_stamp_band].data)

                psf_dict = phot_tools.PSFTools.load_obs_psf_dict(band=nircam_stamp_band, instrument='nircam')
                mask_psf_rad = psf_dict['radius_arcsec'] < np.max(radius)

                ax_rad_profile.fill_between(radius, profile-error, profile+error, color='gray', alpha=0.7)
                if nircam_col_index == 0:
                    ax_rad_profile.plot(radius, profile, linewidth=4, color='k', label='measured')

                    ax_rad_profile.plot(psf_dict['radius_arcsec'][mask_psf_rad],
                                        psf_dict['psf_profile'][mask_psf_rad] / np.max(psf_dict['psf_profile']),
                                        linewidth=4, color='red', label='PSF')

                    ax_rad_profile.legend(frameon=False, fontsize=fig_dict['stamp_label_size'])
                else:
                    ax_rad_profile.plot(radius, profile, linewidth=4, color='k')
                    ax_rad_profile.plot(psf_dict['radius_arcsec'][mask_psf_rad], psf_dict['psf_profile'][mask_psf_rad] / np.max(psf_dict['psf_profile']), linewidth=4,
                                        color='red')
                ax_rad_profile.set_yticklabels([])
                ax_rad_profile.tick_params(axis='both', which='both', width=2, direction='in', labelsize=fig_dict['stamp_label_size'])
                ax_rad_profile.set_title(nircam_stamp_band.upper(), fontsize=fig_dict['stamp_title_font_size'],
                                         color=fig_dict['nircam_color'])
                if nircam_col_index == 0:
                    ax_rad_profile.set_xlabel('rad. [\"]', fontsize=fig_dict['stamp_label_size'])

            ax_stamp = plotting_tools.AxisTools.add_panel_axis(
                        fig=fig,
                        left_align=fig_dict['stamp_left_align'],
                        bottom_align=fig_dict['stamp_bottom_align'],
                        width=fig_dict['stamp_width'],
                        height=fig_dict['stamp_height'],
                        space_vertical=fig_dict['stamp_space_vertical'],
                        space_horizontal=fig_dict['stamp_space_horizontal'],
                           row_idx=0, col_idx=nircam_col_index, projection=cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].wcs)
            norm_nircam_stamp = plotting_tools.ColorBarTools.compute_cbar_norm(
                cutout_list=cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].data,
                log_scale=True)
            ax_stamp.imshow(
                cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].data, norm=norm_nircam_stamp,
                cmap='Greys')
            plotting_tools.WCSPlottingTools.plot_coord_crosshair(ax=ax_stamp,
                                                            pos=SkyCoord(ra=ra * u.deg,
                                                                         dec=dec * u.deg),
                                                            wcs=cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].wcs,
                                                            rad=0.3, hair_length=0.3,
                                                            color='red', line_width=2)
            plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_stamp, ra_tick_label=False,
                                 dec_tick_label=False, ra_axis_label=' ', dec_axis_label=' ',
                                 fontsize=fig_dict['stamp_label_size'], labelsize=fig_dict['stamp_label_size'])
            if nircam_col_index == 0:
                plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                    ax=ax_stamp, img_shape=cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].data.shape,
                    wcs=cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].wcs,
                    bar_length=fig_dict['stamp_scale_bar_length_1'], length_unit='arcsec',
                    bar_color='tab:red', text_color='tab:red',
                    line_width=4, fontsize=fig_dict['stamp_label_size'],
                               va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)

        miri_col_index = 0
        for miri_col_index, miri_stamp_band in enumerate(miri_band_list):
            if plot_rad_profile:
                ax_rad_profile = plotting_tools.AxisTools.add_panel_axis(
                            fig=fig,
                            left_align=fig_dict['rad_pro_left_align'],
                            bottom_align=fig_dict['rad_pro_bottom_align'],
                            width=fig_dict['rad_pro_width'],
                            height=fig_dict['rad_pro_height'],
                            space_vertical=fig_dict['rad_pro_space_vertical'],
                            space_horizontal=fig_dict['rad_pro_space_horizontal'],
                               row_idx=0, col_idx=nircam_col_index+miri_col_index+1)

                radius, profile, error = phot_tools.ProfileTools.get_rad_profile_from_img(
                    img=cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].data,
                    wcs=cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].wcs,
                    ra=ra, dec=dec, max_rad_arcsec=1.0, img_err=cutout_dict_stamp['%s_err_cutout' % miri_stamp_band].data)

                psf_dict = phot_tools.PSFTools.load_obs_psf_dict(band=miri_stamp_band, instrument='miri')
                mask_psf_rad = psf_dict['radius_arcsec'] < np.max(radius)

                ax_rad_profile.fill_between(radius, profile-error, profile+error, color='gray', alpha=0.7)
                ax_rad_profile.plot(radius, profile, linewidth=4, color='k')

                ax_rad_profile.plot(psf_dict['radius_arcsec'][mask_psf_rad],
                                psf_dict['psf_profile'][mask_psf_rad] / np.max(psf_dict['psf_profile']),
                                linewidth=4, color='red')

                ax_rad_profile.set_yticklabels([])
                ax_rad_profile.tick_params(axis='both', which='both', width=2, direction='in', labelsize=fig_dict['stamp_label_size'])
                ax_rad_profile.set_title(miri_stamp_band.upper(), fontsize=fig_dict['stamp_title_font_size'],
                                         color=fig_dict['miri_color'])
                if miri_col_index == 0:
                    ax_rad_profile.set_xlabel('rad. [\"]', fontsize=fig_dict['stamp_label_size'])

            ax_stamp = plotting_tools.AxisTools.add_panel_axis(
                        fig=fig,
                        left_align=fig_dict['stamp_left_align'],
                        bottom_align=fig_dict['stamp_bottom_align'],
                        width=fig_dict['stamp_width'],
                        height=fig_dict['stamp_height'],
                        space_vertical=fig_dict['stamp_space_vertical'],
                        space_horizontal=fig_dict['stamp_space_horizontal'],
                           row_idx=0, col_idx=nircam_col_index+miri_col_index+1, projection=cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].wcs)
            norm_miri_stamp = plotting_tools.ColorBarTools.compute_cbar_norm(
                cutout_list=cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].data,
                log_scale=True)
            ax_stamp.imshow(
                cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].data, norm=norm_miri_stamp,
                cmap='Greys')
            plotting_tools.WCSPlottingTools.plot_coord_crosshair(ax=ax_stamp,
                                                            pos=SkyCoord(ra=ra * u.deg,
                                                                         dec=dec * u.deg),
                                                            wcs=cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].wcs,
                                                            rad=0.3, hair_length=0.3,
                                                            color='red', line_width=2)
            plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_stamp, ra_tick_label=False,
                                 dec_tick_label=False, ra_axis_label=' ', dec_axis_label=' ',
                                 fontsize=fig_dict['stamp_label_size'], labelsize=fig_dict['stamp_label_size'])
            if miri_col_index == 0:
                plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                    ax=ax_stamp, img_shape=cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].data.shape,
                    wcs=cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].wcs,
                    bar_length=fig_dict['stamp_scale_bar_length_1'], length_unit='arcsec',
                    bar_color='tab:red', text_color='tab:red',
                    line_width=4, fontsize=fig_dict['stamp_label_size'],
                               va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)


    def plot_img_stamps_all(self, fig, fig_dict, ra, dec, plot_rad_profile=True):


        hst_broad_band_list = self.get_covered_hst_broad_band_list(ra=ra, dec=dec)
        hst_ha_band = self.get_covered_hst_ha_band(ra=ra, dec=dec)
        nircam_band_list = self.get_covered_nircam_band_list(ra=ra, dec=dec)
        miri_band_list = self.get_covered_miri_band_list(ra=ra, dec=dec)

        band_list = self.get_covered_hst_broad_band_list(ra=ra, dec=dec)


        # check if H-alpha is available
        if helper_func.ObsTools.check_hst_ha_cont_sub_obs(target=self.phot_hst_ha_cont_sub_target_name):
            if self.check_coords_covered_by_band(telescope="hst", ra=ra, dec=dec, band=helper_func.ObsTools.get_hst_ha_band(target=self.phot_hst_ha_cont_sub_target_name), max_dist_dist2hull_arcsec=2):
                band_list += [hst_ha_band]
                # check if Ha- continuum subtracted image is available
                if self.phot_hst_ha_cont_sub_target_name in phangs_info.hst_ha_cont_sub_dict.keys():
                    band_list += [hst_ha_band + '_cont_sub']
        band_list += nircam_band_list + miri_band_list

        # load data
        self.load_phangs_bands(band_list=band_list, flux_unit='MJy/sr', load_err=True, load_hst=True, load_hst_ha=True,
                              load_nircam=True, load_miri=True, load_astrosat=False)

        # load cutout stamps
        cutout_dict_stamp = self.get_band_cutout_dict(ra_cutout=ra, dec_cutout=dec, cutout_size=fig_dict['stamp_size'],
                                                      band_list=band_list, include_err=True)

        col_idx = 0
        max_n_cols = 9
        running_row_idx = np.ceil(len(band_list) / max_n_cols)

        for hst_stamp_band in hst_broad_band_list:
            if plot_rad_profile:
                ax_rad_profile = plotting_tools.AxisTools.add_panel_axis(
                            fig=fig,
                            left_align=fig_dict['rad_pro_left_align'],
                            bottom_align=fig_dict['rad_pro_bottom_align'],
                            width=fig_dict['rad_pro_width'],
                            height=fig_dict['rad_pro_height'],
                            space_vertical=fig_dict['rad_pro_space_vertical'],
                            space_horizontal=fig_dict['rad_pro_space_horizontal'],
                               row_idx=running_row_idx, col_idx=col_idx)

                radius, profile, error = phot_tools.ProfileTools.get_rad_profile_from_img(
                    img=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].data,
                    wcs=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].wcs,
                    ra=ra, dec=dec, max_rad_arcsec=0.5, img_err=cutout_dict_stamp['%s_err_cutout' % hst_stamp_band].data)

                psf_dict = phot_tools.PSFTools.load_obs_psf_dict(
                    band=hst_stamp_band, instrument=helper_func.ObsTools.get_hst_instrument(
                        target=self.phot_hst_target_name, band=hst_stamp_band))
                mask_psf_rad = psf_dict['radius_arcsec'] < np.max(radius)
                ax_rad_profile.fill_between(radius, profile-error, profile+error, color='gray', alpha=0.7)
                ax_rad_profile.plot(radius, profile, linewidth=4, color='k')
                ax_rad_profile.plot(psf_dict['radius_arcsec'][mask_psf_rad], psf_dict['psf_profile'][mask_psf_rad] / np.max(psf_dict['psf_profile']), linewidth=4, color='red')

                ax_rad_profile.set_yticklabels([])
                ax_rad_profile.tick_params(axis='both', which='both', width=2, direction='in', labelsize=fig_dict['stamp_label_size'])
                ax_rad_profile.set_title(hst_stamp_band.upper(), fontsize=fig_dict['stamp_title_font_size'],
                                         color=fig_dict['hst_broad_band_color'])
                if col_idx == 0:
                    ax_rad_profile.set_xlabel('rad. [\"]', fontsize=fig_dict['stamp_label_size'])

            ax_stamp = plotting_tools.AxisTools.add_panel_axis(
                        fig=fig,
                        left_align=fig_dict['stamp_left_align'],
                        bottom_align=fig_dict['stamp_bottom_align'],
                        width=fig_dict['stamp_width'],
                        height=fig_dict['stamp_height'],
                        space_vertical=fig_dict['stamp_space_vertical'],
                        space_horizontal=fig_dict['stamp_space_horizontal'],
                           row_idx=running_row_idx, col_idx=col_idx, projection=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].wcs)
            norm_hst_stamp = plotting_tools.ColorBarTools.compute_cbar_norm(
                cutout_list=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].data,
                log_scale=True)
            ax_stamp.imshow(
                cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].data, norm=norm_hst_stamp,
                cmap='Greys')
            plotting_tools.WCSPlottingTools.plot_coord_crosshair(ax=ax_stamp,
                                                            pos=SkyCoord(ra=ra * u.deg,
                                                                         dec=dec * u.deg),
                                                            wcs=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].wcs,
                                                            rad=0.3, hair_length=0.3,
                                                            color='red', line_width=2)
            plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_stamp, ra_tick_label=False,
                                 dec_tick_label=False, ra_axis_label=' ', dec_axis_label=' ',
                                 fontsize=fig_dict['stamp_label_size'], labelsize=fig_dict['stamp_label_size'])
            if col_idx == 0:
                plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                    ax=ax_stamp, img_shape=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].data.shape,
                    wcs=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].wcs,
                    bar_length=fig_dict['stamp_scale_bar_length_1'], length_unit='arcsec',
                    bar_color='tab:red', text_color='tab:red',
                    line_width=4, fontsize=fig_dict['stamp_label_size'],
                               va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)
            if col_idx == 1:
                plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                    ax=ax_stamp, img_shape=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].data.shape,
                    wcs=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].wcs,
                    bar_length=fig_dict['stamp_scale_bar_length_2'], length_unit='pc',
                    phangs_target=self.phot_hst_target_name,
                    bar_color='tab:red', text_color='tab:red',
                    line_width=4, fontsize=fig_dict['stamp_label_size'],
                               va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)
            col_idx +=1
            if col_idx >= max_n_cols:
                col_idx = 0
                running_row_idx -= 1

        # add h alpha
        if helper_func.ObsTools.check_hst_ha_cont_sub_obs(target=self.phot_hst_target_name):
            if self.check_coords_covered_by_band(
                    telescope="hst", ra=ra, dec=dec, band=helper_func.ObsTools.get_hst_ha_band(target=self.phot_hst_target_name),
                    max_dist_dist2hull_arcsec=2):
                hst_ha_band = helper_func.ObsTools.get_hst_ha_band(target=self.phot_hst_target_name)
                if plot_rad_profile:
                    ax_rad_profile = plotting_tools.AxisTools.add_panel_axis(
                                fig=fig,
                                left_align=fig_dict['rad_pro_left_align'],
                                bottom_align=fig_dict['rad_pro_bottom_align'],
                                width=fig_dict['rad_pro_width'],
                                height=fig_dict['rad_pro_height'],
                                space_vertical=fig_dict['rad_pro_space_vertical'],
                                space_horizontal=fig_dict['rad_pro_space_horizontal'],
                                   row_idx=running_row_idx, col_idx=col_idx)

                    radius, profile, error = phot_tools.ProfileTools.get_rad_profile_from_img(
                        img=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].data,
                        wcs=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].wcs,
                        ra=ra, dec=dec, max_rad_arcsec=0.5, img_err=cutout_dict_stamp['%s_err_cutout' % hst_ha_band].data)
                    psf_dict = phot_tools.PSFTools.load_obs_psf_dict(
                        band=hst_ha_band, instrument=helper_func.ObsTools.get_hst_instrument(
                            target=self.phot_hst_target_name, band=hst_ha_band))
                    ax_rad_profile.fill_between(radius, profile-error, profile+error, color='gray', alpha=0.7)
                    ax_rad_profile.plot(radius, profile, linewidth=4, color='k')
                    mask_psf_rad = psf_dict['radius_arcsec'] < np.max(radius)

                    ax_rad_profile.plot(psf_dict['radius_arcsec'][mask_psf_rad],
                                        psf_dict['psf_profile'][mask_psf_rad] / np.max(psf_dict['psf_profile']),
                                        linewidth=4, color='red')

                    ax_rad_profile.set_yticklabels([])
                    ax_rad_profile.tick_params(axis='both', which='both', width=2, direction='in', labelsize=fig_dict['stamp_label_size'])
                    ax_rad_profile.set_title(hst_ha_band.upper(), fontsize=fig_dict['stamp_title_font_size'],
                                             color=fig_dict['hst_ha_color'])
                    if col_idx == 0:
                        ax_rad_profile.set_xlabel('rad. [\"]', fontsize=fig_dict['stamp_label_size'])

                ax_stamp = plotting_tools.AxisTools.add_panel_axis(
                            fig=fig,
                            left_align=fig_dict['stamp_left_align'],
                            bottom_align=fig_dict['stamp_bottom_align'],
                            width=fig_dict['stamp_width'],
                            height=fig_dict['stamp_height'],
                            space_vertical=fig_dict['stamp_space_vertical'],
                            space_horizontal=fig_dict['stamp_space_horizontal'],
                               row_idx=running_row_idx, col_idx=col_idx, projection=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].wcs)
                norm_hst_stamp = plotting_tools.ColorBarTools.compute_cbar_norm(
                    cutout_list=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].data,
                    log_scale=True)
                ax_stamp.imshow(
                    cutout_dict_stamp['%s_img_cutout' % hst_ha_band].data, norm=norm_hst_stamp,
                    cmap='Greys')
                plotting_tools.WCSPlottingTools.plot_coord_crosshair(ax=ax_stamp,
                                                                pos=SkyCoord(ra=ra * u.deg,
                                                                             dec=dec * u.deg),
                                                                wcs=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].wcs,
                                                                rad=0.3, hair_length=0.3,
                                                                color='red', line_width=2)
                plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_stamp, ra_tick_label=False,
                                     dec_tick_label=False, ra_axis_label=' ', dec_axis_label=' ',
                                     fontsize=fig_dict['stamp_label_size'], labelsize=fig_dict['stamp_label_size'])
                if col_idx == 0:
                    plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                        ax=ax_stamp, img_shape=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].data.shape,
                        wcs=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].wcs,
                        bar_length=fig_dict['stamp_scale_bar_length_1'], length_unit='arcsec',
                        bar_color='tab:red', text_color='tab:red',
                        line_width=4, fontsize=fig_dict['stamp_label_size'],
                                   va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)
                col_idx += 1
                if col_idx >= max_n_cols:
                    col_idx = 0
                    running_row_idx -= 1



                # same with continuum subtracted H-alpha
                hst_ha_cont_sub_band = hst_ha_band + '_cont_sub'
                if hst_ha_cont_sub_band in band_list:
                    # check if data is not corrupt
                    if (not (np.all(np.isnan(cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].data)) |
                             np.all(cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].data == 0))):


                        if plot_rad_profile:
                            ax_rad_profile = plotting_tools.AxisTools.add_panel_axis(
                                        fig=fig,
                                        left_align=fig_dict['rad_pro_left_align'],
                                        bottom_align=fig_dict['rad_pro_bottom_align'],
                                        width=fig_dict['rad_pro_width'],
                                        height=fig_dict['rad_pro_height'],
                                        space_vertical=fig_dict['rad_pro_space_vertical'],
                                        space_horizontal=fig_dict['rad_pro_space_horizontal'],
                                           row_idx=running_row_idx, col_idx=col_idx)

                            radius, profile, error = phot_tools.ProfileTools.get_rad_profile_from_img(
                                img=cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].data,
                                wcs=cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].wcs,
                                ra=ra, dec=dec, max_rad_arcsec=0.5, img_err=cutout_dict_stamp['%s_err_cutout' % hst_ha_cont_sub_band].data)
                            psf_dict = phot_tools.PSFTools.load_obs_psf_dict(
                                band=hst_ha_band, instrument=helper_func.ObsTools.get_hst_instrument(
                                    target=self.phot_hst_target_name, band=hst_ha_band))
                            ax_rad_profile.fill_between(radius, profile-error, profile+error, color='gray', alpha=0.7)
                            ax_rad_profile.plot(radius, profile, linewidth=4, color='k')
                            mask_psf_rad = psf_dict['radius_arcsec'] < np.max(radius)

                            ax_rad_profile.plot(psf_dict['radius_arcsec'][mask_psf_rad],
                                                psf_dict['psf_profile'][mask_psf_rad] / np.max(psf_dict['psf_profile']),
                                                linewidth=4, color='red')

                            ax_rad_profile.set_yticklabels([])
                            ax_rad_profile.tick_params(axis='both', which='both', width=2, direction='in', labelsize=fig_dict['stamp_label_size'])
                            ax_rad_profile.set_title(hst_ha_cont_sub_band.upper(), fontsize=fig_dict['stamp_title_font_size'],
                                                     color=fig_dict['hst_ha_color'])
                            if col_idx == 0:
                                ax_rad_profile.set_xlabel('rad. [\"]', fontsize=fig_dict['stamp_label_size'])

                        ax_stamp = plotting_tools.AxisTools.add_panel_axis(
                                    fig=fig,
                                    left_align=fig_dict['stamp_left_align'],
                                    bottom_align=fig_dict['stamp_bottom_align'],
                                    width=fig_dict['stamp_width'],
                                    height=fig_dict['stamp_height'],
                                    space_vertical=fig_dict['stamp_space_vertical'],
                                    space_horizontal=fig_dict['stamp_space_horizontal'],
                                       row_idx=running_row_idx, col_idx=col_idx, projection=cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].wcs)
                        norm_hst_stamp = plotting_tools.ColorBarTools.compute_cbar_norm(
                            cutout_list=cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].data,
                            log_scale=True)
                        ax_stamp.imshow(
                            cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].data, norm=norm_hst_stamp,
                            cmap='Greys')
                        plotting_tools.WCSPlottingTools.plot_coord_crosshair(ax=ax_stamp,
                                                                        pos=SkyCoord(ra=ra * u.deg,
                                                                                     dec=dec * u.deg),
                                                                        wcs=cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].wcs,
                                                                        rad=0.3, hair_length=0.3,
                                                                        color='red', line_width=2)
                        plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_stamp, ra_tick_label=False,
                                             dec_tick_label=False, ra_axis_label=' ', dec_axis_label=' ',
                                             fontsize=fig_dict['stamp_label_size'], labelsize=fig_dict['stamp_label_size'])
                        if col_idx == 0:
                            plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                                ax=ax_stamp, img_shape=cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].data.shape,
                                wcs=cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].wcs,
                                bar_length=fig_dict['stamp_scale_bar_length_1'], length_unit='arcsec',
                                bar_color='tab:red', text_color='tab:red',
                                line_width=4, fontsize=fig_dict['stamp_label_size'],
                                           va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)

                        col_idx += 1
                        if col_idx >= max_n_cols:
                            col_idx = 0
                            running_row_idx -= 1

        for nircam_stamp_band in nircam_band_list:
            if plot_rad_profile:
                ax_rad_profile = plotting_tools.AxisTools.add_panel_axis(
                            fig=fig,
                            left_align=fig_dict['rad_pro_left_align'],
                            bottom_align=fig_dict['rad_pro_bottom_align'],
                            width=fig_dict['rad_pro_width'],
                            height=fig_dict['rad_pro_height'],
                            space_vertical=fig_dict['rad_pro_space_vertical'],
                            space_horizontal=fig_dict['rad_pro_space_horizontal'],
                               row_idx=running_row_idx, col_idx=col_idx)

                radius, profile, error = phot_tools.ProfileTools.get_rad_profile_from_img(
                    img=cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].data,
                    wcs=cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].wcs,
                    ra=ra, dec=dec, max_rad_arcsec=0.5, img_err=cutout_dict_stamp['%s_err_cutout' % nircam_stamp_band].data)

                psf_dict = phot_tools.PSFTools.load_obs_psf_dict(band=nircam_stamp_band, instrument='nircam')
                mask_psf_rad = psf_dict['radius_arcsec'] < np.max(radius)

                ax_rad_profile.fill_between(radius, profile-error, profile+error, color='gray', alpha=0.7)
                if col_idx == 0:
                    ax_rad_profile.plot(radius, profile, linewidth=4, color='k', label='measured')

                    ax_rad_profile.plot(psf_dict['radius_arcsec'][mask_psf_rad],
                                        psf_dict['psf_profile'][mask_psf_rad] / np.max(psf_dict['psf_profile']),
                                        linewidth=4, color='red', label='PSF')

                    ax_rad_profile.legend(frameon=False, fontsize=fig_dict['stamp_label_size'])
                else:
                    ax_rad_profile.plot(radius, profile, linewidth=4, color='k')
                    ax_rad_profile.plot(psf_dict['radius_arcsec'][mask_psf_rad], psf_dict['psf_profile'][mask_psf_rad] / np.max(psf_dict['psf_profile']), linewidth=4,
                                        color='red')
                ax_rad_profile.set_yticklabels([])
                ax_rad_profile.tick_params(axis='both', which='both', width=2, direction='in', labelsize=fig_dict['stamp_label_size'])
                ax_rad_profile.set_title(nircam_stamp_band.upper(), fontsize=fig_dict['stamp_title_font_size'],
                                         color=fig_dict['nircam_color'])
                if col_idx == 0:
                    ax_rad_profile.set_xlabel('rad. [\"]', fontsize=fig_dict['stamp_label_size'])

            ax_stamp = plotting_tools.AxisTools.add_panel_axis(
                        fig=fig,
                        left_align=fig_dict['stamp_left_align'],
                        bottom_align=fig_dict['stamp_bottom_align'],
                        width=fig_dict['stamp_width'],
                        height=fig_dict['stamp_height'],
                        space_vertical=fig_dict['stamp_space_vertical'],
                        space_horizontal=fig_dict['stamp_space_horizontal'],
                           row_idx=running_row_idx, col_idx=col_idx, projection=cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].wcs)
            norm_nircam_stamp = plotting_tools.ColorBarTools.compute_cbar_norm(
                cutout_list=cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].data,
                log_scale=True)
            ax_stamp.imshow(
                cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].data, norm=norm_nircam_stamp,
                cmap='Greys')
            plotting_tools.WCSPlottingTools.plot_coord_crosshair(ax=ax_stamp,
                                                            pos=SkyCoord(ra=ra * u.deg,
                                                                         dec=dec * u.deg),
                                                            wcs=cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].wcs,
                                                            rad=0.3, hair_length=0.3,
                                                            color='red', line_width=2)
            plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_stamp, ra_tick_label=False,
                                 dec_tick_label=False, ra_axis_label=' ', dec_axis_label=' ',
                                 fontsize=fig_dict['stamp_label_size'], labelsize=fig_dict['stamp_label_size'])
            if col_idx == 0:
                plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                    ax=ax_stamp, img_shape=cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].data.shape,
                    wcs=cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].wcs,
                    bar_length=fig_dict['stamp_scale_bar_length_1'], length_unit='arcsec',
                    bar_color='tab:red', text_color='tab:red',
                    line_width=4, fontsize=fig_dict['stamp_label_size'],
                               va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)

            col_idx += 1
            if col_idx >= max_n_cols:
                col_idx = 0
                running_row_idx -= 1

        for miri_stamp_band in miri_band_list:
            if plot_rad_profile:
                ax_rad_profile = plotting_tools.AxisTools.add_panel_axis(
                            fig=fig,
                            left_align=fig_dict['rad_pro_left_align'],
                            bottom_align=fig_dict['rad_pro_bottom_align'],
                            width=fig_dict['rad_pro_width'],
                            height=fig_dict['rad_pro_height'],
                            space_vertical=fig_dict['rad_pro_space_vertical'],
                            space_horizontal=fig_dict['rad_pro_space_horizontal'],
                               row_idx=running_row_idx, col_idx=col_idx)

                radius, profile, error = phot_tools.ProfileTools.get_rad_profile_from_img(
                    img=cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].data,
                    wcs=cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].wcs,
                    ra=ra, dec=dec, max_rad_arcsec=1.0, img_err=cutout_dict_stamp['%s_err_cutout' % miri_stamp_band].data)

                psf_dict = phot_tools.PSFTools.load_obs_psf_dict(band=miri_stamp_band, instrument='miri')
                mask_psf_rad = psf_dict['radius_arcsec'] < np.max(radius)

                ax_rad_profile.fill_between(radius, profile-error, profile+error, color='gray', alpha=0.7)
                ax_rad_profile.plot(radius, profile, linewidth=4, color='k')

                ax_rad_profile.plot(psf_dict['radius_arcsec'][mask_psf_rad],
                                psf_dict['psf_profile'][mask_psf_rad] / np.max(psf_dict['psf_profile']),
                                linewidth=4, color='red')

                ax_rad_profile.set_yticklabels([])
                ax_rad_profile.tick_params(axis='both', which='both', width=2, direction='in', labelsize=fig_dict['stamp_label_size'])
                ax_rad_profile.set_title(miri_stamp_band.upper(), fontsize=fig_dict['stamp_title_font_size'],
                                         color=fig_dict['miri_color'])
                if col_idx == 0:
                    ax_rad_profile.set_xlabel('rad. [\"]', fontsize=fig_dict['stamp_label_size'])

            ax_stamp = plotting_tools.AxisTools.add_panel_axis(
                        fig=fig,
                        left_align=fig_dict['stamp_left_align'],
                        bottom_align=fig_dict['stamp_bottom_align'],
                        width=fig_dict['stamp_width'],
                        height=fig_dict['stamp_height'],
                        space_vertical=fig_dict['stamp_space_vertical'],
                        space_horizontal=fig_dict['stamp_space_horizontal'],
                           row_idx=running_row_idx, col_idx=col_idx, projection=cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].wcs)
            norm_miri_stamp = plotting_tools.ColorBarTools.compute_cbar_norm(
                cutout_list=cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].data,
                log_scale=True)
            ax_stamp.imshow(
                cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].data, norm=norm_miri_stamp,
                cmap='Greys')
            plotting_tools.WCSPlottingTools.plot_coord_crosshair(ax=ax_stamp,
                                                            pos=SkyCoord(ra=ra * u.deg,
                                                                         dec=dec * u.deg),
                                                            wcs=cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].wcs,
                                                            rad=0.3, hair_length=0.3,
                                                            color='red', line_width=2)
            plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_stamp, ra_tick_label=False,
                                 dec_tick_label=False, ra_axis_label=' ', dec_axis_label=' ',
                                 fontsize=fig_dict['stamp_label_size'], labelsize=fig_dict['stamp_label_size'])
            if col_idx == 0:
                plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                    ax=ax_stamp, img_shape=cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].data.shape,
                    wcs=cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].wcs,
                    bar_length=fig_dict['stamp_scale_bar_length_1'], length_unit='arcsec',
                    bar_color='tab:red', text_color='tab:red',
                    line_width=4, fontsize=fig_dict['stamp_label_size'],
                               va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)

            col_idx += 1
            if col_idx >= max_n_cols:
                col_idx = 0
                running_row_idx -= 1


    @staticmethod
    def create_phot_morph_axis(fig, fig_dict, col_index, row_index, stamp_wcs, bkg_wcs):
        # plot stamp and bkg
        # define axis
        ax_stamp = fig.add_axes(
            (fig_dict['stamp_left_align'] + (fig_dict['stamp_width'] + fig_dict['stamp_space_vertical']) * col_index,
             fig_dict['stamp_bottom_align'] + (fig_dict['stamp_height'] + fig_dict['stamp_space_horizontal']) * row_index,
             fig_dict['stamp_width'], fig_dict['stamp_height']),
            projection=stamp_wcs)
        ax_bkg = fig.add_axes(
            (fig_dict['stamp_left_align'] + (fig_dict['stamp_width'] + fig_dict['stamp_space_vertical']) *
             (col_index + 1), fig_dict['stamp_bottom_align'] +
             (fig_dict['stamp_height'] + fig_dict['stamp_space_horizontal']) * row_index,
             fig_dict['stamp_width'], fig_dict['stamp_height']),
            projection=bkg_wcs)

        ax_slit_profile = fig.add_axes(
            (fig_dict['rad_pro_left_align'] + (fig_dict['rad_pro_width'] +
                                               fig_dict['rad_pro_space_vertical']) * col_index,
             fig_dict['rad_pro_bottom_align'] + (fig_dict['rad_pro_height'] +
                                                 fig_dict['rad_pro_space_horizontal']) * row_index,
             fig_dict['rad_pro_width'], fig_dict['rad_pro_height']))

        ax_rad_profile = fig.add_axes(
            (fig_dict['rad_pro_left_align'] + (fig_dict['rad_pro_width'] +
                                               fig_dict['rad_pro_space_vertical']) * (col_index + 1),
             fig_dict['rad_pro_bottom_align'] + (fig_dict['rad_pro_height'] +
                                                 fig_dict['rad_pro_space_horizontal']) * row_index,
             fig_dict['rad_pro_width'], fig_dict['rad_pro_height']))

        ax_cbar_stamp = fig.add_axes(
            (fig_dict['cbar_left_align'] + (fig_dict['rad_pro_width'] +
                                            fig_dict['rad_pro_space_vertical']) * col_index,
             fig_dict['cbar_bottom_align'] + (fig_dict['rad_pro_height'] +
                                              fig_dict['rad_pro_space_horizontal']) * row_index,
             fig_dict['cbar_width'], fig_dict['cbar_height']))
        ax_cbar_bkg = fig.add_axes(
            (fig_dict['cbar_left_align'] + (fig_dict['rad_pro_width'] +
                                            fig_dict['rad_pro_space_vertical']) * (col_index + 1),
             fig_dict['cbar_bottom_align'] + (fig_dict['rad_pro_height'] +
                                              fig_dict['rad_pro_space_horizontal']) * row_index,
             fig_dict['cbar_width'], fig_dict['cbar_height']))

        return ax_stamp, ax_bkg, ax_slit_profile, ax_rad_profile, ax_cbar_stamp, ax_cbar_bkg

    def get_scaled_bkg(self, ra, dec, band, scale_size_arcsec, cutout_size, bkg_img_size_factor=40, box_size_factor=2,
                       filter_size_factor=1, do_sigma_clip=True, sigma=3.0, maxiters=10,
                       bkg_method='SExtractorBackground'):

        # get the cutout size to compute the bkg
        bkg_cutout_size = bkg_img_size_factor * scale_size_arcsec
        cutout_dict_bkg = self.get_band_cutout_dict(ra_cutout=ra, dec_cutout=dec,
                                                    cutout_size=(bkg_cutout_size, bkg_cutout_size),
                                                    band_list=[band], include_err=True)

        # estimate the bkg_box_size
        box_size = helper_func.CoordTools.transform_world2pix_scale(
            length_in_arcsec=scale_size_arcsec * box_size_factor, wcs=cutout_dict_bkg['%s_img_cutout' % band].wcs,
            dim=0)
        box_size = int(np.round(box_size))
        # estimate filter sie
        filter_size = helper_func.CoordTools.transform_world2pix_scale(
            length_in_arcsec=scale_size_arcsec * filter_size_factor, wcs=cutout_dict_bkg['%s_img_cutout' % band].wcs,
            dim=0)
        filter_size = int(np.round(filter_size))
        # filter size musst be an odd number
        if filter_size % 2 == 0:
            filter_size += 1

        bkg = phot_tools.PhotTools.compute_2d_bkg(data=cutout_dict_bkg['%s_img_cutout' % band].data,
                                                  box_size=(box_size, box_size),
                                                  filter_size=(filter_size, filter_size), do_sigma_clip=do_sigma_clip,
                                                  sigma=sigma, maxiters=maxiters, bkg_method=bkg_method)

        cutout_stamp_bkg = helper_func.CoordTools.get_img_cutout(
            img=bkg.background,
            wcs=cutout_dict_bkg['%s_img_cutout' % band].wcs,
            coord=SkyCoord(ra=ra * u.deg, dec=dec * u.deg), cutout_size=cutout_size)

        cutout_stamp_bkg_rms = helper_func.CoordTools.get_img_cutout(
            img=bkg.background_rms,
            wcs=cutout_dict_bkg['%s_img_cutout' % band].wcs,
            coord=SkyCoord(ra=ra * u.deg, dec=dec * u.deg), cutout_size=cutout_size)

        return cutout_stamp_bkg, cutout_stamp_bkg_rms

    @staticmethod
    def get_rad_profile_dict(img, wcs, ra, dec, n_slits, max_rad_arcsec, img_err=None):
        # load cutout stamps

        # what is the needed background estimation for one source?
        # get first the radial profile:
        rad, profile, profile_err = phot_tools.ProfileTools.get_rad_profile_from_img(
            img=img,
            wcs=wcs,
            ra=ra, dec=dec,
            max_rad_arcsec=max_rad_arcsec,
            img_err=img_err,
            norm_profile=True)

        # get profiles along a slit
        slit_profile_dict = phot_tools.ProfileTools.compute_axis_profiles_from_img(
            img=img, wcs=wcs, ra=ra, dec=dec, max_rad_arcsec=max_rad_arcsec, n_slits=n_slits, err=img_err)

        return {'rad': rad, 'profile': profile, 'profile_err': profile_err, 'slit_profile_dict': slit_profile_dict}

    @staticmethod
    def plot_stamp_bkg_cbar(ax_stamp, ax_bkg, ax_cbar_stamp, ax_cbar_bkg, cutout_stamp_data, cutout_stamp_bkg,
                            ra, dec, fig_dict, instrument, band, target_name):

        vmin_norm_bkg = np.nanmin(cutout_stamp_bkg.data)
        vmax_norm_bkg = np.nanmax(cutout_stamp_bkg.data)
        bkg_norm = ImageNormalize(stretch=SqrtStretch(), vmin=vmin_norm_bkg, vmax=vmax_norm_bkg, )

        vmin_norm_data = np.nanmin(cutout_stamp_data.data)
        vmax_norm_data = np.nanmax(cutout_stamp_data.data)
        data_norm = ImageNormalize(stretch=SqrtStretch(), vmin=vmin_norm_data, vmax=vmax_norm_data, )

        ax_stamp.imshow(cutout_stamp_data.data, norm=data_norm, cmap='Grays')
        ax_bkg.imshow(cutout_stamp_bkg.data, norm=bkg_norm, cmap='Grays')

        plotting_tools.WCSPlottingTools.plot_coord_crosshair(
            ax=ax_stamp, pos=SkyCoord(ra=ra * u.deg, dec=dec * u.deg),
            wcs=cutout_stamp_data.wcs, rad=0.3, hair_length=0.3, color='red', line_width=2)
        plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_stamp, ra_tick_label=False,
                                                        dec_tick_label=False, ra_axis_label=' ', dec_axis_label=' ',
                                                        fontsize=fig_dict['stamp_label_size'],
                                                        labelsize=fig_dict['stamp_label_size'])
        plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_bkg, ra_tick_label=False,
                                                        dec_tick_label=False, ra_axis_label=' ', dec_axis_label=' ',
                                                        fontsize=fig_dict['stamp_label_size'],
                                                        labelsize=fig_dict['stamp_label_size'])
        plotting_tools.WCSPlottingTools.plot_img_scale_bar(
            ax=ax_stamp, img_shape=cutout_stamp_data.data.shape,
            wcs=cutout_stamp_data.wcs,
            bar_length=fig_dict['stamp_scale_bar_length_%s' % instrument], length_unit='arcsec',
            bar_color='tab:red', text_color='tab:red',
            line_width=4, fontsize=fig_dict['stamp_label_size'],
            va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)

        plotting_tools.WCSPlottingTools.plot_img_scale_bar(
            ax=ax_bkg, img_shape=cutout_stamp_data.data.shape,
            wcs=cutout_stamp_data.wcs,
            bar_length=fig_dict['stamp_scale_bar_length_pc_%s' % instrument], length_unit='pc',
            phangs_target=target_name,
            bar_color='tab:red', text_color='tab:red',
            line_width=4, fontsize=fig_dict['stamp_label_size'],
            va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)

        plotting_tools.StrTools.display_text_in_corner(ax=ax_stamp, text=instrument.upper() + ' ' + band.upper(),
                                                       fontsize=fig_dict['stamp_label_size'], text_color='tab:red',
                                                       x_frac=0.02, y_frac=0.98, horizontal_alignment='left',
                               vertical_alignment='top', path_eff=True, path_err_linewidth=3, path_eff_color='white', rotation=0)

        plotting_tools.ColorBarTools.create_cbar(ax_cbar=ax_cbar_stamp, cmap='Grays', norm=data_norm,
                                                 cbar_label='mJy', fontsize=fig_dict['stamp_label_size'],
                                                 ticks=None,
                                                 labelpad=2, tick_width=2, orientation='horizontal',
                                                 top_lable=False, extend='neither')
        plotting_tools.ColorBarTools.create_cbar(ax_cbar=ax_cbar_bkg, cmap='Grays', norm=bkg_norm,
                                                 cbar_label='mJy', fontsize=fig_dict['stamp_label_size'],
                                                 ticks=None,
                                                 labelpad=2, tick_width=2, orientation='horizontal',
                                                 top_lable=False, extend='neither')
    @staticmethod
    def plot_radial_profiles(ax_slit_profile, ax_slit_profile_bkg_sub, rad_profile_dict, rad_profile_bkg_sub_dict, psf_dict,
                             phot_dict,
                             median_bkg, median_bkg_rms):
        max_peak_value = 0
        max_rad = 0

        for idx in rad_profile_dict['slit_profile_dict']['list_angle_idx']:
            ax_slit_profile.plot(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'],
                                 rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'],
                                 color='gray')
            mask_center = ((rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'] > psf_dict[
                'gaussian_std'] * 3 * -1) &
                           (rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'] < psf_dict[
                               'gaussian_std'] * 3))
            max_value_in_center = np.max(rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'][mask_center])
            if max_value_in_center > max_peak_value:
                max_peak_value = max_value_in_center
            if np.max(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data']) > max_rad:
                max_rad = np.max(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'])

        ax_slit_profile.fill_between(
            [max_rad * -1, max_rad],
            [median_bkg - median_bkg_rms, median_bkg - median_bkg_rms],
            [median_bkg + median_bkg_rms, median_bkg + median_bkg_rms],
            color='indianred')
        ax_slit_profile.plot([max_rad * -1, max_rad],
                             [median_bkg, median_bkg], color='red')

        # plot gaussian psf approximation
        dummy_radius = np.linspace(max_rad * -1, max_rad,
                                   1000)
        gaussian_psf =  max_peak_value * np.exp(-(dummy_radius - psf_dict['gaussian_mean']) ** 2 /
                                                (2 * psf_dict['gaussian_std'] ** 2))
        ax_slit_profile.plot(dummy_radius, gaussian_psf, color='green')




        for idx in rad_profile_bkg_sub_dict['slit_profile_dict']['list_angle_idx']:
            ax_slit_profile_bkg_sub.plot(rad_profile_bkg_sub_dict['slit_profile_dict'][str(idx)]['radius_data'],
                                 rad_profile_bkg_sub_dict['slit_profile_dict'][str(idx)]['profile_data'],
                                 color='gray')
            mask_center = ((rad_profile_bkg_sub_dict['slit_profile_dict'][str(idx)]['radius_data'] > psf_dict[
                'gaussian_std'] * 3 * -1) &
                           (rad_profile_bkg_sub_dict['slit_profile_dict'][str(idx)]['radius_data'] < psf_dict[
                               'gaussian_std'] * 3))

        ax_slit_profile_bkg_sub.plot(phot_dict['dummy_rad'], phot_dict['gauss'], color='red')

    @staticmethod
    def measure_morph_photometry(rad_profile_dict, psf_dict, img, bkg, img_err, wcs, ra, dec):


        # get average value in the PSF aperture:
        # print(psf_dict['gaussian_fwhm'])
        # print(psf_dict['gaussian_std'])

        radius_of_interes = psf_dict['gaussian_std'] * 3

        central_apert_stats_source = phot_tools.PhotTools.get_circ_apert_stats(data=img - bkg, data_err=img_err, wcs=wcs,
                                                                        ra=ra, dec=dec, aperture_rad=radius_of_interes)
        central_apert_stats_bkg = phot_tools.PhotTools.get_circ_apert_stats(data=bkg, data_err=img_err, wcs=wcs,
                                                                        ra=ra, dec=dec, aperture_rad=radius_of_interes)



        amp_list = []
        mu_list = []
        sig_list = []
        amp_err_list = []
        mu_err_list = []
        sig_err_list = []

        # fig, ax = plt.subplots(nrows=len(rad_profile_dict['slit_profile_dict']['list_angle_idx']) +1)

        for idx in rad_profile_dict['slit_profile_dict']['list_angle_idx']:
            mask_center = ((rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'] > psf_dict[
                'gaussian_std'] * 3 * -1) &
                           (rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'] < psf_dict[
                               'gaussian_std'] * 3))
            min_value_in_center = np.min(rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'][mask_center])
            max_value_in_center = np.max(rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'][mask_center])

            # ax[idx].plot(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'],
            #              rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'],
            #              color='gray')
            # ax[-1].plot(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'],
            #              rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'],
            #              color='gray')
            #
            # ax[idx].errorbar(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'],
            #          rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'],
            #              yerr=rad_profile_dict['slit_profile_dict'][str(idx)]['profile_err'],
            #          fmt='.')
            # plt.plot(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'],
            #              rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'],
            #              color='gray')
            # plt.show()

            mask_central_pixels = ((rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'] >
                                    psf_dict['gaussian_std'] * -3) &
                                   (rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'] <
                                    psf_dict['gaussian_std'] * 3))
            # ax[idx].scatter(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'][mask_central_pixels],
            #              rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'][mask_central_pixels],
            #              color='red')

            # select the lower amplitude. There is a chance that the values are negative
            lower_amp = min_value_in_center
            upper_amp = max_value_in_center + np.abs(max_value_in_center * 2)

            # plt.plot(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'],
            #              rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'],
            #              color='gray')

            gaussian_fit_dict = helper_func.FitTools.fit_gauss(
                x_data=rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'][mask_central_pixels],
                y_data=rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'][mask_central_pixels],
                y_data_err=rad_profile_dict['slit_profile_dict'][str(idx)]['profile_err'][mask_central_pixels],
                amp_guess=max_value_in_center, mu_guess=0, sig_guess=psf_dict['gaussian_std'],
                lower_amp=lower_amp, upper_amp=upper_amp,
                lower_mu=psf_dict['gaussian_std'] * -5, upper_mu=psf_dict['gaussian_std'] * 5,
                lower_sigma=psf_dict['gaussian_std'], upper_sigma=psf_dict['gaussian_std'] * 5)


            # dummy_rad = np.linspace(np.min(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data']),
            #                         np.max(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data']), 500)
            # gauss = helper_func.FitTools.gaussian_func(
            #     amp=gaussian_fit_dict['amp'], mu=gaussian_fit_dict['mu'], sig=gaussian_fit_dict['sig'], x_data=dummy_rad)
            #
            # # ax[idx].plot(dummy_rad, gauss)
            # plt.plot(dummy_rad, gauss)
            # plt.show()


            amp_list.append(gaussian_fit_dict['amp'])
            mu_list.append(gaussian_fit_dict['mu'])
            sig_list.append(gaussian_fit_dict['sig'])

            amp_err_list.append(gaussian_fit_dict['amp_err'])
            mu_err_list.append(gaussian_fit_dict['mu_err'])
            sig_err_list.append(gaussian_fit_dict['sig_err'])


        # get the best matching gauss

        amp_list = np.array(amp_list)
        mu_list = np.array(mu_list)
        sig_list = np.array(sig_list)
        amp_err_list = np.array(amp_err_list)
        mu_err_list = np.array(mu_err_list)
        sig_err_list = np.array(sig_err_list)

        # get all the gaussian functions that are central
        mask_mu = np.abs(mu_list) < psf_dict['gaussian_std'] * 1
        mask_amp = (amp_list > 0)

        # if no function was detected in the center
        if sum(mask_mu * mask_amp) == 0:
            # non detection
            mean_amp = central_apert_stats_source.max
            mean_mu = 0
            mean_sig = psf_dict['gaussian_std']
            # get flux inside the 3 sigma aperture
            flux = central_apert_stats_source.sum
            flux_err = np.sqrt(central_apert_stats_source.sum_err **2 + central_apert_stats_bkg.sum_err ** 2)
            detect_flag = False
        else:
            # print(sum(mask_mu * mask_amp))
            # print(amp_list[mask_mu * mask_amp])
            # print(mu_list[mask_mu * mask_amp])
            # print(sig_list[mask_mu * mask_amp])

            mean_amp = np.mean(amp_list[mask_mu * mask_amp])
            mean_mu = np.mean(mu_list[mask_mu * mask_amp])
            mean_sig = np.mean(sig_list[mask_mu * mask_amp])

            mean_amp_err = np.mean(amp_err_list[mask_mu*mask_amp])
            mean_sig_err = np.mean(sig_err_list[mask_mu*mask_amp])

            # we need to do the sigma in pixel scale though
            mean_sig_pix = helper_func.CoordTools.transform_world2pix_scale(length_in_arcsec=mean_sig, wcs=wcs)
            # get gaussian integaral as flux
            flux = mean_amp * 2 * np.pi * (mean_sig_pix ** 2)
            flux_err = np.sqrt(mean_amp_err ** 2 * (2*mean_sig_err) **2)
            flux_err = np.sqrt(flux_err **2 + central_apert_stats_bkg.sum_err ** 2)
            detect_flag = True

        dummy_rad = np.linspace(np.min(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data']),
                                np.max(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data']), 500)
        gauss = helper_func.FitTools.gaussian_func(
                amp=mean_amp, mu=mean_mu, sig=mean_sig, x_data=dummy_rad)


        photometry_dict = {
            'dummy_rad': dummy_rad,
            'gauss': gauss,
            'flux': flux,
            'flux_err': flux_err,
            'detect_flag': detect_flag
        }
        return photometry_dict


        # ax[-1].plot(dummy_rad, gauss)
        #
        #
        # plt.show()
        # # exit()

    def plot_phot_morph(self, fig, fig_dict, ra, dec, return_flux_dict=False):

        hst_band_list = self.get_covered_hst_band_list(ra=ra, dec=dec)

        nircam_band_list = self.get_covered_nircam_band_list(ra=ra, dec=dec)
        miri_band_list = self.get_covered_miri_band_list(ra=ra, dec=dec)

        band_list = hst_band_list + nircam_band_list + miri_band_list

        # load data
        self.load_phangs_bands(band_list=band_list, flux_unit='mJy', load_err=True, load_hst=True, load_hst_ha=True,
                              load_nircam=True, load_miri=True, load_astrosat=False)

        ax_sed = fig.add_axes([
            fig_dict['sed_left_align'],
            fig_dict['sed_bottom_align'],
            fig_dict['sed_width'],
            fig_dict['sed_height'],
        ])

        # gathering data in a dictionary
        flux_dict = {}

        # plot HST bands
        col_index = 4
        row_index = 4
        if hst_band_list:

            cutout_dict_source = self.get_band_cutout_dict(
                ra_cutout=ra, dec_cutout=dec, cutout_size=fig_dict['stamp_size_hst'], band_list=hst_band_list,
                include_err=True)
            for band in hst_band_list:
                # band = 'F438W'

                # get psf dict
                psf_dict = phot_tools.PSFTools.load_hst_psf_dict(
                    band=band, instrument=helper_func.ObsTools.get_hst_instrument(target=self.phot_hst_target_name,
                                                                                  band=band))
                # get the background
                cutout_stamp_bkg, cutout_stamp_bkg_rms = self.get_scaled_bkg(
                    ra=ra, dec=dec, band=band, scale_size_arcsec=psf_dict['gaussian_fwhm'],
                    cutout_size=fig_dict['stamp_size_hst'], bkg_img_size_factor=fig_dict['bkg_img_size_factor_hst'],
                    box_size_factor=fig_dict['box_size_factor_hst'], filter_size_factor=fig_dict['filter_size_factor_hst'],
                    do_sigma_clip=True, sigma=3.0, maxiters=10, bkg_method='SExtractorBackground')

                rad_profile_dict = self.get_rad_profile_dict(img=cutout_dict_source['%s_img_cutout' % band].data,
                                                             wcs=cutout_dict_source['%s_img_cutout' % band].wcs,
                                                             ra=ra, dec=dec, n_slits=fig_dict['n_profile_slits'],
                                                             max_rad_arcsec=fig_dict['max_rad_profile_hst_arcsec'],
                                                             img_err=cutout_dict_source['%s_err_cutout' % band].data)
                rad_profile_bkg_sub_dict = self.get_rad_profile_dict(
                    img=cutout_dict_source['%s_img_cutout' % band].data - cutout_stamp_bkg.data,
                    wcs=cutout_dict_source['%s_img_cutout' % band].wcs, ra=ra, dec=dec,
                    n_slits=fig_dict['n_profile_slits'], max_rad_arcsec=fig_dict['max_rad_profile_hst_arcsec'],
                    img_err=cutout_dict_source['%s_err_cutout' % band].data)

                # print(band)
                # plt.close(fig)
                # plt.imshow(cutout_dict_source['%s_img_cutout' % band].data)
                # plt.show()
                phot_dict = self.measure_morph_photometry(
                    rad_profile_dict=rad_profile_bkg_sub_dict, psf_dict=psf_dict,
                    img=cutout_dict_source['%s_img_cutout' % band].data, bkg=cutout_stamp_bkg.data,
                    img_err=cutout_dict_source['%s_err_cutout' % band].data,
                    wcs=cutout_dict_source['%s_img_cutout' % band].wcs,
                    ra=ra, dec=dec)

                # get aperture corrected photometry
                if band in ['F657N', 'F658N']:
                    obs = 'hst_ha'
                else:
                    obs = 'hst'
                apert_flux_dict = phot_tools.PhotTools.compute_ap_corr_phot_jimena(target=self.phot_hst_target_name, ra=ra, dec=dec,
                                                                  data=cutout_dict_source['%s_img_cutout' % band].data,
                                                                  err=cutout_dict_source['%s_err_cutout' % band].data,
                                                                  wcs=cutout_dict_source['%s_img_cutout' % band].wcs,
                                                                  obs=obs, band=band)
                mean_wave = helper_func.ObsTools.get_hst_band_wave(
                    band=band,
                    instrument=helper_func.ObsTools.get_hst_instrument(target=self.phot_hst_target_name, band=band))
                min_wave = helper_func.ObsTools.get_hst_band_wave(
                    band=band,
                    instrument=helper_func.ObsTools.get_hst_instrument(target=self.phot_hst_target_name, band=band),
                    wave_estimator='min_wave')
                max_wave = helper_func.ObsTools.get_hst_band_wave(
                    band=band,
                    instrument=helper_func.ObsTools.get_hst_instrument(target=self.phot_hst_target_name, band=band),
                    wave_estimator='max_wave')


                flux_dict.update({
                    band: {
                        'mean_wave':mean_wave,
                        'min_wave':min_wave,
                        'max_wave':max_wave,
                        'apert_corr_flux': apert_flux_dict['flux'],
                        'apert_corr_flux_err': apert_flux_dict['flux_err'],
                        'morph_flux': phot_dict['flux'],
                        'morph_flux_err': phot_dict['flux_err']
                    }
                })

                ax_sed.errorbar(mean_wave, apert_flux_dict['flux'],
                                xerr=[[mean_wave - min_wave], [max_wave - mean_wave]],
                                yerr=apert_flux_dict['flux_err'],
                                fmt='v', color='k', ms=20)

                ax_sed.errorbar(mean_wave, phot_dict['flux'],
                                xerr=[[mean_wave - min_wave], [max_wave - mean_wave]],
                                yerr=phot_dict['flux_err'],
                                fmt='o', color=fig_dict['hst_broad_band_color'], ms=20)

                # plot
                # define axis
                ax_stamp, ax_bkg, ax_slit_profile, ax_slit_profile_bkg_sub, ax_cbar_stamp, ax_cbar_bkg = (
                    self.create_phot_morph_axis(fig=fig, fig_dict=fig_dict, col_index=col_index, row_index=row_index,
                                                stamp_wcs=cutout_dict_source['%s_img_cutout' % band].wcs,
                                                bkg_wcs=cutout_stamp_bkg.wcs))
                # plot stamp and bkg!
                self.plot_stamp_bkg_cbar(
                    ax_stamp=ax_stamp, ax_bkg=ax_bkg, ax_cbar_stamp=ax_cbar_stamp, ax_cbar_bkg=ax_cbar_bkg,
                    cutout_stamp_data=cutout_dict_source['%s_img_cutout' % band], cutout_stamp_bkg=cutout_stamp_bkg,
                                    ra=ra, dec=dec, fig_dict=fig_dict, instrument='hst', band=band, target_name=self.phot_target_name)
                self.plot_radial_profiles(ax_slit_profile=ax_slit_profile,
                                          ax_slit_profile_bkg_sub=ax_slit_profile_bkg_sub,
                                          rad_profile_dict=rad_profile_dict,
                                          rad_profile_bkg_sub_dict=rad_profile_bkg_sub_dict,
                                          psf_dict=psf_dict,
                                          phot_dict=phot_dict,
                                          median_bkg=np.median(cutout_stamp_bkg.data),
                                          median_bkg_rms=np.median(cutout_stamp_bkg_rms.data))



                col_index +=2
                if col_index > 7:
                    col_index = 0
                    row_index -= 1

        if nircam_band_list:
            cutout_dict_source = self.get_band_cutout_dict(
                ra_cutout=ra, dec_cutout=dec, cutout_size=fig_dict['stamp_size_nircam'], band_list=nircam_band_list,
                include_err=True)
            for band in nircam_band_list:
                # get psf dict
                psf_dict = phot_tools.PSFTools.load_jwst_psf_dict(band=band, instrument='nircam')
                # get the background
                cutout_stamp_bkg, cutout_stamp_bkg_rms = self.get_scaled_bkg(
                    ra=ra, dec=dec, band=band, scale_size_arcsec=psf_dict['gaussian_fwhm'],
                    cutout_size=fig_dict['stamp_size_nircam'],
                    bkg_img_size_factor=fig_dict['bkg_img_size_factor_nircam'],
                    box_size_factor=fig_dict['box_size_factor_nircam'],
                    filter_size_factor=fig_dict['filter_size_factor_nircam'],
                    do_sigma_clip=True, sigma=3.0, maxiters=10, bkg_method='SExtractorBackground')

                rad_profile_dict = self.get_rad_profile_dict(img=cutout_dict_source['%s_img_cutout' % band].data,
                                                             wcs=cutout_dict_source['%s_img_cutout' % band].wcs,
                                                             ra=ra, dec=dec, n_slits=fig_dict['n_profile_slits'],
                                                             max_rad_arcsec=fig_dict['max_rad_profile_nircam_arcsec'],
                                                             img_err=cutout_dict_source['%s_err_cutout' % band].data)
                rad_profile_bkg_sub_dict = self.get_rad_profile_dict(
                    img=cutout_dict_source['%s_img_cutout' % band].data - cutout_stamp_bkg.data,
                    wcs=cutout_dict_source['%s_img_cutout' % band].wcs, ra=ra, dec=dec,
                    n_slits=fig_dict['n_profile_slits'], max_rad_arcsec=fig_dict['max_rad_profile_nircam_arcsec'],
                    img_err=cutout_dict_source['%s_err_cutout' % band].data)

                phot_dict = self.measure_morph_photometry(
                    rad_profile_dict=rad_profile_bkg_sub_dict, psf_dict=psf_dict,
                    img=cutout_dict_source['%s_img_cutout' % band].data, bkg=cutout_stamp_bkg.data,
                    img_err=cutout_dict_source['%s_err_cutout' % band].data,
                    wcs=cutout_dict_source['%s_img_cutout' % band].wcs,
                    ra=ra, dec=dec)

                apert_flux_dict = phot_tools.PhotTools.compute_ap_corr_phot_jimena(target=self.phot_nircam_target_name, ra=ra, dec=dec,
                                                                  data=cutout_dict_source['%s_img_cutout' % band].data,
                                                                  err=cutout_dict_source['%s_err_cutout' % band].data,
                                                                  wcs=cutout_dict_source['%s_img_cutout' % band].wcs,
                                                                  obs='nircam', band=band)
                mean_wave = helper_func.ObsTools.get_jwst_band_wave(band=band, instrument='nircam')
                min_wave = helper_func.ObsTools.get_jwst_band_wave(band=band, instrument='nircam',
                                                                   wave_estimator='min_wave')
                max_wave = helper_func.ObsTools.get_jwst_band_wave(band=band, instrument='nircam',
                                                                   wave_estimator='max_wave')

                flux_dict.update({
                    band: {
                        'mean_wave': mean_wave,
                        'min_wave': min_wave,
                        'max_wave': max_wave,
                        'apert_corr_flux': apert_flux_dict['flux'],
                        'apert_corr_flux_err': apert_flux_dict['flux_err'],
                        'morph_flux': phot_dict['flux'],
                        'morph_flux_err': phot_dict['flux_err']
                    }
                })

                ax_sed.errorbar(mean_wave, apert_flux_dict['flux'],
                                xerr=[[mean_wave - min_wave], [max_wave - mean_wave]],
                                yerr=apert_flux_dict['flux_err'],
                                fmt='v', color='k', ms=20)
                ax_sed.errorbar(mean_wave, phot_dict['flux'],
                                xerr=[[mean_wave - min_wave], [max_wave - mean_wave]],
                                yerr=phot_dict['flux_err'],
                                fmt='o', color=fig_dict['nircam_color'], ms=20)

                # plot
                # define axis
                ax_stamp, ax_bkg, ax_slit_profile, ax_slit_profile_bkg_sub, ax_cbar_stamp, ax_cbar_bkg = (
                    self.create_phot_morph_axis(fig=fig, fig_dict=fig_dict, col_index=col_index, row_index=row_index,
                                                stamp_wcs=cutout_dict_source['%s_img_cutout' % band].wcs,
                                                bkg_wcs=cutout_stamp_bkg.wcs))
                # plot stamp and bkg!
                self.plot_stamp_bkg_cbar(
                    ax_stamp=ax_stamp, ax_bkg=ax_bkg, ax_cbar_stamp=ax_cbar_stamp, ax_cbar_bkg=ax_cbar_bkg,
                    cutout_stamp_data=cutout_dict_source['%s_img_cutout' % band], cutout_stamp_bkg=cutout_stamp_bkg,
                    ra=ra, dec=dec, fig_dict=fig_dict, instrument='nircam', band=band, target_name=self.phot_target_name)
                self.plot_radial_profiles(ax_slit_profile=ax_slit_profile,
                                          ax_slit_profile_bkg_sub=ax_slit_profile_bkg_sub,
                                          rad_profile_dict=rad_profile_dict,
                                          rad_profile_bkg_sub_dict=rad_profile_bkg_sub_dict,
                                          psf_dict=psf_dict,
                                          phot_dict=phot_dict,
                                          median_bkg=np.median(cutout_stamp_bkg.data),
                                          median_bkg_rms=np.median(cutout_stamp_bkg_rms.data))

                col_index += 2
                if col_index > 7:
                    col_index = 0
                    row_index -= 1

        if miri_band_list:
            cutout_dict_source = self.get_band_cutout_dict(
                ra_cutout=ra, dec_cutout=dec, cutout_size=fig_dict['stamp_size_miri'], band_list=miri_band_list,
                include_err=True)
            for band in miri_band_list:
                # get psf dict
                psf_dict = phot_tools.PSFTools.load_jwst_psf_dict(band=band, instrument='miri')
                # get the background
                cutout_stamp_bkg, cutout_stamp_bkg_rms = self.get_scaled_bkg(
                    ra=ra, dec=dec, band=band, scale_size_arcsec=psf_dict['gaussian_fwhm'],
                    cutout_size=fig_dict['stamp_size_miri'], bkg_img_size_factor=fig_dict['bkg_img_size_factor_miri'],
                    box_size_factor=fig_dict['box_size_factor_miri'],
                    filter_size_factor=fig_dict['filter_size_factor_miri'],
                    do_sigma_clip=True, sigma=3.0, maxiters=10, bkg_method='SExtractorBackground')

                rad_profile_dict = self.get_rad_profile_dict(img=cutout_dict_source['%s_img_cutout' % band].data,
                                                             wcs=cutout_dict_source['%s_img_cutout' % band].wcs,
                                                             ra=ra, dec=dec, n_slits=fig_dict['n_profile_slits'],
                                                             max_rad_arcsec=fig_dict['max_rad_profile_miri_arcsec'],
                                                             img_err=cutout_dict_source['%s_err_cutout' % band].data)
                rad_profile_bkg_sub_dict = self.get_rad_profile_dict(
                    img=cutout_dict_source['%s_img_cutout' % band].data - cutout_stamp_bkg.data,
                    wcs=cutout_dict_source['%s_img_cutout' % band].wcs, ra=ra, dec=dec,
                    n_slits=fig_dict['n_profile_slits'], max_rad_arcsec=fig_dict['max_rad_profile_miri_arcsec'],
                    img_err=cutout_dict_source['%s_err_cutout' % band].data)
                # print(band)
                # plt.close(fig)
                # plt.imshow(cutout_dict_source['%s_img_cutout' % band].data)
                # plt.show()
                phot_dict = self.measure_morph_photometry(
                    rad_profile_dict=rad_profile_bkg_sub_dict, psf_dict=psf_dict,
                    img=cutout_dict_source['%s_img_cutout' % band].data, bkg=cutout_stamp_bkg.data,
                    img_err=cutout_dict_source['%s_err_cutout' % band].data,
                    wcs=cutout_dict_source['%s_img_cutout' % band].wcs,
                    ra=ra, dec=dec)

                apert_flux_dict = phot_tools.PhotTools.compute_ap_corr_phot_jimena(target=self.phot_nircam_target_name, ra=ra,
                                                                             dec=dec,
                                                                             data=cutout_dict_source[
                                                                                 '%s_img_cutout' % band].data,
                                                                             err=cutout_dict_source[
                                                                                 '%s_err_cutout' % band].data,
                                                                             wcs=cutout_dict_source[
                                                                                 '%s_img_cutout' % band].wcs,
                                                                             obs='miri', band=band)
                mean_wave = helper_func.ObsTools.get_jwst_band_wave(band=band, instrument='miri')
                min_wave = helper_func.ObsTools.get_jwst_band_wave(band=band, instrument='miri',
                                                                   wave_estimator='min_wave')
                max_wave = helper_func.ObsTools.get_jwst_band_wave(band=band, instrument='miri',
                                                                   wave_estimator='max_wave')

                flux_dict.update({
                    band: {
                        'mean_wave': mean_wave,
                        'min_wave': min_wave,
                        'max_wave': max_wave,
                        'apert_corr_flux': apert_flux_dict['flux'],
                        'apert_corr_flux_err': apert_flux_dict['flux_err'],
                        'morph_flux': phot_dict['flux'],
                        'morph_flux_err': phot_dict['flux_err']
                    }
                })
                ax_sed.errorbar(mean_wave, apert_flux_dict['flux'],
                                xerr=[[mean_wave - min_wave], [max_wave - mean_wave]],
                                yerr=apert_flux_dict['flux_err'],
                                fmt='v', color='k', ms=20)
                ax_sed.errorbar(mean_wave, phot_dict['flux'],
                                xerr=[[mean_wave - min_wave], [max_wave - mean_wave]],
                                yerr=phot_dict['flux_err'],
                                fmt='o', color=fig_dict['miri_color'], ms=20)

                # plot
                # define axis
                ax_stamp, ax_bkg, ax_slit_profile, ax_slit_profile_bkg_sub, ax_cbar_stamp, ax_cbar_bkg = (
                    self.create_phot_morph_axis(fig=fig, fig_dict=fig_dict, col_index=col_index, row_index=row_index,
                                                stamp_wcs=cutout_dict_source['%s_img_cutout' % band].wcs,
                                                bkg_wcs=cutout_stamp_bkg.wcs))
                # plot stamp and bkg!
                self.plot_stamp_bkg_cbar(
                    ax_stamp=ax_stamp, ax_bkg=ax_bkg, ax_cbar_stamp=ax_cbar_stamp, ax_cbar_bkg=ax_cbar_bkg,
                    cutout_stamp_data=cutout_dict_source['%s_img_cutout' % band], cutout_stamp_bkg=cutout_stamp_bkg,
                    ra=ra, dec=dec, fig_dict=fig_dict, instrument='miri', band=band, target_name=self.phot_target_name)
                self.plot_radial_profiles(ax_slit_profile=ax_slit_profile,
                                          ax_slit_profile_bkg_sub=ax_slit_profile_bkg_sub,
                                          rad_profile_dict=rad_profile_dict,
                                          rad_profile_bkg_sub_dict=rad_profile_bkg_sub_dict,
                                          psf_dict=psf_dict,
                                          phot_dict=phot_dict,
                                          median_bkg=np.median(cutout_stamp_bkg.data),
                                          median_bkg_rms=np.median(cutout_stamp_bkg_rms.data))

                col_index += 2
                if col_index > 7:
                    col_index = 0
                    row_index -= 1

        ax_sed.set_yscale('log')
        ax_sed.set_xscale('log')
        ax_sed.set_xlabel(r'Wavelength [$\mu$m]', fontsize=fig_dict['sed_label_size'])
        ax_sed.set_ylabel(r'Flux [mJy]', fontsize=fig_dict['sed_label_size'])
        ax_sed.tick_params(axis='both', which='both', width=2, direction='in', labelsize=fig_dict['sed_label_size'])

        if return_flux_dict:
            return flux_dict

    def plot_sed_panel(self, fig, fig_dict, ra, dec):

        hst_broad_band_list = self.get_covered_hst_broad_band_list(ra=ra, dec=dec)
        hst_ha_band = self.get_covered_hst_ha_band(ra=ra, dec=dec)
        nircam_band_list = self.get_covered_nircam_band_list(ra=ra, dec=dec)
        miri_band_list = self.get_covered_miri_band_list(ra=ra, dec=dec)

        band_list = self.get_covered_hst_broad_band_list(ra=ra, dec=dec)
        if helper_func.ObsTools.check_hst_ha_obs(target=self.phot_hst_target_name):
            if self.check_coords_covered_by_band(telescope="hst", ra=ra, dec=dec, band=helper_func.ObsTools.get_hst_ha_band(
                    target=self.phot_hst_target_name), max_dist_dist2hull_arcsec=2):
                band_list += [hst_ha_band]
                # check if Ha- continuum subtracted image is available
                if self.phot_hst_ha_cont_sub_target_name in phangs_info.hst_ha_cont_sub_dict.keys():
                    band_list += [hst_ha_band + '_cont_sub']
        band_list += nircam_band_list + miri_band_list
        print(band_list)
        # load data in case it is not yet loaded
        self.load_phangs_bands(band_list=band_list, flux_unit='mJy', load_err=True, load_hst=True, load_hst_ha=True,
                              load_nircam=True, load_miri=True, load_astrosat=False)
        # load large cutout dict for flux density estimation
        self.change_phangs_band_units(band_list=band_list, new_unit='mJy')


        # # load cutout stamps
        # cutout_dict_stamp = self.get_band_cutout_dict(ra_cutout=ra, dec_cutout=dec, cutout_size=fig_dict['sed_size'],
        #                                               band_list=band_list, include_err=True)

        # add sed axis
        ax_sed = fig.add_axes([
            fig_dict['sed_left_align'],
            fig_dict['sed_bottom_align'],
            fig_dict['sed_width'],
            fig_dict['sed_height'],
        ])

        for band in hst_broad_band_list:
            morph_flux_dict = self.compute_morph_phot(
                ra=ra, dec=dec, band=band,
                instrument=helper_func.ObsTools.get_hst_instrument(target=self.phot_hst_target_name, band=band),
                cutout_size=fig_dict['sed_size_hst'])

            mean_wave = helper_func.ObsTools.get_hst_band_wave(
                band=band, instrument=helper_func.ObsTools.get_hst_instrument(target=self.phot_hst_target_name, band=band))
            min_wave = helper_func.ObsTools.get_hst_band_wave(
                band=band, instrument=helper_func.ObsTools.get_hst_instrument(target=self.phot_hst_target_name, band=band),
                wave_estimator='min_wave')
            max_wave = helper_func.ObsTools.get_hst_band_wave(
                band=band, instrument=helper_func.ObsTools.get_hst_instrument(target=self.phot_hst_target_name, band=band),
                wave_estimator='max_wave')

            ax_sed.errorbar(mean_wave, morph_flux_dict['flux'],
                            xerr=[[mean_wave - min_wave], [max_wave - mean_wave]],
                            yerr=morph_flux_dict['flux_err'],
                            fmt='v', color=fig_dict['hst_broad_band_color'], ms=20)


        if hst_ha_band:
            morph_flux_dict = self.compute_morph_phot(
                ra=ra, dec=dec, band=hst_ha_band,
                instrument=helper_func.ObsTools.get_hst_instrument(target=self.phot_hst_target_name, band=hst_ha_band),
                cutout_size=fig_dict['sed_size_hst'])

            mean_wave = helper_func.ObsTools.get_hst_band_wave(
                band=hst_ha_band, instrument=helper_func.ObsTools.get_hst_instrument(target=self.phot_hst_target_name, band=hst_ha_band))
            min_wave = helper_func.ObsTools.get_hst_band_wave(
                band=hst_ha_band, instrument=helper_func.ObsTools.get_hst_instrument(target=self.phot_hst_target_name, band=hst_ha_band),
                wave_estimator='min_wave')
            max_wave = helper_func.ObsTools.get_hst_band_wave(
                band=hst_ha_band, instrument=helper_func.ObsTools.get_hst_instrument(target=self.phot_hst_target_name, band=hst_ha_band),
                wave_estimator='max_wave')

            ax_sed.errorbar(mean_wave, morph_flux_dict['flux'],
                            xerr=[[mean_wave - min_wave], [max_wave - mean_wave]],
                            yerr=morph_flux_dict['flux_err'],
                            fmt='v', color=fig_dict['hst_ha_color'], ms=20)


        for band in nircam_band_list:
            morph_flux_dict = self.compute_morph_phot(
                ra=ra, dec=dec, band=band,
                instrument='nircam',
                cutout_size=fig_dict['sed_size_nircam'])

            mean_wave = helper_func.ObsTools.get_jwst_band_wave(
                band=band,
                instrument='nircam')
            min_wave = helper_func.ObsTools.get_jwst_band_wave(
                band=band,
                instrument='nircam',
                wave_estimator='min_wave')
            max_wave = helper_func.ObsTools.get_jwst_band_wave(
                band=band,
                instrument='nircam',
                wave_estimator='max_wave')

            ax_sed.errorbar(mean_wave, morph_flux_dict['flux'],
                            xerr=[[mean_wave - min_wave], [max_wave - mean_wave]],
                            yerr=morph_flux_dict['flux_err'],
                            fmt='v', color=fig_dict['nircam_color'], ms=20)




        for band in miri_band_list:
            morph_flux_dict = self.compute_morph_phot(
                ra=ra, dec=dec, band=band,
                instrument='miri',
                cutout_size=fig_dict['sed_size_miri'])

            mean_wave = helper_func.ObsTools.get_jwst_band_wave(
                band=band,
                instrument='miri')
            min_wave = helper_func.ObsTools.get_jwst_band_wave(
                band=band,
                instrument='miri',
                wave_estimator='min_wave')
            max_wave = helper_func.ObsTools.get_jwst_band_wave(
                band=band,
                instrument='miri',
                wave_estimator='max_wave')

            ax_sed.errorbar(mean_wave, morph_flux_dict['flux'],
                            xerr=[[mean_wave - min_wave], [max_wave - mean_wave]],
                            yerr=morph_flux_dict['flux_err'],
                            fmt='v', color=fig_dict['miri_color'], ms=20)

        # add cluster catalog cross-match
        if self.phot_hst_target_name in phangs_info.hst_cluster_cat_target_list:
            phangs_cluster = ClusterCatAccess()
            cross_match_map = phangs_cluster.get_hst_cc_cross_match_mask(target=self.phot_hst_target_name,
                                                                         ra=ra, dec=dec, toleance_arcsec=0.1,
                                                                         classify='human', cluster_class='class12')
            if sum(cross_match_map) > 0:
                sed_age = phangs_cluster.get_hst_cc_age(target=self.phot_hst_target_name)[cross_match_map]
                sed_mstar = phangs_cluster.get_hst_cc_mstar(target=self.phot_hst_target_name)[cross_match_map]
                sed_ebv = phangs_cluster.get_hst_cc_ebv(target=self.phot_hst_target_name)[cross_match_map]
                age_string = plotting_tools.StrTools.age2label(age=sed_age[0])
                mstar_string = plotting_tools.StrTools.mstar2label(mstar=sed_mstar)
                muse_title_str = (r'HST-SED fit params'
                                  + '\n' +
                                  r'age = ' + age_string
                                  + '\n' +
                                  r'M$_{*}$ = ' + mstar_string
                                  + '\n' +
                                  r'E(B-V) = %.2f mag' % sed_ebv)
                t = ax_sed.text(1.03, 0.93, muse_title_str, horizontalalignment='left', verticalalignment='top',
                                      transform=ax_sed.transAxes, fontsize=fig_dict['sed_title_font_size'])
                t.set_bbox(dict(facecolor='grey', alpha=0.5, edgecolor='black', boxstyle='round,pad=1'))

        ax_sed.set_yscale('log')
        ax_sed.set_xscale('log')
        ax_sed.set_xlabel(r'Wavelength [$\mu$m]', fontsize=fig_dict['sed_label_size'])
        ax_sed.set_ylabel(r'Flux [mJy]', fontsize=fig_dict['sed_label_size'])
        ax_sed.tick_params(axis='both', which='both', width=2, direction='in', labelsize=fig_dict['sed_label_size'])

    def compute_ha_ew(self, fig, fig_dict, ra, dec):

        if not self.check_coords_covered_by_band(obs="hst", ra=ra, dec=dec,
                                             band=helper_func.ObsTools.get_hst_ha_band(target=self.phot_hst_target_name),
                                             max_dist_dist2hull_arcsec=2):
            return None
        left_band = 'F555W'
        right_band = 'F814W'
        hst_ha_band = self.get_covered_hst_ha_band(ra=ra, dec=dec)

        band_list =  [left_band] + [right_band] + [hst_ha_band]

        # load data in case it is not yet loaded
        self.load_phangs_bands(band_list=band_list, flux_unit='mJy', load_err=True, load_hst=True, load_hst_ha=True,
                               load_nircam=True, load_miri=True, load_astrosat=False)

        # load large cutout dict for flux density estimation
        self.change_phangs_band_units(band_list=band_list, new_unit='mJy')

        # load cutout stamps
        cutout_dict_stamp = self.get_band_cutout_dict(ra_cutout=ra, dec_cutout=dec, cutout_size=fig_dict['sed_size'],
                                                      band_list=band_list, include_err=True)

        flux_dict_left_band_appr_corr = phot_tools.PhotTools.compute_ap_corr_phot_jimena(target=self.phot_hst_target_name, ra=ra, dec=dec,
                                                            data=cutout_dict_stamp['%s_img_cutout' % left_band].data,
                                                            err=cutout_dict_stamp['%s_err_cutout' % left_band].data,
                                                            wcs=cutout_dict_stamp['%s_img_cutout' % left_band].wcs,
                                                            obs='hst', band=left_band)
        flux_dict_right_band_appr_corr = phot_tools.PhotTools.compute_ap_corr_phot_jimena(target=self.phot_hst_target_name, ra=ra, dec=dec,
                                                            data=cutout_dict_stamp['%s_img_cutout' % right_band].data,
                                                            err=cutout_dict_stamp['%s_err_cutout' % right_band].data,
                                                            wcs=cutout_dict_stamp['%s_img_cutout' % right_band].wcs,
                                                            obs='hst', band=right_band)
        flux_dict_ha_appr_corr = phot_tools.PhotTools.compute_ap_corr_phot_jimena(target=self.phot_hst_target_name, ra=ra, dec=dec,
                                                            data=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].data,
                                                            err=cutout_dict_stamp['%s_err_cutout' % hst_ha_band].data,
                                                            wcs=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].wcs,
                                                            obs='hst_ha', band=hst_ha_band)
        # compute EW
        ha_ew_apr_corr, ha_ew_err_apr_corr = phot_tools.PhotTools.compute_hst_photo_ew(
            target=self.phot_hst_target_name, left_band=left_band, right_band=right_band,  narrow_band=hst_ha_band,
            flux_left_band=flux_dict_left_band_appr_corr['flux'],
            flux_right_band=flux_dict_right_band_appr_corr['flux'],
            flux_narrow_band=flux_dict_ha_appr_corr['flux'],
            flux_err_left_band=flux_dict_left_band_appr_corr['flux_err'],
            flux_err_right_band=flux_dict_right_band_appr_corr['flux_err'],
            flux_err_narrow_band=flux_dict_ha_appr_corr['flux_err'])

        # get multiple EW measurements
        annulus_rad_in_left = helper_func.CoordTools.transform_pix2world_scale(
            length_in_pix=fig_dict['ha_ew_annulus_rad_in_pix'],
            wcs=cutout_dict_stamp['%s_img_cutout' % left_band].wcs)
        annulus_rad_out_left = helper_func.CoordTools.transform_pix2world_scale(
            length_in_pix=fig_dict['ha_ew_annulus_rad_out_pix'],
            wcs=cutout_dict_stamp['%s_img_cutout' % left_band].wcs)
        annulus_rad_in_right = helper_func.CoordTools.transform_pix2world_scale(
            length_in_pix=fig_dict['ha_ew_annulus_rad_in_pix'],
            wcs=cutout_dict_stamp['%s_img_cutout' % right_band].wcs)
        annulus_rad_out_right = helper_func.CoordTools.transform_pix2world_scale(
            length_in_pix=fig_dict['ha_ew_annulus_rad_out_pix'],
            wcs=cutout_dict_stamp['%s_img_cutout' % right_band].wcs)
        annulus_rad_in_hst_ha = helper_func.CoordTools.transform_pix2world_scale(
            length_in_pix=fig_dict['ha_ew_annulus_rad_in_pix'],
            wcs=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].wcs)
        annulus_rad_out_hst_ha = helper_func.CoordTools.transform_pix2world_scale(
            length_in_pix=fig_dict['ha_ew_annulus_rad_out_pix'],
            wcs=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].wcs)

        ew_apr_list = []
        ew_err_apr_list = []
        rad_list = []
        for rad in fig_dict['ha_ew_ap_rad_pix_list']:
            aperture_rad_left = helper_func.CoordTools.transform_pix2world_scale(
                length_in_pix=rad, wcs=cutout_dict_stamp['%s_img_cutout' % left_band].wcs)
            aperture_rad_right = helper_func.CoordTools.transform_pix2world_scale(
                length_in_pix=rad, wcs=cutout_dict_stamp['%s_img_cutout' % right_band].wcs)
            aperture_rad_hst_ha = helper_func.CoordTools.transform_pix2world_scale(
                length_in_pix=rad, wcs=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].wcs)
            rad_list.append(aperture_rad_hst_ha)

            flux_dict_left_band = phot_tools.PhotTools.compute_phot_jimena(target=self.phot_hst_target_name, ra=ra, dec=dec,
                                                                data=cutout_dict_stamp['%s_img_cutout' % left_band].data,
                                                                err=cutout_dict_stamp['%s_err_cutout' % left_band].data,
                                                                wcs=cutout_dict_stamp['%s_img_cutout' % left_band].wcs,
                                                                obs='hst', band=left_band, aperture_rad=aperture_rad_left,
                                                                annulus_rad_in=annulus_rad_in_left,
                                                                annulus_rad_out=annulus_rad_out_left)
            flux_dict_right_band = phot_tools.PhotTools.compute_phot_jimena(target=self.phot_hst_target_name, ra=ra, dec=dec,
                                                                data=cutout_dict_stamp['%s_img_cutout' % right_band].data,
                                                                err=cutout_dict_stamp['%s_err_cutout' % right_band].data,
                                                                wcs=cutout_dict_stamp['%s_img_cutout' % right_band].wcs,
                                                                obs='hst', band=right_band, aperture_rad=aperture_rad_right,
                                                                annulus_rad_in=annulus_rad_in_right,
                                                                annulus_rad_out=annulus_rad_out_right)
            flux_dict_ha = phot_tools.PhotTools.compute_phot_jimena(target=self.phot_hst_target_name, ra=ra, dec=dec,
                                                                data=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].data,
                                                                err=cutout_dict_stamp['%s_err_cutout' % hst_ha_band].data,
                                                                wcs=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].wcs,
                                                                obs='hst_ha', band=hst_ha_band, aperture_rad=aperture_rad_hst_ha,
                                                                annulus_rad_in=annulus_rad_in_hst_ha,
                                                                annulus_rad_out=annulus_rad_out_hst_ha)

            ha_ew_at_rad, ha_ew_err_at_rad = phot_tools.PhotTools.compute_hst_photo_ew(
                target=self.phot_hst_target_name, left_band=left_band, right_band=right_band,  narrow_band=hst_ha_band,
                flux_left_band=flux_dict_left_band['flux'],
                flux_right_band=flux_dict_right_band['flux'],
                flux_narrow_band=flux_dict_ha['flux'],
                flux_err_left_band=flux_dict_left_band['flux_err'],
                flux_err_right_band=flux_dict_right_band['flux_err'],
                flux_err_narrow_band=flux_dict_ha['flux_err'])
            ew_apr_list.append(ha_ew_at_rad)
            ew_err_apr_list.append(ha_ew_err_at_rad)

        # add sed axis
        ax_ha_ew = fig.add_axes([
            fig_dict['ha_ew_left_align'],
            fig_dict['ha_ew_bottom_align'],
            fig_dict['ha_ew_width'],
            fig_dict['ha_ew_height'],
        ])

        ax_ha_ew.plot(rad_list, ew_apr_list, color='k', linewidth=2)
        ax_ha_ew.errorbar(rad_list, ew_apr_list, yerr=ew_err_apr_list, color='k', fmt='.')
        ax_ha_ew.fill_between([np.min(rad_list), np.max(rad_list)], y1=ha_ew_apr_corr+ha_ew_err_apr_corr,
                              y2=ha_ew_apr_corr-ha_ew_err_apr_corr, color='grey', alpha=0.5)
        ax_ha_ew.plot([np.min(rad_list), np.max(rad_list)], [ha_ew_apr_corr,ha_ew_apr_corr], color='red', linewidth=3, label='Apert. corr.')

        ax_ha_ew.plot([phangs_info.muse_obs_res_dict[self.phot_target_name]['copt_res'] / 2,
                       phangs_info.muse_obs_res_dict[self.phot_target_name]['copt_res'] / 2],
                      [np.min(ew_apr_list), np.max(ew_apr_list)],
                      color='k', linewidth=3, linestyle='--' ,label='MUSE res. rad')


        ax_ha_ew.set_xlabel(r'rad. [\"]', fontsize=fig_dict['ha_ew_label_size'])
        ax_ha_ew.set_ylabel(r'EW(H$\alpha$) [$\AA$]', fontsize=fig_dict['ha_ew_label_size'])
        ax_ha_ew.tick_params(axis='both', which='both', width=2, direction='in', labelsize=fig_dict['ha_ew_label_size'])
        ax_ha_ew.legend(frameon=False, fontsize=fig_dict['ha_ew_label_size'])

    def plot_muse_spec(self, fig, fig_dict, ra, dec, ppxf_fit_dict):

        if not self.check_coords_covered_by_muse(ra=ra, dec=dec, res='copt', max_dist_dist2hull_arcsec=2):
            return None

        # add sed axis
        ax_muse_spec = fig.add_axes([
            fig_dict['muse_spec_left_align'],
            fig_dict['muse_spec_bottom_align'],
            fig_dict['muse_spec_width'],
            fig_dict['muse_spec_height'],
        ])

        ax_muse_spec.plot(ppxf_fit_dict['wave'], ppxf_fit_dict['total_flux'] * 1e16, linewidth=3, color='k',
                          label='Obs. Spectrum')
        ax_muse_spec.plot(ppxf_fit_dict['wave'], ppxf_fit_dict['best_fit'] * 1e16, linewidth=3, color='tab:red',
                          label='Best total fit')
        ax_muse_spec.plot(ppxf_fit_dict['wave'], ppxf_fit_dict['continuum_best_fit'] * 1e16, linewidth=3,
                          color='tab:orange', label='Stellar Continuum fit')
        # plt.rc('text', usetex=True)
        # muse_title_str = (r'$\underline{\rm PPxF\,fit\,parameters}$'
        muse_title_str = (r'${\rm PPxF\,fit\,parameters}$'
                          + '\n' +
                          r'age = %.2f Myr ' % (10 ** (ppxf_fit_dict['ages']) * 1e-6)
                          + '\n' +
                          r'[M/H] = %.3f' % ppxf_fit_dict['met']
                          + '\n' +
                          r'A$_{\rm v} (star) = $%.2f mag' % (ppxf_fit_dict['star_red'])
                          + '\n' +
                          r'A$_{\rm v} (gas) = $%.2f mag' % (ppxf_fit_dict['gas_red']))
        t = ax_muse_spec.text(1.02, 0.05, muse_title_str, horizontalalignment='left', verticalalignment='bottom',
                              transform=ax_muse_spec.transAxes, fontsize=fig_dict['muse_spec_title_font_size'])
        t.set_bbox(dict(facecolor='grey', alpha=0.5, edgecolor='black', boxstyle='round,pad=1'))

        ax_muse_spec.tick_params(axis='both', which='both', width=2, direction='in',
                                 labelsize=fig_dict['muse_spec_label_size'])
        ax_muse_spec.set_xlabel(r'Wavelength [${\rm \AA}$]', fontsize=fig_dict['muse_spec_label_size'])
        ax_muse_spec.set_ylabel(r'$\phi$ [10$^{-16}$ erg cm$^{-2}$ s$^{-1}$ ${\rm \AA^{-1}}$]',
                                fontsize=fig_dict['muse_spec_label_size'])

        ax_muse_spec.legend(frameon=False, fontsize=fig_dict['muse_spec_label_size'])

        # get x limits
        if fig_dict['muse_spec_x_lim'][0] == 'min':
            wave_min = np.nanmin(ppxf_fit_dict['wave'])
        else:
            wave_min = fig_dict['muse_spec_x_lim'][0]

        if fig_dict['muse_spec_x_lim'][1] == 'max':
            wave_max = np.nanmax(ppxf_fit_dict['wave'])
        else:
            wave_max = fig_dict['muse_spec_x_lim'][1]

        # get y-limits
        mask_displayed_wave = (ppxf_fit_dict['wave'] > wave_min) & (ppxf_fit_dict['wave'] < wave_max)
        if isinstance(fig_dict['muse_spec_y_lim'], str):
            if fig_dict['muse_spec_y_lim'] == 'cont':
                flux_min = np.nanmin(ppxf_fit_dict['continuum_best_fit'][mask_displayed_wave] * 1e16)
                flux_max = np.nanmax(ppxf_fit_dict['continuum_best_fit'][mask_displayed_wave] * 1e16)
                flux_lim_lo = flux_min - (flux_max - flux_min) * 0.05
                flux_lim_hi = flux_max + (flux_max - flux_min) * 0.05
            elif fig_dict['muse_spec_y_lim'] == 'total':
                flux_min = np.nanmin(ppxf_fit_dict['total_flux'][mask_displayed_wave] * 1e16)
                flux_max = np.nanmax(ppxf_fit_dict['total_flux'][mask_displayed_wave] * 1e16)
                flux_lim_lo = flux_min - (flux_max - flux_min) * 0.05
                flux_lim_hi = flux_max + (flux_max - flux_min) * 0.05
            else:
                flux_lim_lo = None
                flux_lim_hi = None
        elif isinstance(fig_dict['muse_spec_y_lim'], tuple):
            flux_lim_lo, flux_lim_hi = fig_dict['muse_spec_y_lim']
        else:
            flux_lim_lo = None
            flux_lim_hi = None

        # set x-labels
        ax_muse_spec.set_xlim(wave_min, wave_max)
        ax_muse_spec.set_ylim(flux_lim_lo, flux_lim_hi)

        # plot ha plot cutout
        muse_dap_map_cutout_dict = self.get_muse_dap_map_cutout(
            ra_cutout=ra, dec_cutout=dec, cutout_size=fig_dict['muse_ha_zoom_in_size'],
            map_type_list=fig_dict['muse_ha_zoom_in_map_typ'], res=fig_dict['muse_ha_zoom_in_res'],
            ssp_model=fig_dict['muse_ha_zoom_in_ssp_model'])
        dap_data_identifier = helper_func.SpecHelper.get_dap_data_identifier(res=fig_dict['muse_ha_zoom_in_res'],
                                                                             ssp_model=fig_dict[
                                                                                 'muse_ha_zoom_in_ssp_model'])
        muse_ha_cutout = muse_dap_map_cutout_dict[
            '%s_%s_img_cutout' % (dap_data_identifier, fig_dict['muse_ha_zoom_in_map_typ'])]

        # add axis
        ax_muse_ha_zoom_in = fig.add_axes([
            fig_dict['muse_ha_zoom_in_left_align'],
            fig_dict['muse_ha_zoom_in_bottom_align'],
            fig_dict['muse_ha_zoom_in_width'],
            fig_dict['muse_ha_zoom_in_height']], projection=muse_ha_cutout.wcs)

        norm_muse_ha = plotting_tools.ColorBarTools.compute_cbar_norm(cutout_list=muse_ha_cutout.data, log_scale=True)
        ax_muse_ha_zoom_in.imshow(muse_ha_cutout.data, norm=norm_muse_ha, cmap='cividis')
        ax_muse_ha_zoom_in.set_title(r'MUSE H$\alpha$', fontsize=fig_dict['muse_spec_title_font_size'], color='k')
        plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_muse_ha_zoom_in,
                                                          pos=SkyCoord(ra=ra * u.deg, dec=dec * u.deg),
                                                          rad=ppxf_fit_dict['rad_arcsec'],
                                                          color='cyan', line_width=4)
        plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_muse_ha_zoom_in,
                                                          pos=SkyCoord(ra=ra * u.deg, dec=dec * u.deg),
                                                          rad=ppxf_fit_dict['rad_arcsec'],
                                                          color='k', line_width=4, line_style='--')
        ax_muse_ha_zoom_in.plot([], [], color='cyan', linewidth=4, label='Spec. apert.')
        ax_muse_ha_zoom_in.plot([], [], color='k', linewidth=4, linestyle='--', label='PSF FWHM')
        ax_muse_ha_zoom_in.legend(frameon=False, bbox_to_anchor=[0.95, 0.3], fontsize=fig_dict['muse_spec_label_size'])

        plotting_tools.WCSPlottingTools.plot_img_scale_bar(
            ax=ax_muse_ha_zoom_in, img_shape=muse_ha_cutout.data.shape,
            wcs=muse_ha_cutout.wcs,
            bar_length=fig_dict['muse_scale_bar_length_1'], length_unit='arcsec',
            bar_color='tab:red', text_color='tab:red',
            line_width=4, fontsize=fig_dict['muse_spec_label_size'],
            va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)
        plotting_tools.WCSPlottingTools.plot_img_scale_bar(
            ax=ax_muse_ha_zoom_in, img_shape=muse_ha_cutout.data.shape,
            wcs=muse_ha_cutout.wcs,
            bar_length=fig_dict['muse_scale_bar_length_2'], length_unit='pc',
            phangs_target=self.phot_target_name,
            bar_color='tab:red', text_color='tab:red',
            line_width=4, fontsize=fig_dict['muse_spec_label_size'],
            va='top', ha='right', x_offset=0.1, y_offset=0.15, text_y_offset_diff=0.01)
        plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_muse_ha_zoom_in, ra_tick_label=False,
                                                        dec_tick_label=False,
                                                        ra_axis_label=' ', dec_axis_label=' ',
                                                        fontsize=fig_dict['muse_spec_label_size'],
                                                        labelsize=fig_dict['muse_spec_label_size'])
        return None

    def plot_spec_features(self, fig, fig_dict, em_line_fit_dict):
        # add sed axis




        if ((4863 in em_line_fit_dict['ln_list']) & (4960 in em_line_fit_dict['ln_list']) &
                (5008 in em_line_fit_dict['ln_list'])):
            ax_hb_oiii_spec = fig.add_axes([
                fig_dict['hb_oiii_left_align'],
                fig_dict['hb_oiii_bottom_align'],
                fig_dict['hb_oiii_width'],
                fig_dict['hb_oiii_height'],
            ])
            ax_hb_oiii_res_spec = fig.add_axes([
                fig_dict['hb_oiii_res_left_align'],
                fig_dict['hb_oiii_res_bottom_align'],
                fig_dict['hb_oiii_res_width'],
                fig_dict['hb_oiii_res_height'],
            ])

            plotting_tools.SpecPlotTools.plot_em_line_spec(ax=ax_hb_oiii_spec, em_line_fit_dict=em_line_fit_dict,
                                                           line_list=[4863, 4960, 5008],
                                                           left_offset=fig_dict['hb_oiii_left_lim_offst'],
                                                           right_offset=fig_dict['hb_oiii_right_lim_offst'],
                                                           display_legend=True,
                                                           display_x_label=False,
                                                           ax_res=ax_hb_oiii_res_spec,
                                                           font_size_label=fig_dict['em_label_size'],
                                                           font_size_title=fig_dict['em_title_font_size'])

        if ((6550 in em_line_fit_dict['ln_list']) & (6565 in em_line_fit_dict['ln_list']) &
                (6585 in em_line_fit_dict['ln_list'])):
            ax_ha_nii_spec = fig.add_axes([
                fig_dict['ha_nii_left_align'],
                fig_dict['ha_nii_bottom_align'],
                fig_dict['ha_nii_width'],
                fig_dict['ha_nii_height'],
            ])
            ax_ha_nii_res_spec = fig.add_axes([
                fig_dict['ha_nii_res_left_align'],
                fig_dict['ha_nii_res_bottom_align'],
                fig_dict['ha_nii_res_width'],
                fig_dict['ha_nii_res_height'],
            ])
            plotting_tools.SpecPlotTools.plot_em_line_spec(ax=ax_ha_nii_spec, em_line_fit_dict=em_line_fit_dict,
                                                           line_list=[6550, 6565, 6585],
                                                           left_offset=fig_dict['ha_nii_left_lim_offst'],
                                                           right_offset=fig_dict['ha_nii_right_lim_offst'],
                                                           display_legend=False,
                                                           display_x_label=True,
                                                           ax_res=ax_ha_nii_res_spec,
                                                           font_size_label=fig_dict['em_label_size'],
                                                           font_size_title=fig_dict['em_title_font_size'])

        if 6302 in em_line_fit_dict['ln_list']:
            ax_oi6302_spec = fig.add_axes([
                fig_dict['oi6302_left_align'],
                fig_dict['oi6302_bottom_align'],
                fig_dict['oi6302_width'],
                fig_dict['oi6302_height'],
            ])
            ax_oi6302_res_spec = fig.add_axes([
                fig_dict['oi6302_res_left_align'],
                fig_dict['oi6302_res_bottom_align'],
                fig_dict['oi6302_res_width'],
                fig_dict['oi6302_res_height'],
            ])
            plotting_tools.SpecPlotTools.plot_em_line_spec(ax=ax_oi6302_spec, em_line_fit_dict=em_line_fit_dict,
                                                           line_list=[6302],
                                                           left_offset=fig_dict['oi6302_left_lim_offst'],
                                                           right_offset=fig_dict['oi6302_right_lim_offst'],
                                                           display_legend=False,
                                                           display_x_label=False,
                                                           display_y_label=False,
                                                           ax_res=ax_oi6302_res_spec,
                                                           font_size_label=fig_dict['em_label_size'],
                                                           font_size_title=fig_dict['em_title_font_size'])

        if (6718 in em_line_fit_dict['ln_list']) & (6733 in em_line_fit_dict['ln_list']):
            ax_sii_spec = fig.add_axes([
                fig_dict['sii_left_align'],
                fig_dict['sii_bottom_align'],
                fig_dict['sii_width'],
                fig_dict['sii_height'],
            ])
            ax_sii_res_spec = fig.add_axes([
                fig_dict['sii_res_left_align'],
                fig_dict['sii_res_bottom_align'],
                fig_dict['sii_res_width'],
                fig_dict['sii_res_height'],
            ])
            plotting_tools.SpecPlotTools.plot_em_line_spec(ax=ax_sii_spec, em_line_fit_dict=em_line_fit_dict,
                                                           line_list=[6718, 6733],
                                                           left_offset=fig_dict['sii_left_lim_offst'],
                                                           right_offset=fig_dict['sii_right_lim_offst'],
                                                           display_legend=False,
                                                           display_x_label=True,
                                                           display_y_label=False,
                                                           ax_res=ax_sii_res_spec,
                                                           font_size_label=fig_dict['em_label_size'],
                                                           font_size_title=fig_dict['em_title_font_size'])



        # ax_red_bump = fig.add_axes([
        #     fig_dict['red_bump_left_align'],
        #     fig_dict['red_bump_bottom_align'],
        #     fig_dict['red_bump_width'],
        #     fig_dict['red_bump_height'],
        # ])
        #
        # ax_hei6680_spec = fig.add_axes([
        #     fig_dict['hei6680_left_align'],
        #     fig_dict['hei6680_bottom_align'],
        #     fig_dict['hei6680_width'],
        #     fig_dict['hei6680_height'],
        # ])
        # plotting_tools.SpecPlotTools.plot_red_bump(ax=ax_red_bump, ppxf_fit_dict=ppxf_fit_dict,
        #                                            font_size_label=fig_dict['em_label_size'],
        #                                            font_size_title=fig_dict['em_title_font_size'])
        #
        # plotting_tools.SpecPlotTools.plot_stellar_hei6680(ax=ax_hei6680_spec, ppxf_fit_dict=ppxf_fit_dict,
        #                                                   y_axis_scale=1e16, font_size_label=fig_dict['em_label_size'],
        #                                                   display_label=True, font_size_title=fig_dict['em_title_font_size'],
        #                                                   display_y_label=False, display_x_label=True)

    def plot_ism_cutout_and_bkg(self, fig, fig_dict, ra, dec):


        miri_band_list = self.get_covered_miri_band_list(ra=ra, dec=dec)

        self.load_phangs_bands(band_list=miri_band_list, flux_unit='mJy', load_err=True, load_hst=True,
                                      load_hst_ha=True,
                                      load_nircam=True, load_miri=True, load_astrosat=False)

        # add sed axis
        ax_sed = fig.add_axes([
            fig_dict['sed_left_align'],
            fig_dict['sed_bottom_align'],
            fig_dict['sed_width'],
            fig_dict['sed_height'],
        ])

        # load cutout
        cutout_dict_bkg_env = self.get_band_cutout_dict(ra_cutout=ra, dec_cutout=dec,
                                                          cutout_size=fig_dict['bkg_cutout_size'],
                                                          band_list=miri_band_list, include_err=True)
        miri_count = 0
        for miri_count, band in enumerate(miri_band_list):
            ax_data = fig.add_axes([fig_dict['bkg_env_left_align'] + (fig_dict['bkg_env_width'] + fig_dict['bkg_env_space_vertical']) * (miri_count),
                                    fig_dict['bkg_env_bottom_align'] + (fig_dict['bkg_env_height'] + fig_dict['bkg_env_space_horizontal']) * 2,
                                    fig_dict['bkg_env_width'],
                                    fig_dict['bkg_env_height']
                                    ], projection=cutout_dict_bkg_env['%s_img_cutout' % band].wcs)
            ax_bkg_1 = fig.add_axes([fig_dict['bkg_env_left_align'] + (fig_dict['bkg_env_width'] + fig_dict['bkg_env_space_vertical']) * (miri_count),
                                    fig_dict['bkg_env_bottom_align'] + (fig_dict['bkg_env_height'] + fig_dict['bkg_env_space_horizontal']) * 1,
                                    fig_dict['bkg_env_width'],
                                    fig_dict['bkg_env_height']
                                    ], projection=cutout_dict_bkg_env['%s_img_cutout' % band].wcs)
            ax_bkg_2 = fig.add_axes([fig_dict['bkg_env_left_align'] + (fig_dict['bkg_env_width'] + fig_dict['bkg_env_space_vertical']) * (miri_count),
                                    fig_dict['bkg_env_bottom_align'] + (fig_dict['bkg_env_height'] + fig_dict['bkg_env_space_horizontal']) * 0,
                                    fig_dict['bkg_env_width'],
                                    fig_dict['bkg_env_height']
                                    ], projection=cutout_dict_bkg_env['%s_img_cutout' % band].wcs)

            bkg_1 = phot_tools.PhotTools.compute_2d_bkg(data=cutout_dict_bkg_env['%s_img_cutout' % band].data, box_size=fig_dict['bkg_1_box_size'])
            bkg_2 = phot_tools.PhotTools.compute_2d_bkg(data=cutout_dict_bkg_env['%s_img_cutout' % band].data, box_size=fig_dict['bkg_2_box_size'])

            vmin_norm = np.nanmin(cutout_dict_bkg_env['%s_img_cutout' % band].data)
            vmax_norm = np.nanmax(cutout_dict_bkg_env['%s_img_cutout' % band].data)
            bkg_norm = ImageNormalize(stretch=LogStretch(), vmin=vmin_norm, vmax=vmax_norm,)

            ax_data.imshow(cutout_dict_bkg_env['%s_img_cutout' % band].data, norm=bkg_norm, cmap='coolwarm',)
            ax_bkg_1.imshow(bkg_1.background, norm=bkg_norm, cmap='coolwarm')
            ax_bkg_2.imshow(bkg_2.background, norm=bkg_norm, cmap='coolwarm')
            plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_data, ra_tick_label=False,
                                                            dec_tick_label=False, ra_axis_label=' ', dec_axis_label=' ',
                                                            fontsize=fig_dict['stamp_label_size'],
                                                            labelsize=fig_dict['stamp_label_size'])
            plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_bkg_1, ra_tick_label=False,
                                                            dec_tick_label=False, ra_axis_label=' ', dec_axis_label=' ',
                                                            fontsize=fig_dict['stamp_label_size'],
                                                            labelsize=fig_dict['stamp_label_size'])
            plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_bkg_2, ra_tick_label=False,
                                                            dec_tick_label=False, ra_axis_label=' ', dec_axis_label=' ',
                                                            fontsize=fig_dict['stamp_label_size'],
                                                            labelsize=fig_dict['stamp_label_size'])


            plotting_tools.WCSPlottingTools.draw_box(ax=ax_data, wcs=cutout_dict_bkg_env['%s_img_cutout' % band].wcs,
                                                     coord=SkyCoord(ra=ra * u.deg, dec=dec * u.deg),
                                                     box_size=fig_dict['obj_cutout_size'], color='k', line_width=2, line_style='--')
            plotting_tools.WCSPlottingTools.draw_box(ax=ax_bkg_1, wcs=cutout_dict_bkg_env['%s_img_cutout' % band].wcs,
                                                     coord=SkyCoord(ra=ra * u.deg, dec=dec * u.deg),
                                                     box_size=fig_dict['obj_cutout_size'], color='k', line_width=2, line_style='--')
            plotting_tools.WCSPlottingTools.draw_box(ax=ax_bkg_2, wcs=cutout_dict_bkg_env['%s_img_cutout' % band].wcs,
                                                     coord=SkyCoord(ra=ra * u.deg, dec=dec * u.deg),
                                                     box_size=fig_dict['obj_cutout_size'], color='k', line_width=2, line_style='--')

            plotting_tools.StrTools.display_text_in_corner(ax=ax_data, text=band.upper(),
                                                           fontsize=fig_dict['zoom_in_title_font_size'],
                                                           text_color='k', x_frac=0.02, y_frac=0.98,
                                                           horizontal_alignment='left',
                                                           vertical_alignment='top',
                                                           path_eff=True, path_err_linewidth=3,
                                                           path_eff_color='k')
            plotting_tools.StrTools.display_text_in_corner(ax=ax_bkg_1, text='BKG1 %ix%i pix' % (fig_dict['bkg_1_box_size'][0], fig_dict['bkg_1_box_size'][1]),
                                                           fontsize=fig_dict['zoom_in_title_font_size'],
                                                           text_color='k', x_frac=0.02, y_frac=0.98,
                                                           horizontal_alignment='left',
                                                           vertical_alignment='top',
                                                           path_eff=True, path_err_linewidth=3,
                                                           path_eff_color='k')
            plotting_tools.StrTools.display_text_in_corner(ax=ax_bkg_2, text='BKG2 %ix%i pix' % (fig_dict['bkg_2_box_size'][0], fig_dict['bkg_2_box_size'][1]),
                                                           fontsize=fig_dict['zoom_in_title_font_size'],
                                                           text_color='k', x_frac=0.02, y_frac=0.98,
                                                           horizontal_alignment='left',
                                                           vertical_alignment='top',
                                                           path_eff=True, path_err_linewidth=3,
                                                           path_eff_color='k')

            plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                ax=ax_data, img_shape=cutout_dict_bkg_env['%s_img_cutout' % band].data.shape,
                wcs=cutout_dict_bkg_env['%s_img_cutout' % band].wcs,
                bar_length=fig_dict['zoom_in_scale_bar_length_1'], length_unit='pc',
                phangs_target=self.phot_target_name,
                bar_color='k', text_color='k',
                line_width=4, fontsize=fig_dict['zoom_in_label_size'],
                va='bottom', ha='left', x_offset=0.1, y_offset=0.05, text_y_offset_diff=0.01)

            plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                ax=ax_data, img_shape=cutout_dict_bkg_env['%s_img_cutout' % band].data.shape,
                wcs=cutout_dict_bkg_env['%s_img_cutout' % band].wcs,
                bar_length=fig_dict['zoom_in_scale_bar_length_2'], length_unit='arcsec',
                phangs_target=self.phot_target_name,
                bar_color='k', text_color='k',
                line_width=4, fontsize=fig_dict['zoom_in_label_size'],
                va='top', ha='right', x_offset=0.05, y_offset=0.15, text_y_offset_diff=0.01)

            # get smaller cutouts

            cutout_stamp_data = helper_func.CoordTools.get_img_cutout(
                img=cutout_dict_bkg_env['%s_img_cutout' % band].data,
                wcs=cutout_dict_bkg_env['%s_img_cutout' % band].wcs,
                coord=SkyCoord(ra=ra * u.deg, dec=dec * u.deg), cutout_size=fig_dict['obj_cutout_size'])
            cutout_stamp_err = helper_func.CoordTools.get_img_cutout(
                img=cutout_dict_bkg_env['%s_err_cutout' % band].data,
                wcs=cutout_dict_bkg_env['%s_err_cutout' % band].wcs,
                coord=SkyCoord(ra=ra * u.deg, dec=dec * u.deg), cutout_size=fig_dict['obj_cutout_size'])
            cutout_stamp_bkg_1 = helper_func.CoordTools.get_img_cutout(
                img=bkg_1.background,
                wcs=cutout_dict_bkg_env['%s_img_cutout' % band].wcs,
                coord=SkyCoord(ra=ra * u.deg, dec=dec * u.deg), cutout_size=fig_dict['obj_cutout_size'])
            cutout_stamp_bkg_2 = helper_func.CoordTools.get_img_cutout(
                img=bkg_2.background,
                wcs=cutout_dict_bkg_env['%s_img_cutout' % band].wcs,
                coord=SkyCoord(ra=ra * u.deg, dec=dec * u.deg), cutout_size=fig_dict['obj_cutout_size'])

            ax_stamp_data = fig.add_axes([fig_dict['bkg_env_left_align'] + (fig_dict['bkg_env_width'] + fig_dict['bkg_env_space_vertical']) * (miri_count),
                                    fig_dict['stamp_bottom_align'] + (fig_dict['stamp_height'] + fig_dict['stamp_space_horizontal']) * 0,
                                    fig_dict['stamp_width'], fig_dict['stamp_height']], projection=cutout_stamp_data.wcs)
            ax_stamp_rad_prof = fig.add_axes([fig_dict['bkg_env_left_align'] + (fig_dict['bkg_env_width'] + fig_dict['bkg_env_space_vertical']) * (miri_count) + (fig_dict['stamp_width'] + fig_dict['stamp_space_vertical']),
                                    fig_dict['rad_prof_bottom_align'] + (fig_dict['stamp_height'] + fig_dict['stamp_space_horizontal']) * 0,
                                    fig_dict['stamp_width'],
                                    fig_dict['rad_prof_height']])
            ax_stamp_bkg1 = fig.add_axes([fig_dict['bkg_env_left_align'] + (fig_dict['bkg_env_width'] + fig_dict['bkg_env_space_vertical']) * (miri_count),
                                    fig_dict['stamp_bottom_align'] - (fig_dict['stamp_height'] + fig_dict['stamp_space_horizontal']),
                                    fig_dict['stamp_width'], fig_dict['stamp_height']], projection=cutout_stamp_data.wcs)
            ax_stamp_bkg2 = fig.add_axes([fig_dict['bkg_env_left_align'] + (fig_dict['bkg_env_width'] + fig_dict['bkg_env_space_vertical']) * (miri_count) + (fig_dict['stamp_width'] + fig_dict['stamp_space_vertical']),
                                    fig_dict['stamp_bottom_align'] - (fig_dict['stamp_height'] + fig_dict['stamp_space_horizontal']),
                                    fig_dict['stamp_width'], fig_dict['stamp_height']], projection=cutout_stamp_data.wcs)

            vmin_norm = np.nanmin(cutout_stamp_data.data)
            vmax_norm = np.nanmax(cutout_stamp_data.data)
            bkg_norm = ImageNormalize(stretch=LogStretch(), vmin=vmin_norm, vmax=vmax_norm,)
            ax_stamp_data.imshow(cutout_stamp_data.data, norm=bkg_norm, cmap='coolwarm',)
            ax_stamp_bkg1.imshow(cutout_stamp_bkg_1.data, norm=bkg_norm, cmap='coolwarm',)
            ax_stamp_bkg2.imshow(cutout_stamp_bkg_2.data, norm=bkg_norm, cmap='coolwarm',)


            plotting_tools.StrTools.display_text_in_corner(ax=ax_stamp_bkg1, text='BKG1',
                                                           fontsize=fig_dict['stamp_label_size'],
                                                           text_color='k', x_frac=0.02, y_frac=0.98,
                                                           horizontal_alignment='left',
                                                           vertical_alignment='top',
                                                           path_eff=True, path_err_linewidth=3,
                                                           path_eff_color='k')
            plotting_tools.StrTools.display_text_in_corner(ax=ax_stamp_bkg2, text='BKG2',
                                                           fontsize=fig_dict['stamp_label_size'],
                                                           text_color='k', x_frac=0.02, y_frac=0.98,
                                                           horizontal_alignment='left',
                                                           vertical_alignment='top',
                                                           path_eff=True, path_err_linewidth=3,
                                                           path_eff_color='k')
            plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_stamp_data, ra_tick_label=False,
                                                            dec_tick_label=False, ra_axis_label=' ', dec_axis_label=' ',
                                                            fontsize=fig_dict['stamp_label_size'],
                                                            labelsize=fig_dict['stamp_label_size'])
            plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_stamp_bkg1, ra_tick_label=False,
                                                            dec_tick_label=False, ra_axis_label=' ', dec_axis_label=' ',
                                                            fontsize=fig_dict['stamp_label_size'],
                                                            labelsize=fig_dict['stamp_label_size'])
            plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_stamp_bkg2, ra_tick_label=False,
                                                            dec_tick_label=False, ra_axis_label=' ', dec_axis_label=' ',
                                                            fontsize=fig_dict['stamp_label_size'],
                                                            labelsize=fig_dict['stamp_label_size'])
            plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                ax=ax_stamp_data, img_shape=cutout_stamp_data.data.shape,
                wcs=cutout_stamp_data.wcs,
                bar_length=fig_dict['stamp_scale_bar_length_1'], length_unit='arcsec',
                bar_color='k', text_color='k',
                line_width=4, fontsize=fig_dict['stamp_label_size'],
                va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)

            plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_stamp_data, pos=SkyCoord(ra=ra * u.deg, dec=dec * u.deg), rad=phys_params.miri_encircle_apertures_arcsec[band]['ee50'], color='k', line_style='-')
            plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_stamp_bkg1, pos=SkyCoord(ra=ra * u.deg, dec=dec * u.deg), rad=phys_params.miri_encircle_apertures_arcsec[band]['ee50'], color='k', line_style='-')
            plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_stamp_bkg2, pos=SkyCoord(ra=ra * u.deg, dec=dec * u.deg), rad=phys_params.miri_encircle_apertures_arcsec[band]['ee50'], color='k', line_style='-')

            plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_stamp_data, pos=SkyCoord(ra=ra * u.deg, dec=dec * u.deg), rad=phys_params.miri_encircle_apertures_arcsec[band]['ee50']*2, color='k', line_style='--')
            plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_stamp_bkg1, pos=SkyCoord(ra=ra * u.deg, dec=dec * u.deg), rad=phys_params.miri_encircle_apertures_arcsec[band]['ee50']*2, color='k', line_style='--')
            plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_stamp_bkg2, pos=SkyCoord(ra=ra * u.deg, dec=dec * u.deg), rad=phys_params.miri_encircle_apertures_arcsec[band]['ee50']*2, color='k', line_style='--')
            plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_stamp_data, pos=SkyCoord(ra=ra * u.deg, dec=dec * u.deg), rad=phys_params.miri_encircle_apertures_arcsec[band]['ee50']*3, color='k', line_style='--')
            plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_stamp_bkg1, pos=SkyCoord(ra=ra * u.deg, dec=dec * u.deg), rad=phys_params.miri_encircle_apertures_arcsec[band]['ee50']*3, color='k', line_style='--')
            plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_stamp_bkg2, pos=SkyCoord(ra=ra * u.deg, dec=dec * u.deg), rad=phys_params.miri_encircle_apertures_arcsec[band]['ee50']*3, color='k', line_style='--')

            radius, profile, error = ProfileTools.get_rad_profile_from_img(
                img=cutout_stamp_data.data,
                wcs=cutout_stamp_data.wcs,
                ra=ra, dec=dec, max_rad_arcsec=1.0, img_err=cutout_stamp_err.data)
            ax_stamp_rad_prof.fill_between(radius, profile-error, profile+error, color='gray', alpha=0.7)
            ax_stamp_rad_prof.plot(radius, profile, linewidth=4, color='k')
            ax_stamp_rad_prof.set_yticklabels([])
            ax_stamp_rad_prof.tick_params(axis='both', which='both', width=2, direction='in', labelsize=fig_dict['stamp_label_size'])
            ax_stamp_rad_prof.set_xlabel('rad. [\"]', fontsize=fig_dict['stamp_label_size'])

            mean_wave = helper_func.ObsTools.get_jwst_band_wave(band=band, instrument='miri')
            min_wave = helper_func.ObsTools.get_jwst_band_wave(band=band, instrument='miri',
                                                                wave_estimator='min_wave')
            max_wave = helper_func.ObsTools.get_jwst_band_wave(band=band, instrument='miri',
                                                                wave_estimator='max_wave')
            bkg_in_aprt_1 = PhotTools.extract_bkg_from_circ_aperture(data=cutout_stamp_bkg_1.data,
                                                                     data_err=cutout_stamp_err.data,
                                                                     wcs=cutout_stamp_bkg_1.wcs, ra=ra, dec=dec,
                                                                     aperture_rad=phys_params.miri_encircle_apertures_arcsec[band]['ee50'])
            bkg_in_aprt_2 = PhotTools.extract_bkg_from_circ_aperture(data=cutout_stamp_bkg_2.data, data_err=cutout_stamp_err.data,
                                                                     wcs=cutout_stamp_bkg_2.wcs, ra=ra, dec=dec,
                                                                     aperture_rad=phys_params.miri_encircle_apertures_arcsec[band]['ee50'])
            bkg_from_annulus = PhotTools.get_bkg_from_annulus(data=cutout_stamp_data.data,
                                                              data_err=cutout_stamp_err.data,
                                                              wcs=cutout_stamp_data.wcs,
                                                              ra=ra, dec=dec,
                                                              annulus_rad_in=phys_params.miri_encircle_apertures_arcsec[band]['ee50']*2,
                                                              annulus_rad_out=phys_params.miri_encircle_apertures_arcsec[band]['ee50']*3,
                                                              do_sigma_clip=True, sigma=3.0, maxiters=5)
            flux_in_ee_rad, flux_in_ee_rad_err = PhotTools.extract_flux_from_circ_aperture(data=cutout_stamp_data.data - bkg_from_annulus.median,
                                                              data_err=cutout_stamp_err.data,
                                                              wcs=cutout_stamp_data.wcs,
                                                              ra=ra, dec=dec,
                                                              aperture_rad=phys_params.miri_encircle_apertures_arcsec[band]['ee50'])
            flux_in_ee_rad_bkg_1, flux_in_ee_rad_bkg_1_err = PhotTools.extract_flux_from_circ_aperture(data=cutout_stamp_data.data - cutout_stamp_bkg_1.data,
                                                              data_err=cutout_stamp_err.data,
                                                              wcs=cutout_stamp_data.wcs,
                                                              ra=ra, dec=dec,
                                                              aperture_rad=phys_params.miri_encircle_apertures_arcsec[band]['ee50'])
            flux_in_ee_rad_bkg_2, flux_in_ee_rad_bkg_2_err = PhotTools.extract_flux_from_circ_aperture(data=cutout_stamp_data.data - cutout_stamp_bkg_2.data,
                                                              data_err=cutout_stamp_err.data,
                                                              wcs=cutout_stamp_data.wcs,
                                                              ra=ra, dec=dec,
                                                              aperture_rad=phys_params.miri_encircle_apertures_arcsec[band]['ee50'])


            if miri_count == 0:
                label_aprt_1 = 'Median BKG1'
                label_aprt_2 = 'Median BKG2'
                label_aprt_annulus = 'Median BKG in Annulus'
                label_flux_ee = 'Flux in 50% aper.'
                label_flux_ee_bkg_1 = 'Flux in 50% aper. - BKG1'
                label_flux_ee_bkg_2 = 'Flux in 50% aper. - BKG2'
            else:
                label_aprt_1 = None
                label_aprt_2 = None
                label_aprt_annulus = None
                label_flux_ee = None
                label_flux_ee_bkg_1 = None
                label_flux_ee_bkg_2 = None

            ax_sed.plot([min_wave, max_wave], [bkg_in_aprt_1.median, bkg_in_aprt_1.median], linewidth=4, color='r', label=label_aprt_1)
            ax_sed.plot([min_wave, max_wave], [bkg_in_aprt_2.median, bkg_in_aprt_2.median], linewidth=4, color='blue', label=label_aprt_2)
            ax_sed.plot([min_wave, max_wave], [bkg_from_annulus.median, bkg_from_annulus.median], linewidth=4, color='k', label=label_aprt_annulus)

            ax_sed.errorbar(mean_wave, flux_in_ee_rad,
                            xerr=[[mean_wave - min_wave], [max_wave - mean_wave]],
                            yerr=flux_in_ee_rad_err,
                            fmt='v', ecolor='gray', color='k', ms=20, label=label_flux_ee)
            ax_sed.errorbar(mean_wave, flux_in_ee_rad_bkg_1,
                            xerr=[[mean_wave - min_wave], [max_wave - mean_wave]],
                            yerr=flux_in_ee_rad_bkg_1_err,
                            fmt='v', ecolor='gray', color='red', ms=20, label=label_flux_ee_bkg_1)
            ax_sed.errorbar(mean_wave, flux_in_ee_rad_bkg_2,
                            xerr=[[mean_wave - min_wave], [max_wave - mean_wave]],
                            yerr=flux_in_ee_rad_bkg_2_err,
                            fmt='v', ecolor='gray', color='blue', ms=20, label=label_flux_ee_bkg_2)


            for rad in fig_dict['miri_ap_rad_pix_list']:
                aperture_rad = helper_func.CoordTools.transform_pix2world_scale(
                    length_in_pix=rad, wcs=cutout_dict_bkg_env['%s_img_cutout' % band].wcs)
                annulus_rad_in = helper_func.CoordTools.transform_pix2world_scale(
                    length_in_pix=fig_dict['miri_annulus_rad_in_pix'],
                    wcs=cutout_dict_bkg_env['%s_img_cutout' % band].wcs)
                annulus_rad_out = helper_func.CoordTools.transform_pix2world_scale(
                    length_in_pix=fig_dict['miri_annulus_rad_out_pix'],
                    wcs=cutout_dict_bkg_env['%s_img_cutout' % band].wcs)
                flux_dict = phot_tools.PhotTools.compute_phot_jimena(ra=ra, dec=dec,
                                                                  data=cutout_dict_bkg_env['%s_img_cutout' % band].data,
                                                                  err=cutout_dict_bkg_env['%s_err_cutout' % band].data,
                                                                  wcs=cutout_dict_bkg_env['%s_img_cutout' % band].wcs,
                                                                  obs='miri', band=band, target=self.phot_miri_target_name,
                                                          aperture_rad=aperture_rad,
                                                          annulus_rad_in=annulus_rad_in,
                                                          annulus_rad_out=annulus_rad_out)
                ax_sed.scatter(mean_wave, flux_dict['flux'], color='gray')
        ax_sed.legend(frameon=False, bbox_to_anchor=(0.5, 0.7), bbox_transform=ax_sed.transAxes, fontsize=fig_dict['sed_label_size'])
        ax_sed.set_yscale('log')
        ax_sed.set_xscale('log')
        ax_sed.set_xlabel(r'Wavelength [$\mu$m]', fontsize=fig_dict['sed_label_size'])
        ax_sed.set_ylabel(r'Flux [mJy]', fontsize=fig_dict['sed_label_size'])
        ax_sed.tick_params(axis='both', which='both', width=2, direction='in', labelsize=fig_dict['sed_label_size'])

    def compute_complete_photometry(self, ra_list, dec_list, roi_arcsec, bkg_roi_rad_in_arcsec, bkg_roi_rad_out_arcsec,
                                    idx_list=None,
                                    detect_sub_src=True,
                                    max_n_sub_src=10,
                                    psf_substructure_ratio_lim=2,
                                    substructure_roi_frac=0.9,
                                    band_list=None,
                                    flux_unit='mJy',
                                    verbose_flag=True,
                                    plot_flag=True,
                                    plot_output_path='plot_output/',
                                    ):

        # check if the coordinates are a list or just a float
        if isinstance(ra_list, float):
            ra_list = [ra_list]
            dec_list = [dec_list]

        if idx_list is not None:
            if isinstance(idx_list, float) | isinstance(idx_list, int):
                idx_list = [idx_list]
        else:
            idx_list = np.arange(len(ra_list))

        # get entire band list to fit
        complete_hst_band_list = helper_func.ObsTools.get_hst_obs_band_list(target=self.phot_hst_target_name)
        complete_nircam_band_list = helper_func.ObsTools.get_nircam_obs_band_list(
            target=self.phot_nircam_target_name, version=self.nircam_data_ver)
        complete_miri_band_list = helper_func.ObsTools.get_miri_obs_band_list(target=self.phot_miri_target_name,
                                                                              version=self.miri_data_ver)

        # put together band list for the fitting
        if band_list is None:
            hst_band_list = complete_hst_band_list
            nircam_band_list = complete_nircam_band_list
            miri_band_list = complete_miri_band_list
            band_list = hst_band_list + nircam_band_list + miri_band_list
        else:
            hst_band_list = []
            nircam_band_list = []
            miri_band_list = []
            for band in band_list:
                if band in complete_hst_band_list:
                    hst_band_list.append(band)
                elif band in complete_nircam_band_list:
                    nircam_band_list.append(band)
                elif band in complete_miri_band_list:
                    miri_band_list.append(band)

        # check if the bands are loaded!
        self.load_phangs_bands(band_list=band_list, flux_unit=flux_unit, load_err=True)

        # get observation abd instrument list
        obs_list = []
        instrument_list = []
        target_name_list = []
        flux_table_names = []

        standard_phot_aperture_arcsec_list = []
        phot_aperture_pix_list = []
        phot_aperture_arcsec_list = []
        bkg_rad_in_arcsec_list = []
        bkg_rad_out_arcsec_list = []
        search_substructure_flag_list = []
        fwhm_pix_list = []

        sub_structure_colname_list = []

        for band in band_list:
            flux_table_names.append('%s_flux' % band)
            flux_table_names.append('%s_flux_err' % band)
            flux_table_names.append('%s_peak' % band)
            flux_table_names.append('%s_flux_flag' % band)
            flux_table_names.append('%s_median_bkg' % band)
            flux_table_names.append('%s_mean_bkg' % band)
            flux_table_names.append('%s_bkg_flag' % band)


        for band in band_list:
            # get observations
            if band in hst_band_list:
                obs_list.append('hst')
                instrument_list.append(
                    helper_func.ObsTools.get_hst_instrument(target=self.phot_hst_target_name, band=band))
                target_name_list.append(self.phot_hst_target_name)
            elif band in nircam_band_list:
                obs_list.append('nircam')
                instrument_list.append('nircam')
                target_name_list.append(self.phot_nircam_target_name)
            elif band in miri_band_list:
                obs_list.append('miri')
                instrument_list.append('miri')
                target_name_list.append(self.phot_miri_target_name)
            else:
                raise RuntimeError(band, ' is in no observation present.')

            # get psf scales for band
            psf_dict = phot_tools.PSFTools.load_obs_psf_dict(band=band, instrument=instrument_list[-1])
            fwhm_arcsec = psf_dict['gaussian_fwhm']

            # get the standard aperture radii of the current band
            wcs = getattr(self, '%s_bands_data' % obs_list[-1])['%s_wcs_img' % band]
            fwhm_pix = helper_func.CoordTools.transform_world2pix_scale(length_in_arcsec=fwhm_arcsec, wcs=wcs).value
            # estimate the standard_aperture_radius
            standard_apert_rad_arcsec = phot_tools.ApertTools.get_standard_ap_rad_arcsec(obs=obs_list[-1],
                                                                                         band=band, wcs=wcs)

            # now check if the standard aperture is smaller than the radius of interest
            if standard_apert_rad_arcsec < roi_arcsec:
                # in this case we will simply compute the photometry inside this larger aperture
                phot_aperture_arcsec = roi_arcsec
                rad_in_arcsec, rad_out_arcsec = bkg_roi_rad_in_arcsec, bkg_roi_rad_out_arcsec
            else:
                phot_aperture_arcsec = standard_apert_rad_arcsec
                rad_in_arcsec, rad_out_arcsec = phot_tools.ApertTools.get_standard_bkg_annulus_rad_arcsec(
                    obs=obs_list[-1], band=band, wcs=wcs)

            phot_aperture_pix = helper_func.CoordTools.transform_world2pix_scale(
                length_in_arcsec=phot_aperture_arcsec, wcs=wcs).value

            # check if sub sources need to be considered

            if roi_arcsec / (fwhm_arcsec / 2) > psf_substructure_ratio_lim:
                search_substructure_flag = True
                # add sub structure names
                for sub_structure_idx in range(max_n_sub_src):

                    sub_structure_colname_list.append('%s_sub_struct_ra_%i' % (band, sub_structure_idx))
                    sub_structure_colname_list.append('%s_sub_struct_dec_%i' % (band, sub_structure_idx))
                    sub_structure_colname_list.append('%s_sub_struct_flux_%i' % (band, sub_structure_idx))
                    sub_structure_colname_list.append('%s_sub_struct_flux_err_%i' % (band, sub_structure_idx))
                    sub_structure_colname_list.append('%s_sub_struct_peak_%i' % (band, sub_structure_idx))
                    sub_structure_colname_list.append('%s_sub_struct_median_bkg_%i' % (band, sub_structure_idx))

            else:
                search_substructure_flag = False
                # scale_data, scale_header, scale_wcs = None, None, None

            standard_phot_aperture_arcsec_list.append(standard_apert_rad_arcsec)
            phot_aperture_pix_list.append(phot_aperture_pix)
            phot_aperture_arcsec_list.append(phot_aperture_arcsec)
            bkg_rad_in_arcsec_list.append(rad_in_arcsec)
            bkg_rad_out_arcsec_list.append(rad_out_arcsec)
            search_substructure_flag_list.append(search_substructure_flag)
            fwhm_pix_list.append(fwhm_pix)

        if verbose_flag: print('roi_arcsec ', roi_arcsec)
        #################################
        # now loop over all coordinates #
        #################################

        # construct a table

        r_rows = len(idx_list)
        n_cols = len(band_list) * 7

        flux_table = Table(np.zeros((r_rows, n_cols)), names=flux_table_names)

        sub_structure_data_table = Table(np.zeros((r_rows, len(sub_structure_colname_list))), names=sub_structure_colname_list)


        for running_idx, obj_idx, ra, dec in zip(range(len(idx_list)), idx_list, ra_list, dec_list):
            if verbose_flag: print(running_idx, obj_idx, ra, dec)

            if plot_flag:
                fontsize_small_label = 30
                fontsize_large_label = 40

                fig_size_individual = (5, 5)
                n_cols = 5
                n_rows = int(np.ceil((len(band_list)) / n_cols) + 2)


                fig = plt.figure(figsize=(fig_size_individual[0] * n_cols, fig_size_individual[1] * n_rows))
                gs = fig.add_gridspec(ncols=n_cols, nrows=n_rows, left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
                gs_sed = fig.add_gridspec(ncols=n_cols, nrows=n_rows, left=0.08, bottom=0.01, right=0.99, top=0.99, wspace=0.5, hspace=0.5)
                ax_sed = fig.add_subplot(gs_sed[:2, :])

            ###########################
            # now loop over all bands #
            ###########################
            idx_row = 2
            idx_col = 0
            for band_idx, band in enumerate(band_list):

                if not band in band_list:
                    continue

                if verbose_flag: print(band)
                # get the cutout_size
                img_cutout_size = (3 * bkg_rad_out_arcsec_list[band_idx], 3 * bkg_rad_out_arcsec_list[band_idx])
                # print(band, fwhm_arcsec, phot_aperture_arcsec, rad_in_arcsec, bkg_rad_out_arcsec_list[band_idx, img_cutout_size)

                obs_cutout_dict = self.get_band_cutout_dict(ra_cutout=ra, dec_cutout=dec, cutout_size=img_cutout_size,
                                                            include_err=True, band_list=[band])
                if obs_cutout_dict['%s_img_cutout' % band].data is None:
                    continue

                forced_photomerty_dict = phot_tools.ApertTools.compute_apert_photometry(
                    apert_rad_arcsec=phot_aperture_arcsec_list[band_idx], bkg_rad_annulus_in_arcsec=bkg_rad_in_arcsec_list[band_idx],
                    bkg_rad_annulus_out_arcsec=bkg_rad_out_arcsec_list[band_idx], data=obs_cutout_dict['%s_img_cutout' % band].data,
                    data_err=obs_cutout_dict['%s_err_cutout' % band].data,
                    wcs=obs_cutout_dict['%s_img_cutout' % band].wcs,
                    ra=ra, dec=dec, mask=None, sigma_clip_sig=3, sigma_clip_maxiters=10, sum_method='exact')


                flux_table['%s_flux' % band][running_idx] = forced_photomerty_dict['src_flux']
                flux_table['%s_flux_err' % band][running_idx] = forced_photomerty_dict['src_flux_err']
                flux_table['%s_peak' % band][running_idx] = forced_photomerty_dict['apert_max']
                flux_table['%s_flux_flag' % band][running_idx] = forced_photomerty_dict['flux_measure_ok_flag']
                flux_table['%s_median_bkg' % band][running_idx] = forced_photomerty_dict['bkg_median']
                flux_table['%s_mean_bkg' % band][running_idx] = forced_photomerty_dict['bkg_mean']
                flux_table['%s_bkg_flag' % band][running_idx] = forced_photomerty_dict['bkg_ok_flag']

                if plot_flag:

                    # plot img
                    ax_img = fig.add_subplot(gs[idx_row, idx_col],
                                             projection=obs_cutout_dict['%s_img_cutout' % band].wcs)
                    idx_col += 1
                    if idx_col >= n_cols:
                        idx_col = 0
                        idx_row += 1

                    mean, median, std = sigma_clipped_stats(obs_cutout_dict['%s_img_cutout' % band].data, sigma=3.0)
                    vmin = median - 1 * std
                    vmax = median + 30 * std

                    norm_img = ImageNormalize(stretch=LogStretch(), vmin=vmin, vmax=vmax)
                    ax_img.imshow(obs_cutout_dict['%s_img_cutout' % band].data, norm=norm_img, cmap='Greys')
                    ax_img.axis('off')
                    plotting_tools.StrTools.display_text_in_corner(ax=ax_img, text=band.upper(), fontsize=fontsize_large_label, text_color='tab:orange', x_frac=0.02, y_frac=0.98, horizontal_alignment='left',
                               vertical_alignment='top', path_eff=True, path_err_linewidth=3, path_eff_color='white', rotation=0)

                    plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_img, pos=SkyCoord(ra=ra*u.deg, dec=dec*u.deg), rad=phot_aperture_arcsec_list[band_idx], color='tab:red', line_style='-', line_width=3, alpha=1., fill=False)
                    plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_img, pos=SkyCoord(ra=ra*u.deg, dec=dec*u.deg), rad=bkg_rad_in_arcsec_list[band_idx], color='tab:blue', line_style='--', line_width=3, alpha=1., fill=False)
                    plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_img, pos=SkyCoord(ra=ra*u.deg, dec=dec*u.deg), rad=bkg_rad_out_arcsec_list[band_idx], color='tab:blue', line_style='--', line_width=3, alpha=1., fill=False)


                    plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                        ax=ax_img, img_shape=obs_cutout_dict['%s_img_cutout' % band].data.shape, wcs=obs_cutout_dict['%s_img_cutout' % band].wcs,
                        bar_length=1, length_unit='arcsec', bar_color='red', text_color='red', line_width=4, fontsize=fontsize_small_label,
                           va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01,
                           path_eff=True, path_eff_color='white')


                    # plot sed point
                    mean_band_wavelength = helper_func.ObsTools.get_phangs_telescope_wave(
                        target=target_name_list[band_idx], band=band, telescope=obs_list[band_idx],
                        wave_estimator='mean_wave', unit='mu')
                    min_band_wavelength = helper_func.ObsTools.get_phangs_telescope_wave(
                        target=target_name_list[band_idx], band=band, telescope=obs_list[band_idx],
                        wave_estimator='min_wave', unit='mu')
                    max_band_wavelength = helper_func.ObsTools.get_phangs_telescope_wave(
                        target=target_name_list[band_idx], band=band, telescope=obs_list[band_idx],
                        wave_estimator='max_wave', unit='mu')

                    ax_sed.errorbar(mean_band_wavelength, forced_photomerty_dict['src_flux'],
                                    xerr=[[mean_band_wavelength - min_band_wavelength],
                                          [max_band_wavelength - mean_band_wavelength]],
                                    yerr=forced_photomerty_dict['src_flux_err'],
                                    fmt='.', color='k', ms=30)

                if search_substructure_flag_list[band_idx] & detect_sub_src:
                    # get cutout
                    # scale_map_cutout = helper_func.CoordTools.get_img_cutout(
                    #     img=scale_data, wcs=scale_wcs, coord=SkyCoord(ra=ra*u.deg, dec=dec*u.deg),
                    #     cutout_size=img_cutout_size)
                    # mean_sigclip, median_sigclip, std_sigclip = sigma_clipped_stats(obs_cutout_dict['%s_img_cutout' % band].data)
                    # source detection
                    sub_src_detect = phot_tools.SrcTools.detect_star_like_src(
                        data=obs_cutout_dict['%s_img_cutout' % band].data, detection_threshold=forced_photomerty_dict['bkg_median'],
                        src_fwhm_pix=fwhm_pix_list[band_idx], min_separation=fwhm_pix_list[band_idx], roundhi=1, roundlo=-1, sharphi=1.0, sharplo=0.2)
                    if sub_src_detect is None: continue

                    # central_coordx, central_coordy = obs_cutout_dict['%s_img_cutout' % band].data.shape[0] / 2, obs_cutout_dict['%s_img_cutout' % band].data.shape[1] / 2
                    central_coordx, central_coordy = obs_cutout_dict['%s_img_cutout' % band].wcs.world_to_pixel(SkyCoord(ra=ra*u.deg, dec=dec*u.deg))
                    # sort sources by distance
                    dist2center = np.sqrt((sub_src_detect['xcentroid'] - central_coordx) ** 2 + (sub_src_detect['ycentroid'] - central_coordy) ** 2)
                    sort = np.argsort(dist2center)
                    sub_src_detect = sub_src_detect[sort]
                    mask_src_in_apert = np.sqrt((sub_src_detect['xcentroid'] - central_coordx) ** 2 + (sub_src_detect['ycentroid'] - central_coordy) ** 2) < (substructure_roi_frac * phot_aperture_pix_list[band_idx])


                    if sum(mask_src_in_apert) > 0:

                        coords_sub_src = obs_cutout_dict['%s_img_cutout' % band].wcs.pixel_to_world(sub_src_detect['xcentroid'], sub_src_detect['ycentroid'])
                        ra_sub_src_list = coords_sub_src.ra.degree
                        dec_sub_src_list = coords_sub_src.dec.degree

                        native_rad_in_arcsec, native_rad_out_arcsec = phot_tools.ApertTools.get_standard_bkg_annulus_rad_arcsec(
                            obs=obs_list[band_idx], band=band, wcs=wcs)

                        for sub_structure_idx, ra_sub_src, dec_sub_src, src_x, src_y in zip(
                                range(len(ra_sub_src_list[mask_src_in_apert])), ra_sub_src_list[mask_src_in_apert],
                                dec_sub_src_list[mask_src_in_apert], sub_src_detect['xcentroid'][mask_src_in_apert],
                                sub_src_detect['ycentroid'][mask_src_in_apert]):

                            if sub_structure_idx >= (max_n_sub_src - 1):
                                continue

                            sub_phot_dict = phot_tools.ApertTools.compute_apert_photometry(
                                apert_rad_arcsec=standard_phot_aperture_arcsec_list[band_idx],
                                bkg_rad_annulus_in_arcsec=native_rad_in_arcsec,
                                bkg_rad_annulus_out_arcsec=native_rad_out_arcsec,
                                data=obs_cutout_dict['%s_img_cutout' % band].data,
                                data_err=obs_cutout_dict['%s_err_cutout' % band].data,
                                wcs=obs_cutout_dict['%s_img_cutout' % band].wcs,
                                ra=ra_sub_src, dec=dec_sub_src, mask=None, sigma_clip_sig=3, sigma_clip_maxiters=5, sum_method='exact')

                            sub_structure_data_table['%s_sub_struct_ra_%i' % (band, sub_structure_idx)][running_idx] = ra_sub_src
                            sub_structure_data_table['%s_sub_struct_dec_%i' % (band, sub_structure_idx)][running_idx] = dec_sub_src

                            sub_structure_data_table['%s_sub_struct_flux_%i' % (band, sub_structure_idx)][running_idx] = \
                            sub_phot_dict['src_flux']
                            sub_structure_data_table['%s_sub_struct_flux_err_%i' % (band, sub_structure_idx)][
                                running_idx] = sub_phot_dict['src_flux_err']

                            sub_structure_data_table['%s_sub_struct_peak_%i' % (band, sub_structure_idx)][running_idx] = \
                            sub_phot_dict['apert_max']
                            sub_structure_data_table['%s_sub_struct_median_bkg_%i' % (band, sub_structure_idx)][
                                running_idx] = sub_phot_dict['bkg_median']

                            if plot_flag:
                                ax_img.scatter(src_x, src_y, color='r', s=100)

                                ax_sed.errorbar(mean_band_wavelength, sub_phot_dict['src_flux'],
                                                xerr=[[mean_band_wavelength - min_band_wavelength],
                                                      [max_band_wavelength - mean_band_wavelength]],
                                                yerr=sub_phot_dict['src_flux_err'],
                                                fmt='.', color='tab:blue', ms=20)



            if plot_flag:
                ax_sed.set_xscale('log')
                ax_sed.set_yscale('log')
                ax_sed.set_xlabel(r'Wavelength [$\mu$m]', fontsize=fontsize_large_label, labelpad=-10)
                ax_sed.set_ylabel(r'flux [mJy]', fontsize=fontsize_large_label)
                ax_sed.tick_params(which='both', width=3, length=5, direction='in', color='k',
                               labelsize=fontsize_large_label)

                # plt.subplots_adjust(left=0.05, bottom=0.01, right=0.99, top=0.99, wspace=0.01, hspace=0.05)
                fig.savefig(plot_output_path + 'complete_src_photmetry_%s_%i.png' % (self.phot_target_name, obj_idx))
                plt.close(fig)


        # remove empty columns
        for band_idx, band in enumerate(band_list):
            if not search_substructure_flag_list[band_idx]:
                continue
            for sub_structure_idx in range(max_n_sub_src):
                if np.all(sub_structure_data_table['%s_sub_struct_ra_%i' % (band, sub_structure_idx)] == 0):
                    sub_structure_data_table.remove_column('%s_sub_struct_ra_%i' % (band, sub_structure_idx))
                    sub_structure_data_table.remove_column('%s_sub_struct_dec_%i' % (band, sub_structure_idx))
                    sub_structure_data_table.remove_column('%s_sub_struct_flux_%i' % (band, sub_structure_idx))
                    sub_structure_data_table.remove_column('%s_sub_struct_flux_err_%i' % (band, sub_structure_idx))
                    sub_structure_data_table.remove_column('%s_sub_struct_peak_%i' % (band, sub_structure_idx))
                    sub_structure_data_table.remove_column('%s_sub_struct_median_bkg_%i' % (band, sub_structure_idx))

        return flux_table, sub_structure_data_table


    def compute_apert_corr_photometry(self, ra_list, dec_list,
                                    idx_list=None,
                                    band_list=None, flux_unit='mJy',
                                    verbose_flag=True,
                                    plot_flag=True,
                                    plot_output_path='plot_output/',
                                    ):

        # check if the coordinates are a list or just a float
        if isinstance(ra_list, float):
            ra_list = [ra_list]
            dec_list = [dec_list]

        if idx_list is not None:
            if isinstance(idx_list, float) | isinstance(idx_list, int):
                idx_list = [idx_list]
        else:
            idx_list = np.arange(len(ra_list))

        # get entire band list to fit
        complete_hst_band_list = helper_func.ObsTools.get_hst_obs_band_list(target=self.phot_hst_target_name)
        complete_nircam_band_list = helper_func.ObsTools.get_nircam_obs_band_list(target=self.phot_nircam_target_name,
                                                                                  version=self.nircam_data_ver)
        complete_miri_band_list = helper_func.ObsTools.get_miri_obs_band_list(target=self.phot_miri_target_name,
                                                                              version=self.miri_data_ver)

        # put together band list for the fitting
        if band_list is None:
            hst_band_list = complete_hst_band_list
            nircam_band_list = complete_nircam_band_list
            miri_band_list = complete_miri_band_list
            band_list = hst_band_list + nircam_band_list + miri_band_list
        else:
            hst_band_list = []
            nircam_band_list = []
            miri_band_list = []
            for band in band_list:
                if band in complete_hst_band_list:
                    hst_band_list.append(band)
                elif band in complete_nircam_band_list:
                    nircam_band_list.append(band)
                elif band in complete_miri_band_list:
                    miri_band_list.append(band)

        # check if the bands are loaded!
        self.load_phangs_bands(band_list=band_list, flux_unit=flux_unit, load_err=True)

        # get observation abd instrument list
        obs_list = []
        instrument_list = []
        target_name_list = []
        flux_table_names = []

        phot_aperture_pix_list = []
        phot_aperture_arcsec_list = []
        bkg_rad_in_arcsec_list = []
        bkg_rad_out_arcsec_list = []
        apert_corr_factor_list = []
        foreground_ext_mag_list = []

        for band in band_list:
            flux_table_names.append('%s_flux' % band)
            flux_table_names.append('%s_flux_err' % band)
            flux_table_names.append('%s_peak' % band)
            flux_table_names.append('%s_flux_flag' % band)
            flux_table_names.append('%s_median_bkg' % band)
            flux_table_names.append('%s_mean_bkg' % band)
            flux_table_names.append('%s_bkg_flag' % band)
            # get observations
            if band in hst_band_list:
                obs_list.append('hst')
                instrument_list.append(
                    helper_func.ObsTools.get_hst_instrument(target=self.phot_hst_target_name, band=band))
                target_name_list.append(self.phot_hst_target_name)
            elif band in nircam_band_list:
                obs_list.append('nircam')
                instrument_list.append('nircam')
                target_name_list.append(self.phot_nircam_target_name)
            elif band in miri_band_list:
                obs_list.append('miri')
                instrument_list.append('miri')
                target_name_list.append(self.phot_miri_target_name)
            else:
                raise RuntimeError(band, ' is in no observation present.')

            # get psf scales for band
            psf_dict = phot_tools.PSFTools.load_obs_psf_dict(band=band, instrument=instrument_list[-1])
            fwhm_arcsec = psf_dict['gaussian_fwhm']

            # get the standard aperture radii of the current band
            wcs = getattr(self, '%s_bands_data' % obs_list[-1])['%s_wcs_img' % band]

            # get standard photometry aperture
            phot_aperture_arcsec = phot_tools.ApertTools.get_standard_ap_rad_arcsec(obs=obs_list[-1],
                                                                                    band=band, wcs=wcs)
            # get aperture corr factor
            phot_apert_corr_fact = phot_tools.ApertTools.get_standard_ap_corr_fact(
                obs=obs_list[-1], band=band, target=target_name_list[-1])

            foreground_ext_mag = extinction_tools.ExtinctionTools.get_target_gal_ext_band(target=helper_func.FileTools.target_name_no_directions(target=target_name_list[-1]), obs=obs_list[-1], band=band)

            rad_in_arcsec, rad_out_arcsec = phot_tools.ApertTools.get_standard_bkg_annulus_rad_arcsec(
                    obs=obs_list[-1], band=band, wcs=wcs)

            phot_aperture_pix = helper_func.CoordTools.transform_world2pix_scale(
                length_in_arcsec=phot_aperture_arcsec, wcs=wcs).value


            phot_aperture_pix_list.append(phot_aperture_pix)
            phot_aperture_arcsec_list.append(phot_aperture_arcsec)
            bkg_rad_in_arcsec_list.append(rad_in_arcsec)
            bkg_rad_out_arcsec_list.append(rad_out_arcsec)

            apert_corr_factor_list.append(phot_apert_corr_fact)
            foreground_ext_mag_list.append(foreground_ext_mag)

        #################################
        # now loop over all coordinates #
        #################################

        # construct a table

        r_rows = len(idx_list)
        n_cols = len(band_list) * 7

        flux_table = Table(np.zeros((r_rows, n_cols)), names=flux_table_names)

        for running_idx, obj_idx, ra, dec in zip(range(len(idx_list)), idx_list, ra_list, dec_list):
            if verbose_flag: print(running_idx, obj_idx, ra, dec)

            if plot_flag:
                fontsize_small_label = 30
                fontsize_large_label = 40

                fig_size_individual = (5, 5)
                n_cols = 5
                n_rows = int(np.ceil((len(band_list)) / n_cols) + 2)


                fig = plt.figure(figsize=(fig_size_individual[0] * n_cols, fig_size_individual[1] * n_rows))
                gs = fig.add_gridspec(ncols=n_cols, nrows=n_rows, left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
                gs_sed = fig.add_gridspec(ncols=n_cols, nrows=n_rows, left=0.08, bottom=0.01, right=0.99, top=0.99, wspace=0.5, hspace=0.5)
                ax_sed = fig.add_subplot(gs_sed[:2, :])

            ###########################
            # now loop over all bands #
            ###########################
            idx_row = 2
            idx_col = 0
            for band_idx, band in enumerate(band_list):

                if verbose_flag: print(band)
                # get the cutout_size
                img_cutout_size = (3 * bkg_rad_out_arcsec_list[band_idx], 3 * bkg_rad_out_arcsec_list[band_idx])
                # print(band, fwhm_arcsec, phot_aperture_arcsec, rad_in_arcsec, bkg_rad_out_arcsec_list[band_idx, img_cutout_size)

                obs_cutout_dict = self.get_band_cutout_dict(ra_cutout=ra, dec_cutout=dec, cutout_size=img_cutout_size,
                                                            include_err=True, band_list=[band])
                if obs_cutout_dict['%s_img_cutout' % band].data is None:
                    continue

                apert_photomerty_dict = phot_tools.ApertTools.compute_apert_photometry(
                    apert_rad_arcsec=phot_aperture_arcsec_list[band_idx], bkg_rad_annulus_in_arcsec=bkg_rad_in_arcsec_list[band_idx],
                    bkg_rad_annulus_out_arcsec=bkg_rad_out_arcsec_list[band_idx], data=obs_cutout_dict['%s_img_cutout' % band].data,
                    data_err=obs_cutout_dict['%s_err_cutout' % band].data,
                    wcs=obs_cutout_dict['%s_img_cutout' % band].wcs,
                    ra=ra, dec=dec, mask=None, sigma_clip_sig=3, sigma_clip_maxiters=10, sum_method='exact')


                # perform aperture and extinction correction


                print('src_flux ', apert_photomerty_dict['src_flux'])
                print('src_flux_err ', apert_photomerty_dict['src_flux_err'])
                print('apert_corr_factor_list[band_idx] ', foreground_ext_mag_list[band_idx])
                print('foreground_ext_mag_list[band_idx] ', foreground_ext_mag_list[band_idx], 10 ** (-foreground_ext_mag_list[band_idx] / -2.5))
                flux = apert_photomerty_dict['src_flux'] * apert_corr_factor_list[band_idx]
                flux_err = apert_photomerty_dict['src_flux_err'] * apert_corr_factor_list[band_idx]
                flux *= 10 ** (-foreground_ext_mag_list[band_idx] / -2.5)
                flux_err *= 10 ** (-foreground_ext_mag_list[band_idx] / -2.5)



                flux_table['%s_flux' % band][running_idx] = apert_photomerty_dict['src_flux']
                flux_table['%s_flux_err' % band][running_idx] = apert_photomerty_dict['src_flux_err']

                flux_table['%s_peak' % band][running_idx] = apert_photomerty_dict['apert_max']
                flux_table['%s_flux_flag' % band][running_idx] = apert_photomerty_dict['flux_measure_ok_flag']
                flux_table['%s_median_bkg' % band][running_idx] = apert_photomerty_dict['bkg_median']
                flux_table['%s_mean_bkg' % band][running_idx] = apert_photomerty_dict['bkg_mean']
                flux_table['%s_bkg_flag' % band][running_idx] = apert_photomerty_dict['bkg_ok_flag']

                if plot_flag:

                    # plot img
                    ax_img = fig.add_subplot(gs[idx_row, idx_col],
                                             projection=obs_cutout_dict['%s_img_cutout' % band].wcs)
                    idx_col += 1
                    if idx_col >= n_cols:
                        idx_col = 0
                        idx_row += 1

                    mean, median, std = sigma_clipped_stats(obs_cutout_dict['%s_img_cutout' % band].data, sigma=3.0)
                    vmin = median - 1 * std
                    vmax = median + 30 * std

                    norm_img = ImageNormalize(stretch=LogStretch(), vmin=vmin, vmax=vmax)
                    ax_img.imshow(obs_cutout_dict['%s_img_cutout' % band].data, norm=norm_img, cmap='Greys')
                    ax_img.axis('off')
                    plotting_tools.StrTools.display_text_in_corner(ax=ax_img, text=band.upper(), fontsize=fontsize_large_label, text_color='tab:orange', x_frac=0.02, y_frac=0.98, horizontal_alignment='left',
                               vertical_alignment='top', path_eff=True, path_err_linewidth=3, path_eff_color='white', rotation=0)

                    plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_img, pos=SkyCoord(ra=ra*u.deg, dec=dec*u.deg), rad=phot_aperture_arcsec_list[band_idx], color='tab:red', line_style='-', line_width=3, alpha=1., fill=False)
                    plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_img, pos=SkyCoord(ra=ra*u.deg, dec=dec*u.deg), rad=bkg_rad_in_arcsec_list[band_idx], color='tab:blue', line_style='--', line_width=3, alpha=1., fill=False)
                    plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_img, pos=SkyCoord(ra=ra*u.deg, dec=dec*u.deg), rad=bkg_rad_out_arcsec_list[band_idx], color='tab:blue', line_style='--', line_width=3, alpha=1., fill=False)

                    plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                        ax=ax_img, img_shape=obs_cutout_dict['%s_img_cutout' % band].data.shape, wcs=obs_cutout_dict['%s_img_cutout' % band].wcs,
                        bar_length=1, length_unit='arcsec', bar_color='red', text_color='red', line_width=4, fontsize=fontsize_small_label,
                           va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01,
                           path_eff=True, path_eff_color='white')

                    # plot sed point
                    mean_band_wavelength = helper_func.ObsTools.get_phangs_telescope_wave(
                        target=target_name_list[band_idx], band=band, telescope=obs_list[band_idx],
                        wave_estimator='mean_wave', unit='mu')
                    min_band_wavelength = helper_func.ObsTools.get_phangs_telescope_wave(
                        target=target_name_list[band_idx], band=band, telescope=obs_list[band_idx],
                        wave_estimator='min_wave', unit='mu')
                    max_band_wavelength = helper_func.ObsTools.get_phangs_telescope_wave(
                        target=target_name_list[band_idx], band=band, telescope=obs_list[band_idx],
                        wave_estimator='max_wave', unit='mu')

                    ax_sed.errorbar(mean_band_wavelength, apert_photomerty_dict['src_flux'],
                                    xerr=[[mean_band_wavelength - min_band_wavelength],
                                          [max_band_wavelength - mean_band_wavelength]],
                                    yerr=apert_photomerty_dict['src_flux_err'],
                                    fmt='.', color='k', ms=30)


            if plot_flag:
                ax_sed.set_xscale('log')
                ax_sed.set_yscale('log')
                ax_sed.set_xlabel(r'Wavelength [$\mu$m]', fontsize=fontsize_large_label, labelpad=-10)
                ax_sed.set_ylabel(r'flux [mJy]', fontsize=fontsize_large_label)
                ax_sed.tick_params(which='both', width=3, length=5, direction='in', color='k',
                               labelsize=fontsize_large_label)

                # plt.subplots_adjust(left=0.05, bottom=0.01, right=0.99, top=0.99, wspace=0.01, hspace=0.05)
                fig.savefig(plot_output_path + 'complete_src_photmetry_%s_%i.png' % (self.phot_target_name, obj_idx))
                plt.close(fig)


        return flux_table
