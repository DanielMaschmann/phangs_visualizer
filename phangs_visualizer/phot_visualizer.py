"""
Tool to visualize PHANGS imaging data
"""


import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground

import multicolorfits as mcf
from matplotlib.colors import Normalize, LogNorm

import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.legend_handler import HandlerTuple

from phangs_data_access.phot_access import PhotAccess
from phangs_data_access import helper_func, phangs_info, phys_params
from phangs_data_access.cluster_cat_access import ClusterCatAccess
from phangs_data_access.gas_access import GasAccess

from phangs_visualizer import plotting_tools


class PhotVisualizer(PhotAccess):
    """
    Class to plot cutouts in multiple bands
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot_cluster_overview_photo_sed_co_ccd(self, phangs_cluster_id,
                                               hst_cluster_cat_ref_dict,
                                               include_h_alpha=True,
                                               include_nircam=True,
                                               include_miri=True,
                                               env_cutout_size=(10, 10), circle_rad_region=1.25,
                                               stamp_cutout_size=(2.5, 2.5), cbar_log_scale=True,
                                               fontsize_large=30, fontsize_small=20,
                                               ):

        # load photometry
        band_list = []
        # load HST bands
        band_list += helper_func.BandTools.get_hst_obs_band_list(target=self.target_name)
        hst_stamp_band_list = band_list.copy()
        # get BVI filters
        # specify color the bands
        hst_bvi_band_red = 'F814W'
        hst_bvi_band_green = 'F555W'
        if 'F438W' in band_list:
            hst_bvi_band_blue = 'F438W'
        else:
            hst_bvi_band_blue = 'F435W'

        # load ha native and continuum subtracted
        if include_h_alpha:
            # check if H-alpha observations are available
            if helper_func.BandTools.check_hst_ha_obs(target=self.target_ha_name):
                # get H-alpha observations
                hst_habu_band_red = helper_func.BandTools.get_hst_ha_band(target=self.target_ha_name)
                hst_ha_stamp_band = hst_habu_band_red
                hst_ha_cont_sub_stamp_band = hst_habu_band_red + '_cont_sub'
                band_list += [hst_habu_band_red]
                # get H-alpha continuum subtracted image
                band_list += [hst_habu_band_red + '_cont_sub']
            else:
                hst_ha_stamp_band = None
                hst_habu_band_red = None
                hst_ha_cont_sub_stamp_band = None
            hst_habu_band_green = hst_bvi_band_blue
            hst_habu_band_blue = 'F336W'
        else:
            hst_ha_stamp_band = None
            hst_ha_cont_sub_stamp_band = None
            hst_habu_band_red = None
            hst_habu_band_green = None
            hst_habu_band_blue = None

        if include_nircam & helper_func.BandTools.check_nircam_obs(target=self.target_name):
            nircam_stamp_band_list = helper_func.BandTools.get_nircam_obs_band_list(target=self.target_name)
            band_list += nircam_stamp_band_list
            nircam_band_red = 'F300M'
            nircam_band_green = 'F335M'
            nircam_band_blue = 'F200W'
        else:
            nircam_stamp_band_list = []
            nircam_band_red = None
            nircam_band_green = None
            nircam_band_blue = None

        if include_miri & helper_func.BandTools.check_miri_obs(target=self.target_name):
            miri_stamp_band_list = helper_func.BandTools.get_miri_obs_band_list(target=self.target_name)
            band_list += miri_stamp_band_list
            miri_band_red = 'F1130W'
            miri_band_green = 'F1000W'
            miri_band_blue = 'F770W'
        else:
            miri_stamp_band_list = []
            miri_band_red = None
            miri_band_green = None
            miri_band_blue = None

        # load all bands into constructor
        self.load_phangs_bands(band_list=band_list, flux_unit='MJy/sr')

        # get data from cluster catalog
        cluster_access = ClusterCatAccess()
        cluster_cat_data = cluster_access.get_quick_access(target=hst_cluster_cat_ref_dict['hst_cluster_cat_name'],
                                                           classify=hst_cluster_cat_ref_dict['classify'],
                                                           cluster_class=hst_cluster_cat_ref_dict['cluster_class'])
        mask_selected_cluster = cluster_cat_data['phangs_cluster_id'] == phangs_cluster_id
        ra_region = cluster_cat_data['ra'][mask_selected_cluster][0]
        dec_region = cluster_cat_data['dec'][mask_selected_cluster][0]


        # get hst_bvi_zoom_in
        img_hst_bvi_overview, wcs_hst_bvi_overview = self.get_target_overview_rgb_img(red_band=hst_bvi_band_red,
                                                                                      green_band=hst_bvi_band_green,
                                                                                      blue_band=hst_bvi_band_blue,
                                                                                      overview_img_size=(500, 500))

        img_hst_bvi_zoom_in, wcs_hst_bvi_zoom_in = self.get_rgb_zoom_in(ra=ra_region, dec=dec_region,
                                                                        cutout_size=env_cutout_size,
                                                                        circle_rad=circle_rad_region,
                                                                        band_red=hst_bvi_band_red,
                                                                        band_green=hst_bvi_band_green,
                                                                        band_blue=hst_bvi_band_blue)

        if hst_ha_stamp_band is not None:
            img_hst_habu_zoom_in, wcs_hst_habu_zoom_in = self.get_rgb_zoom_in(ra=ra_region, dec=dec_region,
                                                                              cutout_size=env_cutout_size,
                                                                              circle_rad=circle_rad_region,
                                                                              band_red=hst_habu_band_red,
                                                                              band_green=hst_habu_band_green,
                                                                              band_blue=hst_habu_band_blue)
        else:
            img_hst_habu_zoom_in, wcs_hst_habu_zoom_in = None, None
        if nircam_stamp_band_list:
            img_nircam_zoom_in, wcs_nircam_zoom_in = self.get_rgb_zoom_in(ra=ra_region, dec=dec_region,
                                                                          cutout_size=env_cutout_size,
                                                                          circle_rad=circle_rad_region,
                                                                          band_red=nircam_band_red,
                                                                          band_green=nircam_band_green,
                                                                          band_blue=nircam_band_blue)
        else:
            img_nircam_zoom_in, wcs_nircam_zoom_in = None, None
        if miri_stamp_band_list:
            img_miri_zoom_in, wcs_miri_zoom_in = self.get_rgb_zoom_in(ra=ra_region, dec=dec_region,
                                                                      cutout_size=env_cutout_size,
                                                                      circle_rad=circle_rad_region,
                                                                      band_red=miri_band_red,
                                                                      band_green=miri_band_green,
                                                                      band_blue=miri_band_blue)
        else:
            img_miri_zoom_in, wcs_miri_zoom_in = None, None

        # load cutout stamps
        cutout_dict_stamp = self.get_band_cutout_dict(ra_cutout=ra_region, dec_cutout=dec_region,
                                                      cutout_size=stamp_cutout_size,
                                                      band_list=band_list)

        # plotting
        figure = plt.figure(figsize=(30, 50))

        # limits for the overview image
        overview_width = 0.45
        overview_height = 0.45
        overview_left_align = 0.035
        overview_bottom_align = 0.628

        # limits for the zoom in panels
        zoom_in_width = 0.24
        zoom_in_height = 0.24
        zoom_in_left_align = 0.51
        zoom_in_bottom_align = 0.66
        zoom_in_space_horizontal = -0.095
        zoom_in_space_vertical = 0.005

        # limits for the stamps
        stamp_width = 0.1
        stamp_height = 0.1
        stamp_left_align = 0.035
        stamp_bottom_align = 0.60
        stamp_space_horizontal = -0.02
        stamp_space_vertical = 0.005

        stamp_dict = {
            'hst_stamp_band_list': hst_stamp_band_list,
            'nircam_stamp_band_list': nircam_stamp_band_list,
            'miri_stamp_band_list': miri_stamp_band_list,
            'cutout_dict_stamp': cutout_dict_stamp, 'stamp_width': stamp_width, 'stamp_height': stamp_height,
            'stamp_left_align': stamp_left_align, 'stamp_bottom_align': stamp_bottom_align,
            'stamp_space_horizontal': stamp_space_horizontal, 'stamp_space_vertical': stamp_space_vertical,
            'hst_ha_stamp_band': hst_ha_stamp_band, 'ha_cont_sub_stamp_band': hst_ha_cont_sub_stamp_band,
            'cbar_log_scale': cbar_log_scale
        }

        ax_hst_bvi_overview = figure.add_axes([overview_left_align, overview_bottom_align,
                                               overview_width, overview_height],
                                              projection=wcs_hst_bvi_overview)

        ax_hst_bvi_zoom_in = figure.add_axes([zoom_in_left_align,
                                              zoom_in_bottom_align + zoom_in_height + zoom_in_space_horizontal,
                                              zoom_in_width, zoom_in_height], projection=wcs_hst_bvi_zoom_in)
        hst_overview_dict = {
            'ax_hst_bvi_overview': ax_hst_bvi_overview,
            'img_hst_bvi_overview': img_hst_bvi_overview,
            'wcs_hst_bvi_overview': wcs_hst_bvi_overview,
            'hst_bvi_band_red': hst_bvi_band_red,
            'hst_bvi_band_green': hst_bvi_band_green,
            'hst_bvi_band_blue': hst_bvi_band_blue,
        }
        hst_zoom_in_dict = {
            'ax_hst_bvi_zoom_in': ax_hst_bvi_zoom_in,
            'img_hst_bvi_zoom_in': img_hst_bvi_zoom_in,
            'wcs_hst_bvi_zoom_in': wcs_hst_bvi_zoom_in,
            'hst_bvi_band_red': hst_bvi_band_red,
            'hst_bvi_band_green': hst_bvi_band_green,
            'hst_bvi_band_blue': hst_bvi_band_blue,
        }

        if include_h_alpha:
            ax_hst_habu_zoom_in = figure.add_axes([zoom_in_left_align + zoom_in_width + zoom_in_space_vertical,
                                                   zoom_in_bottom_align + zoom_in_height + zoom_in_space_horizontal,
                                                   zoom_in_width, zoom_in_height], projection=wcs_hst_habu_zoom_in)
            hst_habu_zoom_in_dict = {
                'ax_hst_habu_zoom_in': ax_hst_habu_zoom_in,
                'img_hst_habu_zoom_in': img_hst_habu_zoom_in,
                'wcs_hst_habu_zoom_in': wcs_hst_habu_zoom_in,
                'hst_habu_band_red': hst_habu_band_red,
                'hst_habu_band_green': hst_habu_band_green,
                'hst_habu_band_blue': hst_habu_band_blue,
            }
        else:
            hst_habu_zoom_in_dict = None
        if include_nircam & (img_nircam_zoom_in is not None):
            ax_nircam_zoom_in = figure.add_axes([zoom_in_left_align,
                                                 zoom_in_bottom_align, zoom_in_width, zoom_in_height],
                                                projection=wcs_nircam_zoom_in)
            nircam_zoom_in_dict = {
                'ax_nircam_zoom_in': ax_nircam_zoom_in,
                'img_nircam_zoom_in': img_nircam_zoom_in,
                'wcs_nircam_zoom_in': wcs_nircam_zoom_in,
                'nircam_band_red': nircam_band_red,
                'nircam_band_green': nircam_band_green,
                'nircam_band_blue': nircam_band_blue,
            }
        else:
            nircam_zoom_in_dict = None
        if include_miri & (img_miri_zoom_in is not None):
            ax_miri_zoom_in = figure.add_axes([zoom_in_left_align + zoom_in_width + zoom_in_space_vertical, zoom_in_bottom_align,
                                               zoom_in_width, zoom_in_height], projection=wcs_miri_zoom_in)
            miri_zoom_in_dict = {
                'ax_miri_zoom_in': ax_miri_zoom_in,
                'img_miri_zoom_in': img_miri_zoom_in,
                'wcs_miri_zoom_in': wcs_miri_zoom_in,
                'miri_band_red': miri_band_red,
                'miri_band_green': miri_band_green,
                'miri_band_blue': miri_band_blue,
            }
        else:
            miri_zoom_in_dict = None

        self.plot_over_view_zoom_in_panel(figure=figure, ra_region=ra_region, dec_region=dec_region,
                                          hst_overview_dict=hst_overview_dict, hst_zoom_in_dict=hst_zoom_in_dict,
                                          stamp_dict=stamp_dict, hst_habu_zoom_in_dict=hst_habu_zoom_in_dict,
                                          nircam_zoom_in_dict=nircam_zoom_in_dict, miri_zoom_in_dict=miri_zoom_in_dict,
                                          radius_dict=None,
                                          env_cutout_size=env_cutout_size, circle_rad_region=None,
                                          stamp_cutout_size=stamp_cutout_size, fontsize_large=fontsize_large,
                                          fontsize_small=fontsize_small)

        # plot SED
        # load large cutout dict for flux density estimation
        self.change_phangs_band_units(band_list=band_list, new_unit='mJy')
        cutout_dict_zoom_in = self.get_band_cutout_dict(ra_cutout=ra_region, dec_cutout=dec_region,
                                                        cutout_size=env_cutout_size,
                                                        band_list=band_list)


        # get 50 and 80 % encirceled energy radius
        radius_dict = {}
        for band in band_list:
            if 'cont_sub' in band:
                continue
            if band in phangs_info.hst_obs_band_dict[self.target_name]['wfc3_uvis_observed_bands']:
                radius_dict.update({'aperture_%s_ee50' % band: phys_params.hst_encircle_apertures_wfc3_uvis2_arcsec[band]['ee50']})
                radius_dict.update({'aperture_%s_ee80' % band: phys_params.hst_encircle_apertures_wfc3_uvis2_arcsec[band]['ee80']})
            elif band in phangs_info.hst_obs_band_dict[self.target_name]['acs_wfc1_observed_bands']:
                radius_dict.update({'aperture_%s_ee50' % band: phys_params.hst_encircle_apertures_acs_wfc1_arcsec[band]['ee50']})
                radius_dict.update({'aperture_%s_ee80' % band: phys_params.hst_encircle_apertures_acs_wfc1_arcsec[band]['ee80']})
            elif band in ['F657N', 'F658N']:
                radius_dict.update({'aperture_%s_ee50' % band: phys_params.hst_encircle_apertures_wfc3_uvis2_arcsec[band]['ee50']})
                radius_dict.update({'aperture_%s_ee80' % band: phys_params.hst_encircle_apertures_wfc3_uvis2_arcsec[band]['ee80']})
            elif band in phangs_info.jwst_obs_band_dict[self.target_name]['nircam_observed_bands']:
                radius_dict.update({'aperture_%s_ee50' % band: phys_params.nircam_encircle_apertures_arcsec[band]['ee50']})
                radius_dict.update({'aperture_%s_ee80' % band: phys_params.nircam_encircle_apertures_arcsec[band]['ee80']})
            elif band in phangs_info.jwst_obs_band_dict[self.target_name]['miri_observed_bands']:
                radius_dict.update({'aperture_%s_ee50' % band: phys_params.miri_encircle_apertures_arcsec[band]['ee50']})
                radius_dict.update({'aperture_%s_ee80' % band: phys_params.miri_encircle_apertures_arcsec[band]['ee80']})


        # get fluxes from circular aperture
        flux_dict_ee50 = {}
        flux_dict_ee80 = {}
        flux_dict_extend_phot = {}

        # get cluster_photometry
        index_ext_phot_table = cluster_access.identify_phangs_id_in_ext_phot_table(
            target=hst_cluster_cat_ref_dict['hst_cluster_cat_name'], single_phangs_cluster_id=phangs_cluster_id)

        pos_obj = SkyCoord(ra=ra_region*u.deg, dec=dec_region*u.deg)
        for band in band_list:
            if 'cont_sub' in band:
                continue
            if cutout_dict_zoom_in['%s_img_cutout' % band].data is None:
                continue
            flux_ee50, flux_ee50_err = helper_func.PhotTools.extract_flux_from_circ_aperture(
                data=cutout_dict_zoom_in['%s_img_cutout' % band].data,
                wcs=cutout_dict_zoom_in['%s_img_cutout' % band].wcs,
                pos=pos_obj,
                aperture_rad=radius_dict['aperture_%s_ee50' % band])
            flux_dict_ee50.update({'flux_%s' % band: flux_ee50, 'flux_err_%s' % band: flux_ee50_err})
            flux_ee80, flux_ee80_err = helper_func.PhotTools.extract_flux_from_circ_aperture(
                data=cutout_dict_zoom_in['%s_img_cutout' % band].data,
                wcs=cutout_dict_zoom_in['%s_img_cutout' % band].wcs,
                pos=pos_obj,
                aperture_rad=radius_dict['aperture_%s_ee80' % band])
            flux_dict_ee80.update({'flux_%s' % band: flux_ee80, 'flux_err_%s' % band: flux_ee80_err})

            extend_phot_flux = cluster_access.get_extend_phot_band_flux(
                target=hst_cluster_cat_ref_dict['hst_cluster_cat_name'], band=band)[index_ext_phot_table]
            extend_phot_flux_err = cluster_access.get_extend_phot_band_flux_err(
                target=hst_cluster_cat_ref_dict['hst_cluster_cat_name'], band=band)[index_ext_phot_table]
            flux_dict_extend_phot.update({'flux_%s' % band: extend_phot_flux, 'flux_err_%s' % band: extend_phot_flux_err})
        # plot flux-density distribution
        ax_sed = figure.add_axes([0.065, 0.32, 0.9, 0.15])

        wave_ee50_list = []
        wave_ee80_list = []
        flux_ee50_list = []
        flux_ee80_list = []
        wave_extend_phot_list = []
        flux_extend_phot_list = []
        for band in band_list:
            if '_cont_sub' in band:
                continue
            if 'flux_%s' % band not in flux_dict_ee50:
                continue
            if band in hst_stamp_band_list:
                color = 'tab:blue'
                mean_wave = helper_func.BandTools.get_hst_band_wave(
                    band=band, instrument=helper_func.BandTools.get_hst_instrument(target=self.target_name, band=band))
                min_wave = helper_func.BandTools.get_hst_band_wave(
                    band=band, instrument=helper_func.BandTools.get_hst_instrument(target=self.target_name, band=band),
                    wave_estimator='min_wave')
                max_wave = helper_func.BandTools.get_hst_band_wave(
                    band=band, instrument=helper_func.BandTools.get_hst_instrument(target=self.target_name, band=band),
                    wave_estimator='max_wave')
            elif band == hst_ha_stamp_band:
                color = 'tab:red'
                mean_wave = helper_func.BandTools.get_hst_band_wave(
                    band=band, instrument=helper_func.BandTools.get_hst_ha_instrument(target=self.target_ha_name))
                min_wave = helper_func.BandTools.get_hst_band_wave(
                    band=band, instrument=helper_func.BandTools.get_hst_ha_instrument(target=self.target_ha_name),
                    wave_estimator='min_wave')
                max_wave = helper_func.BandTools.get_hst_band_wave(
                    band=band, instrument=helper_func.BandTools.get_hst_ha_instrument(target=self.target_ha_name),
                    wave_estimator='max_wave')
            elif band in nircam_stamp_band_list:
                color = 'tab:green'
                mean_wave = helper_func.BandTools.get_jwst_band_wave(band=band)
                min_wave = helper_func.BandTools.get_jwst_band_wave(band=band, wave_estimator='min_wave')
                max_wave = helper_func.BandTools.get_jwst_band_wave(band=band, wave_estimator='max_wave')
            elif band in miri_stamp_band_list:
                color = 'tab:purple'
                mean_wave = helper_func.BandTools.get_jwst_band_wave(band=band, instrument='miri')
                min_wave = helper_func.BandTools.get_jwst_band_wave(band=band, instrument='miri', wave_estimator='min_wave')
                max_wave = helper_func.BandTools.get_jwst_band_wave(band=band, instrument='miri', wave_estimator='max_wave')
            else:
                color = ''
                mean_wave = np.nan
                min_wave = np.nan
                max_wave = np.nan
            if ((flux_dict_ee50['flux_%s' % band] < 0) |
                    (flux_dict_ee50['flux_%s' % band] < 3*flux_dict_ee50['flux_err_%s' % band])):
                ax_sed.errorbar(mean_wave, 3*flux_dict_ee50['flux_err_%s' % band],
                                yerr=flux_dict_ee50['flux_err_%s' % band],
                                xerr=[[mean_wave-min_wave], [max_wave-mean_wave]],
                                ecolor=color, elinewidth=5, capsize=10, uplims=True, xlolims=False)
            else:
                wave_ee50_list.append(mean_wave)
                flux_ee50_list.append(flux_dict_ee50['flux_%s' % band])
                ax_sed.errorbar(mean_wave, flux_dict_ee50['flux_%s' % band],
                                         xerr=[[mean_wave-min_wave], [max_wave-mean_wave]],
                                         yerr=flux_dict_ee50['flux_err_%s' % band],
                                         fmt='v', color=color, ms=20)

            if ((flux_dict_ee80['flux_%s' % band] < 0) |
                    (flux_dict_ee80['flux_%s' % band] < 3*flux_dict_ee80['flux_err_%s' % band])):
                ax_sed.errorbar(mean_wave, 3*flux_dict_ee80['flux_err_%s' % band],
                                yerr=flux_dict_ee80['flux_err_%s' % band],
                                xerr=[[mean_wave-min_wave], [max_wave-mean_wave]],
                                ecolor=color, elinewidth=5, capsize=10, uplims=True, xlolims=False)
            else:
                wave_ee80_list.append(mean_wave)
                flux_ee80_list.append(flux_dict_ee80['flux_%s' % band])
                ax_sed.errorbar(mean_wave, flux_dict_ee80['flux_%s' % band],
                                         xerr=[[mean_wave-min_wave], [max_wave-mean_wave]],
                                         yerr=flux_dict_ee80['flux_err_%s' % band],
                                         fmt='^', color=color, ms=20)

            if ((flux_dict_extend_phot['flux_%s' % band] < 0) |
                    (flux_dict_extend_phot['flux_%s' % band] < 3*flux_dict_extend_phot['flux_err_%s' % band])):
                ax_sed.errorbar(mean_wave, 3*flux_dict_extend_phot['flux_err_%s' % band],
                                yerr=flux_dict_extend_phot['flux_err_%s' % band],
                                xerr=[[mean_wave-min_wave], [max_wave-mean_wave]],
                                ecolor=color, elinewidth=5, capsize=10, uplims=True, xlolims=False)
            else:
                wave_extend_phot_list.append(mean_wave)
                flux_extend_phot_list.append(flux_dict_extend_phot['flux_%s' % band])
                ax_sed.errorbar(mean_wave, flux_dict_extend_phot['flux_%s' % band],
                                xerr=[[mean_wave-min_wave], [max_wave-mean_wave]],
                                yerr=flux_dict_extend_phot['flux_err_%s' % band],
                                fmt='o', color=color, ms=20)

        sort_ee50 = np.argsort(wave_ee50_list)
        sort_ee80 = np.argsort(wave_ee80_list)
        sort_extend_phot = np.argsort(wave_extend_phot_list)
        p1_line, = ax_sed.plot(np.array(wave_ee80_list)[sort_ee80], np.array(flux_ee80_list)[sort_ee80],
                               color='gray', linestyle='--', linewidth=2)
        p2_line, = ax_sed.plot(np.array(wave_ee50_list)[sort_ee50], np.array(flux_ee50_list)[sort_ee50],
                               color='indianred', linestyle=':', linewidth=2)
        p3_line, = ax_sed.plot(np.array(wave_extend_phot_list)[sort_extend_phot],
                               np.array(flux_extend_phot_list)[sort_extend_phot], color='k', linewidth=2)
        p1_scatter = ax_sed.scatter([], [], marker="^", c="gray", s=200)
        p2_scatter = ax_sed.scatter([], [], marker="v", c="gray", s=200)
        p3_scatter = ax_sed.scatter([], [], marker="o", c="gray", s=200)

        ax_sed.legend([(p1_line, p1_scatter), (p2_line, p2_scatter), (p3_line, p3_scatter)],
                      ['EE 80%', 'EE 50%', 'apert. corr.'],
                      handler_map={tuple: HandlerTuple(ndivide=2)}, fontsize=fontsize_large)


        ax_sed.set_yscale('log')
        ax_sed.set_xscale('log')
        ax_sed.set_xlabel(r'Wavelength [$\mu$m]', fontsize=fontsize_large)
        ax_sed.set_ylabel(r'Flux [mJy]', fontsize=fontsize_large)
        ax_sed.tick_params(axis='both', which='both', width=2, direction='in', labelsize=fontsize_large)

        # put information into title
        title_str_list = [
            'PHANGS-ID = %i, hum class = %i, color-color-class = %s, CI = %.1f' %
            (phangs_cluster_id, cluster_cat_data['cluster_class_hum'][mask_selected_cluster][0],
             cluster_cat_data['ccd_class'][mask_selected_cluster][0].decode("utf-8"),
             cluster_cat_data['ci'][mask_selected_cluster][0]),
            'X = %.3f, X = %.3f, R.A. = %.7f, DEC. = %.7f' %
            (cluster_cat_data['x'][mask_selected_cluster][0],
             cluster_cat_data['y'][mask_selected_cluster][0],
             cluster_cat_data['ra'][mask_selected_cluster][0],
             cluster_cat_data['dec'][mask_selected_cluster][0]),
            r'Age $_{\rm (fix)}$ = %i Myr, '
            r'M$_*$ $_{\rm (fix)}$ = %.2f $\times$ 1e4 M$_{\odot}$,'
            r' E(B-V) $_{\rm (fix)}$ = %.2f mag' %
            (cluster_cat_data['age'][mask_selected_cluster][0],
             cluster_cat_data['mstar'][mask_selected_cluster][0] / 1e4,
             cluster_cat_data['ebv'][mask_selected_cluster][0]),
            r'Age $_{\rm (old)}$ = %i Myr, '
            r'M$_*$ $_{\rm (old)}$ = %.2f $\times$ 1e4 '
            r'M$_{\odot}$, E(B-V) $_{\rm (old)}$ = %.2f mag' %
            (cluster_cat_data['age_old'][mask_selected_cluster][0],
             cluster_cat_data['mstar_old'][mask_selected_cluster][0] / 1e4,
             cluster_cat_data['ebv_old'][mask_selected_cluster][0])]

        if 'age_brad' in hst_cluster_cat_ref_dict.keys():
            title_str_list.append(r'Age $_{\rm (Brad)}$ = %i Myr' % (hst_cluster_cat_ref_dict['age_brad']))

        # cross match_age_estimates from Kiana
        extend_sed_cat_obj_index = cluster_access.identify_phangs_id_in_ext_sed_table(
            target=hst_cluster_cat_ref_dict['hst_cluster_cat_name'], single_phangs_cluster_id=phangs_cluster_id)
        print('extend_sed_cat_obj_index ', extend_sed_cat_obj_index)
        if extend_sed_cat_obj_index is not None:
            title_str_list.append(
                r'Age $_{\rm (Kiana)}$ = %i Myr, '
                r'M$_*$ $_{\rm (Kiana)}$ = %.2f $\times$ 1e4 M$_{\odot}$, '
                r'E(B-V) $_{\rm (Kiana)}$ = %.2f mag' %
                (cluster_access.get_extend_sed_age(
                    target=hst_cluster_cat_ref_dict['hst_cluster_cat_name'])[extend_sed_cat_obj_index],
                 cluster_access.get_extend_sed_mstar(
                     target=hst_cluster_cat_ref_dict['hst_cluster_cat_name'])[extend_sed_cat_obj_index]/1e4,
                 cluster_access.get_extend_sed_ebv(
                     target=hst_cluster_cat_ref_dict['hst_cluster_cat_name'])[extend_sed_cat_obj_index]))
        # put up strings as title
        title = None
        for title_string in title_str_list:
            if title is None:
                title = ''
            else:
                title += '\n'
            title += title_string

        ax_sed.set_title(title, fontsize=fontsize_large, loc='left')

        # get alma observations
        gas_access = GasAccess(target_name=self.target_name)

        mol_gas_dens_data, mol_gas_dens_wcs = gas_access.get_alma_h2_map()
        cutout_alma = helper_func.CoordTools.get_img_cutout(img=mol_gas_dens_data, wcs=mol_gas_dens_wcs,
                                                            coord=SkyCoord(ra=ra_region*u.deg, dec=dec_region*u.deg),
                                                            cutout_size=env_cutout_size)

        if not np.isnan(cutout_alma.data).all():
            ax_alma = figure.add_axes([0.05, -0.05, 0.4, 0.4], projection=cutout_alma.wcs)
            ax_cbar_alma = figure.add_axes([0.1, 0.28, 0.3, 0.01])
            cmap_alma = 'inferno'

            min_alma_value = np.nanmin(cutout_alma.data)
            max_alma_value = np.nanmax(cutout_alma.data)
            if min_alma_value <= 0:
                min_alma_value = max_alma_value / 100
            norm = LogNorm(min_alma_value, max_alma_value)
            helper_func.create_cbar(ax_cbar=ax_cbar_alma, cmap=cmap_alma, norm=norm,
                                    cbar_label=r'log($\Sigma_{\rm H2}$/[M$_{\odot}$ kpc$^{-2}$])', fontsize=fontsize_large,
                                    ticks=None, labelpad=2, tick_width=2, orientation='horizontal', extend='neither')

            ax_alma.imshow(cutout_alma.data, norm=norm, cmap=cmap_alma)

            # load clound catalogs
            rad_cloud_arcsec = gas_access.get_cloud_rad_arcsec()
            ra_cloud, dec_cloud = gas_access.get_cloud_coords()
            coord_cloud_world = SkyCoord(ra=ra_cloud*u.deg, dec=dec_cloud*u.deg)

            coord_cloud_pix = cutout_alma.wcs.world_to_pixel(coord_cloud_world)
            for idx, pos in enumerate(coord_cloud_world):
                rad = rad_cloud_arcsec[idx]
                if np.isnan(rad):
                    continue
                if ((coord_cloud_pix[0][idx] < 0) | (coord_cloud_pix[1][idx] < 0) |
                        (coord_cloud_pix[0][idx] > cutout_alma.data.shape[0]) |
                        (coord_cloud_pix[1][idx] > cutout_alma.data.shape[1])):
                    continue
                plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_alma, pos=pos, rad=rad, color='blue', line_style='-', line_width=2)

            # central_coord_pixel = cutout_alma.wcs.world_to_pixel(SkyCoord(ra=ra_region*u.deg, dec=dec_region*u.deg))
            # ax_alma.scatter(central_coord_pixel[0], central_coord_pixel[1], s=180, marker='*', color='k')
            # ax_alma.set_xlim(0, cutout_alma.data.shape[0])
            # ax_alma.set_ylim(0, cutout_alma.data.shape[1])
            plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_alma, pos=SkyCoord(ra=ra_region*u.deg, dec=dec_region*u.deg),
                                          rad=0.1, color='k', line_style='-', line_width=2, fill=True)
            self.arr_axis_params(ax=ax_alma, fontsize=fontsize_large, labelsize=fontsize_large)
            # ax_alma.set_title(info_string, fontsize=fontsize_large)

        # add CCD

        ax_ccd = figure.add_axes([0.55, 0.03, 0.4, 0.25])
        all_cluster_dict_hum_cl12 = cluster_access.get_quick_access(cluster_class='class12')
        x_lim_vi = (-0.6, 1.6)
        y_lim_ub = (0.9, -2.2)

        # get some quality cuts
        mask_detect_ubvi_hum = (all_cluster_dict_hum_cl12['detect_u'] * all_cluster_dict_hum_cl12['detect_b'] *
                                all_cluster_dict_hum_cl12['detect_v'] * all_cluster_dict_hum_cl12['detect_i'])

        # get gauss und segmentations
        gauss_map = plotting_tools.CCDPlottingTools.calc_gauss_weight_map(
            x_data=all_cluster_dict_hum_cl12['color_vi_vega'], y_data=all_cluster_dict_hum_cl12['color_ub_vega'],
            x_data_err=all_cluster_dict_hum_cl12['color_vi_err'], y_data_err=all_cluster_dict_hum_cl12['color_ub_err'],
            x_lim=x_lim_vi, y_lim=tuple(reversed(y_lim_ub)), n_x_bins=90, n_y_bins=90, kernel_size=9, kernel_std=1.0)

        vmax = np.nanmax(gauss_map)
        ax_ccd.imshow(gauss_map, origin='lower', extent=(x_lim_vi[0], x_lim_vi[1], y_lim_ub[1], y_lim_ub[0]),
                      interpolation='nearest', aspect='auto', cmap='Greys', vmin=0+vmax/10, vmax=vmax/1.1)

        # get hulls
        vi_color_ycl_hull, ub_color_ycl_hull = cluster_access.load_ccd_hull()
        vi_color_map_hull, ub_color_map_hull = cluster_access.load_ccd_hull(cluster_region='map')
        vi_color_ogcc_hull, ub_color_ogcc_hull = cluster_access.load_ccd_hull(cluster_region='ogcc')

        ax_ccd.plot(vi_color_ycl_hull, ub_color_ycl_hull, color='blue', linewidth=3)
        ax_ccd.plot(vi_color_map_hull, ub_color_map_hull, color='green', linewidth=3)
        ax_ccd.plot(vi_color_ogcc_hull, ub_color_ogcc_hull, color='red', linewidth=3)

        ax_ccd.errorbar(cluster_cat_data['color_vi_vega'][mask_selected_cluster][0],
                        cluster_cat_data['color_ub_vega'][mask_selected_cluster][0],
                        xerr=cluster_cat_data['color_vi_err'][mask_selected_cluster][0],
                        yerr=cluster_cat_data['color_ub_err'][mask_selected_cluster][0],
                        fmt='o', ms=10, capsize=5, markeredgewidth=2, elinewidth=3, color='red')

        plotting_tools.CCDPlottingTools.display_models(ax=ax_ccd)
        vi_int = 0.95
        ub_int = -1.8
        plotting_tools.CCDPlottingTools.plot_reddening_vect(ax=ax_ccd, text=True,
                                                            x_color_int=vi_int, y_color_int=ub_int,
                                                            x_text_offset=0.1, y_text_offset=-0.05)

        # ax_ccd.plot(color_color_dict['model_vi_sol'], color_color_dict['model_ub_sol'], color='tab:cyan', linewidth=5, label=r'BC03, Z$_{\odot}$')

        # ax_ccd.scatter(color_color_dict['model_vi_sol'][color_color_dict['age_mod_sol'] == 1],
        #             color_color_dict['model_ub_sol'][color_color_dict['age_mod_sol'] == 1], color='k', s=150)
        # ax_ccd.scatter(color_color_dict['model_vi_sol'][color_color_dict['age_mod_sol'] == 5],
        #             color_color_dict['model_ub_sol'][color_color_dict['age_mod_sol'] == 5], color='k', s=150)
        # ax_ccd.scatter(color_color_dict['model_vi_sol'][color_color_dict['age_mod_sol'] == 10],
        #             color_color_dict['model_ub_sol'][color_color_dict['age_mod_sol'] == 10], color='k', s=150)
        # ax_ccd.scatter(color_color_dict['model_vi_sol'][color_color_dict['age_mod_sol'] == 100],
        #             color_color_dict['model_ub_sol'][color_color_dict['age_mod_sol'] == 100], color='k', s=150)
        # ax_ccd.scatter(color_color_dict['model_vi_sol'][color_color_dict['age_mod_sol'] == 1000],
        #             color_color_dict['model_ub_sol'][color_color_dict['age_mod_sol'] == 1000], color='k', s=150)
        # ax_ccd.scatter(color_color_dict['model_vi_sol'][color_color_dict['age_mod_sol'] == 13750],
        #             color_color_dict['model_ub_sol'][color_color_dict['age_mod_sol'] == 13750], color='k', s=150)
        #
        # ax_ccd.text(color_color_dict['model_vi_sol'][color_color_dict['age_mod_sol'] == 1],
        #             color_color_dict['model_ub_sol'][color_color_dict['age_mod_sol'] == 1] - 0.05,
        #             '1 Myr', horizontalalignment='center', verticalalignment='bottom',
        #             color='k', fontsize=fontsize_large)
        # ax_ccd.text(color_color_dict['model_vi_sol'][color_color_dict['age_mod_sol'] == 5] - 0.05,
        #             color_color_dict['model_ub_sol'][color_color_dict['age_mod_sol'] == 5],
        #             '5 Myr', horizontalalignment='right', verticalalignment='center',
        #             color='k', fontsize=fontsize_large)
        # ax_ccd.text(color_color_dict['model_vi_sol'][color_color_dict['age_mod_sol'] == 10] + 0.05,
        #             color_color_dict['model_ub_sol'][color_color_dict['age_mod_sol'] == 10],
        #             '10 Myr', horizontalalignment='left', verticalalignment='center',
        #             color='k', fontsize=fontsize_large)
        # ax_ccd.text(color_color_dict['model_vi_sol'][color_color_dict['age_mod_sol'] == 100] - 0.05,
        #             color_color_dict['model_ub_sol'][color_color_dict['age_mod_sol'] == 100],
        #             '100 Myr', horizontalalignment='right', verticalalignment='center',
        #             color='k', fontsize=fontsize_large)
        # ax_ccd.text(color_color_dict['model_vi_sol'][color_color_dict['age_mod_sol'] == 1000],
        #             color_color_dict['model_ub_sol'][color_color_dict['age_mod_sol'] == 1000] - 0.05,
        #             '1 Gyr', horizontalalignment='center', verticalalignment='bottom',
        #             color='k', fontsize=fontsize_large)
        # ax_ccd.text(color_color_dict['model_vi_sol'][color_color_dict['age_mod_sol'] == 13750],
        #             color_color_dict['model_ub_sol'][color_color_dict['age_mod_sol'] == 13750] + 0.05,
        #             '13.7 Gyr', horizontalalignment='center', verticalalignment='top',
        #             color='k', fontsize=fontsize_large)
        #

        # helper_func.plot_reddening_vect(ax=ax_ccd,
        #                 x_color_int=0.8, y_color_int=-1.6, av_val=1,
        #                 linewidth=3, line_color='k',
        #                 text=True, fontsize=fontsize_large, text_color='k', x_text_offset=0.1, y_text_offset=-0.05)
        #
        # ax_ccd.legend(frameon=False, loc=3, fontsize=fontsize_large)

        color_color_str = ('U-B = %.3f $\pm$ %.3f mag, V-I = %.3f $\pm$ %.3f mag' %
                           (cluster_cat_data['color_ub_vega'][mask_selected_cluster][0],
                            cluster_cat_data['color_ub_err'][mask_selected_cluster][0],
                            cluster_cat_data['color_vi_vega'][mask_selected_cluster][0],
                            cluster_cat_data['color_vi_err'][mask_selected_cluster][0]))

        ax_ccd.set_title(color_color_str, fontsize=fontsize_large)
        ax_ccd.set_xlim(x_lim_vi)
        ax_ccd.set_ylim(y_lim_ub)

        ax_ccd.set_ylabel('U (F336W) - B (F438W/F435W'+'$^*$'+')', fontsize=fontsize_large)
        ax_ccd.set_xlabel('V (F555W) - I (F814W)', fontsize=fontsize_large)

        ax_ccd.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize_large)


        # color_color_dict


        return figure

    def plot_over_view_zoom_in_panel(self,
                                     figure,
                                     ra_region, dec_region,
                                     hst_overview_dict, hst_zoom_in_dict, stamp_dict, hst_habu_zoom_in_dict=None,
                                     nircam_zoom_in_dict=None, miri_zoom_in_dict=None,
                                     radius_dict=None,
                                     env_cutout_size=(10, 10), circle_rad_region=None,
                                     stamp_cutout_size=(2.5, 2.5),
                                     fontsize_large=30, fontsize_small=20,):
        # plot the RGB images
        hst_overview_dict['ax_hst_bvi_overview'].imshow(hst_overview_dict['img_hst_bvi_overview'])
        pe = [patheffects.withStroke(linewidth=3, foreground="w")]
        hst_overview_dict['ax_hst_bvi_overview'].text(0.02, 0.98, 'HST', horizontalalignment='left',
                                                      verticalalignment='top', fontsize=fontsize_large, color='white',
                                                      transform=hst_overview_dict['ax_hst_bvi_overview'].transAxes,
                                                      path_effects=pe)
        hst_overview_dict['ax_hst_bvi_overview'].text(0.02, 0.95, hst_overview_dict['hst_bvi_band_red'].upper(),
                                                      horizontalalignment='left', verticalalignment='top',
                                                      fontsize=fontsize_large, color='red',
                                                      transform=hst_overview_dict['ax_hst_bvi_overview'].transAxes,
                                                      path_effects=pe)
        hst_overview_dict['ax_hst_bvi_overview'].text(0.02, 0.92, hst_overview_dict['hst_bvi_band_green'].upper(),
                                                      horizontalalignment='left', verticalalignment='top',
                                                      fontsize=fontsize_large, color='green',
                                                      transform=hst_overview_dict['ax_hst_bvi_overview'].transAxes,
                                                      path_effects=pe)
        hst_overview_dict['ax_hst_bvi_overview'].text(0.02, 0.89, hst_overview_dict['hst_bvi_band_blue'].upper(),
                                                      horizontalalignment='left', verticalalignment='top',
                                                      fontsize=fontsize_large, color='blue',
                                                      transform=hst_overview_dict['ax_hst_bvi_overview'].transAxes,
                                                      path_effects=pe)
        hst_overview_dict['ax_hst_bvi_overview'].set_title(self.target_name.upper(), fontsize=fontsize_large + 10)
        self.arr_axis_params(ax=hst_overview_dict['ax_hst_bvi_overview'],  tick_color='white',
                             fontsize=fontsize_large, labelsize=fontsize_large)
        plotting_tools.WCSPlottingTools.draw_box(ax=hst_overview_dict['ax_hst_bvi_overview'], wcs=hst_overview_dict['wcs_hst_bvi_overview'],
                             coord=SkyCoord(ra=ra_region*u.deg, dec=dec_region*u.deg),
                             box_size=env_cutout_size, color='red', line_style='--')

        hst_zoom_in_dict['ax_hst_bvi_zoom_in'].imshow(hst_zoom_in_dict['img_hst_bvi_zoom_in'])
        hst_zoom_in_dict['ax_hst_bvi_zoom_in'].text(0.02, 0.98, 'HST', horizontalalignment='left',
                                                    verticalalignment='top', fontsize=fontsize_large, color='white',
                                                    transform=hst_zoom_in_dict['ax_hst_bvi_zoom_in'].transAxes,
                                                    path_effects=pe)
        hst_zoom_in_dict['ax_hst_bvi_zoom_in'].text(0.02, 0.92, hst_zoom_in_dict['hst_bvi_band_red'].upper(),
                                                    horizontalalignment='left', verticalalignment='top',
                                                    fontsize=fontsize_large, color='red',
                                                    transform=hst_zoom_in_dict['ax_hst_bvi_zoom_in'].transAxes,
                                                    path_effects=pe)
        hst_zoom_in_dict['ax_hst_bvi_zoom_in'].text(0.02, 0.86, hst_zoom_in_dict['hst_bvi_band_green'].upper(),
                                                    horizontalalignment='left', verticalalignment='top',
                                                    fontsize=fontsize_large, color='green',
                                                    transform=hst_zoom_in_dict['ax_hst_bvi_zoom_in'].transAxes,
                                                    path_effects=pe)
        hst_zoom_in_dict['ax_hst_bvi_zoom_in'].text(0.02, 0.80, hst_zoom_in_dict['hst_bvi_band_blue'].upper(),
                                                    horizontalalignment='left', verticalalignment='top',
                                                    fontsize=fontsize_large, color='blue',
                                                    transform=hst_zoom_in_dict['ax_hst_bvi_zoom_in'].transAxes,
                                                    path_effects=pe)
        plotting_tools.WCSPlottingTools.draw_box(ax=hst_zoom_in_dict['ax_hst_bvi_zoom_in'], wcs=hst_zoom_in_dict['wcs_hst_bvi_zoom_in'],
                             coord=SkyCoord(ra=ra_region*u.deg, dec=dec_region*u.deg),
                             box_size=stamp_cutout_size, color='cyan', line_style='--')
        if circle_rad_region is not None:
            plotting_tools.WCSPlottingTools.plot_coord_circle(ax=hst_zoom_in_dict['ax_hst_bvi_zoom_in'], pos=SkyCoord(ra=ra_region*u.deg, dec=dec_region*u.deg),
                                          rad=circle_rad_region, color='white', line_style='-', line_width=2, alpha=1., fill=False)
        self.arr_axis_params(ax=hst_zoom_in_dict['ax_hst_bvi_zoom_in'], ra_tick_label=False,  ra_axis_label=' ',
                             dec_axis_label=' ', tick_color='white', fontsize=fontsize_large, labelsize=fontsize_large)
        if hst_habu_zoom_in_dict is not None:
            hst_habu_zoom_in_dict['ax_hst_habu_zoom_in'].imshow(hst_habu_zoom_in_dict['img_hst_habu_zoom_in'])
            self.arr_axis_params(ax=hst_habu_zoom_in_dict['ax_hst_habu_zoom_in'], ra_tick_label=False,
                                 dec_tick_label=False, ra_axis_label=' ', dec_axis_label=' ', tick_color='white',
                                 fontsize=fontsize_large, labelsize=fontsize_large)
            hst_habu_zoom_in_dict['ax_hst_habu_zoom_in'].imshow(hst_habu_zoom_in_dict['img_hst_habu_zoom_in'])
            hst_habu_zoom_in_dict['ax_hst_habu_zoom_in'].text(0.02, 0.98, 'HST', horizontalalignment='left',
                                                              verticalalignment='top', fontsize=fontsize_large,
                                                              color='white',
                                                              transform=
                                                              hst_habu_zoom_in_dict['ax_hst_habu_zoom_in'].transAxes,
                                                              path_effects=pe)
            hst_habu_zoom_in_dict['ax_hst_habu_zoom_in'].text(0.02, 0.92,
                                                              hst_habu_zoom_in_dict['hst_habu_band_red'].upper(),
                                                              horizontalalignment='left', verticalalignment='top',
                                                              fontsize=fontsize_large, color='red',
                                                              transform=
                                                              hst_habu_zoom_in_dict['ax_hst_habu_zoom_in'].transAxes,
                                                              path_effects=pe)
            hst_habu_zoom_in_dict['ax_hst_habu_zoom_in'].text(0.02, 0.86,
                                                              hst_habu_zoom_in_dict['hst_habu_band_green'].upper(),
                                                              horizontalalignment='left', verticalalignment='top',
                                                              fontsize=fontsize_large, color='green',
                                                              transform=
                                                              hst_habu_zoom_in_dict['ax_hst_habu_zoom_in'].transAxes,
                                                              path_effects=pe)
            hst_habu_zoom_in_dict['ax_hst_habu_zoom_in'].text(0.02, 0.80,
                                                              hst_habu_zoom_in_dict['hst_habu_band_blue'].upper(),
                                                              horizontalalignment='left', verticalalignment='top',
                                                              fontsize=fontsize_large, color='blue',
                                                              transform=
                                                              hst_habu_zoom_in_dict['ax_hst_habu_zoom_in'].transAxes,
                                                              path_effects=pe)
            plotting_tools.WCSPlottingTools.draw_box(ax=hst_habu_zoom_in_dict['ax_hst_habu_zoom_in'],
                                 wcs=hst_habu_zoom_in_dict['wcs_hst_habu_zoom_in'],
                                 coord=SkyCoord(ra=ra_region*u.deg, dec=dec_region*u.deg),
                                 box_size=stamp_cutout_size, color='cyan', line_style='--')
            if circle_rad_region is not None:
                plotting_tools.WCSPlottingTools.plot_coord_circle(ax=hst_habu_zoom_in_dict['ax_hst_habu_zoom_in'],
                                              pos=SkyCoord(ra=ra_region*u.deg, dec=dec_region*u.deg),
                                              rad=circle_rad_region, color='white', line_width=2)
        if nircam_zoom_in_dict is not None:
            nircam_zoom_in_dict['ax_nircam_zoom_in'].imshow(nircam_zoom_in_dict['img_nircam_zoom_in'])
            self.arr_axis_params(ax=nircam_zoom_in_dict['ax_nircam_zoom_in'],  ra_axis_label=' ', dec_axis_label=' ',
                                 tick_color='white', fontsize=fontsize_large, labelsize=fontsize_large)
            nircam_zoom_in_dict['ax_nircam_zoom_in'].text(0.02, 0.98, 'NIRCAM', horizontalalignment='left',
                                                          verticalalignment='top', fontsize=fontsize_large,
                                                          color='white', transform=
                                                          nircam_zoom_in_dict['ax_nircam_zoom_in'].transAxes,
                                                          path_effects=pe)
            nircam_zoom_in_dict['ax_nircam_zoom_in'].text(0.02, 0.92, nircam_zoom_in_dict['nircam_band_red'].upper(),
                                                          horizontalalignment='left', verticalalignment='top',
                                                          fontsize=fontsize_large, color='red', transform=
                                                          nircam_zoom_in_dict['ax_nircam_zoom_in'].transAxes,
                                                          path_effects=pe)
            nircam_zoom_in_dict['ax_nircam_zoom_in'].text(0.02, 0.86, nircam_zoom_in_dict['nircam_band_green'].upper(),
                                                          horizontalalignment='left', verticalalignment='top',
                                                          fontsize=fontsize_large, color='green', transform=
                                                          nircam_zoom_in_dict['ax_nircam_zoom_in'].transAxes,
                                                          path_effects=pe)
            nircam_zoom_in_dict['ax_nircam_zoom_in'].text(0.02, 0.80, nircam_zoom_in_dict['nircam_band_blue'].upper(),
                                                          horizontalalignment='left', verticalalignment='top',
                                                          fontsize=fontsize_large, color='blue', transform=
                                                          nircam_zoom_in_dict['ax_nircam_zoom_in'].transAxes,
                                                          path_effects=pe)
            plotting_tools.WCSPlottingTools.draw_box(ax=nircam_zoom_in_dict['ax_nircam_zoom_in'], wcs=nircam_zoom_in_dict['wcs_nircam_zoom_in'],
                                 coord=SkyCoord(ra=ra_region*u.deg, dec=dec_region*u.deg), box_size=stamp_cutout_size,
                                 color='cyan', line_style='--')
            if circle_rad_region is not None:
                plotting_tools.WCSPlottingTools.plot_coord_circle(ax=nircam_zoom_in_dict['ax_nircam_zoom_in'],
                                              pos=SkyCoord(ra=ra_region*u.deg, dec=dec_region*u.deg),
                                              rad=circle_rad_region, color='white', line_width=2)
        if miri_zoom_in_dict is not None:
            miri_zoom_in_dict['ax_miri_zoom_in'].imshow(miri_zoom_in_dict['img_miri_zoom_in'])
            self.arr_axis_params(ax=miri_zoom_in_dict['ax_miri_zoom_in'], dec_tick_label=False, ra_axis_label=' ',
                                 dec_axis_label=' ', tick_color='white',  fontsize=fontsize_large,
                                 labelsize=fontsize_large)
            miri_zoom_in_dict['ax_miri_zoom_in'].text(0.02, 0.98, 'MIRI', horizontalalignment='left',
                                                      verticalalignment='top', fontsize=fontsize_large, color='white',
                                                      transform=miri_zoom_in_dict['ax_miri_zoom_in'].transAxes,
                                                      path_effects=pe)
            miri_zoom_in_dict['ax_miri_zoom_in'].text(0.02, 0.92, miri_zoom_in_dict['miri_band_red'].upper(),
                                                      horizontalalignment='left', verticalalignment='top',
                                                      fontsize=fontsize_large, color='red',
                                                      transform=miri_zoom_in_dict['ax_miri_zoom_in'].transAxes,
                                                      path_effects=pe)
            miri_zoom_in_dict['ax_miri_zoom_in'].text(0.02, 0.86, miri_zoom_in_dict['miri_band_green'].upper(),
                                                      horizontalalignment='left', verticalalignment='top',
                                                      fontsize=fontsize_large, color='green',
                                                      transform=miri_zoom_in_dict['ax_miri_zoom_in'].transAxes,
                                                      path_effects=pe)
            miri_zoom_in_dict['ax_miri_zoom_in'].text(0.02, 0.80, miri_zoom_in_dict['miri_band_blue'].upper(),
                                                      horizontalalignment='left', verticalalignment='top',
                                                      fontsize=fontsize_large, color='blue',
                                                      transform=miri_zoom_in_dict['ax_miri_zoom_in'].transAxes,
                                                      path_effects=pe)
            plotting_tools.WCSPlottingTools.draw_box(ax=miri_zoom_in_dict['ax_miri_zoom_in'], wcs=miri_zoom_in_dict['wcs_miri_zoom_in'],
                                 coord=SkyCoord(ra=ra_region*u.deg, dec=dec_region*u.deg),
                                 box_size=stamp_cutout_size, color='cyan', line_style='--')
            if circle_rad_region is not None:
                plotting_tools.WCSPlottingTools.plot_coord_circle(ax=miri_zoom_in_dict['ax_miri_zoom_in'],
                                              pos=SkyCoord(ra=ra_region*u.deg, dec=dec_region*u.deg),
                                              rad=circle_rad_region, color='white', line_width=2)

        # add stamp axis
        stamp_row_count_down = 0
        # add stamp axis for hst
        ax_hst_stamp_list = []
        hst_stamp_index = 0

        for hst_stamp_index, hst_stamp_band in enumerate(stamp_dict['hst_stamp_band_list']):

            ax_hst_stamp_list.append(
                figure.add_axes([stamp_dict['stamp_left_align'] +
                                 hst_stamp_index*(stamp_dict['stamp_width'] + stamp_dict['stamp_space_vertical']),
                                 stamp_dict['stamp_bottom_align'] +
                                 stamp_row_count_down*(stamp_dict['stamp_height'] +
                                                       stamp_dict['stamp_space_horizontal']),
                                 stamp_dict['stamp_width'], stamp_dict['stamp_height']],
                                projection=stamp_dict['cutout_dict_stamp']['%s_img_cutout' % hst_stamp_band].wcs))
            norm_hst_stamp = helper_func.compute_cbar_norm(
                cutout_list=stamp_dict['cutout_dict_stamp']['%s_img_cutout' % hst_stamp_band].data,
                log_scale=stamp_dict['cbar_log_scale'])
            ax_hst_stamp_list[hst_stamp_index].imshow(
                stamp_dict['cutout_dict_stamp']['%s_img_cutout' % hst_stamp_band].data, norm=norm_hst_stamp,
                cmap='Greys')
            if radius_dict is not None:
                plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_hst_stamp_list[hst_stamp_index],
                                              pos=SkyCoord(ra=ra_region*u.deg, dec=dec_region*u.deg),
                                              rad=radius_dict['aperture_%s_ee50' % hst_stamp_band],
                                              color='cyan', line_width=2)
                plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_hst_stamp_list[hst_stamp_index],
                                              pos=SkyCoord(ra=ra_region*u.deg, dec=dec_region*u.deg),
                                              rad=radius_dict['aperture_%s_ee80' % hst_stamp_band],
                                              color='red', line_width=2)

            plotting_tools.WCSPlottingTools.plot_coord_croshair(ax=ax_hst_stamp_list[hst_stamp_index],
                                              pos=SkyCoord(ra=ra_region*u.deg, dec=dec_region*u.deg),
                                            wcs=stamp_dict['cutout_dict_stamp']['%s_img_cutout' % hst_stamp_band].wcs,
                                              rad=0.3, hair_length=0.3,
                                              color='red', line_width=2)

            ax_hst_stamp_list[hst_stamp_index].set_title(hst_stamp_band.upper(), fontsize=fontsize_large,
                                                         color='tab:blue')
            if hst_stamp_index == 0:
                ra_tick_label, dec_tick_label = True, True
            else:
                ra_tick_label, dec_tick_label = False, False
            self.arr_axis_params(ax=ax_hst_stamp_list[hst_stamp_index], ra_tick_label=ra_tick_label,
                                 dec_tick_label=dec_tick_label, ra_axis_label=' ', dec_axis_label=' ',
                                 fontsize=fontsize_small, labelsize=fontsize_small)

        if hst_habu_zoom_in_dict is not None:
            ax_hst_ha_stamp = figure.add_axes([stamp_dict['stamp_left_align'] +
                                               (hst_stamp_index + 2)*(stamp_dict['stamp_width'] +
                                                                      stamp_dict['stamp_space_vertical']),
                                               stamp_dict['stamp_bottom_align'] +
                                               stamp_row_count_down*(stamp_dict['stamp_height'] +
                                                                     stamp_dict['stamp_space_horizontal']),
                                               stamp_dict['stamp_width'], stamp_dict['stamp_height']],
                                              projection=
                                              stamp_dict['cutout_dict_stamp']['%s_img_cutout' %
                                                                              stamp_dict['hst_ha_stamp_band']].wcs)
            norm_hst_ha_stamp = helper_func.compute_cbar_norm(
                cutout_list=stamp_dict['cutout_dict_stamp']['%s_img_cutout' % stamp_dict['hst_ha_stamp_band']].data,
                log_scale=stamp_dict['cbar_log_scale'])
            ax_hst_ha_stamp.imshow(
                stamp_dict['cutout_dict_stamp']['%s_img_cutout' % stamp_dict['hst_ha_stamp_band']].data,
                norm=norm_hst_ha_stamp, cmap='Greys')

            if radius_dict is not None:
                plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_hst_ha_stamp,
                                              pos=SkyCoord(ra=ra_region*u.deg, dec=dec_region*u.deg),
                                              rad=radius_dict['aperture_%s_ee50' % stamp_dict['hst_ha_stamp_band']],
                                              color='cyan', line_width=2)
                plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_hst_ha_stamp,
                                              pos=SkyCoord(ra=ra_region*u.deg, dec=dec_region*u.deg),
                                              rad=radius_dict['aperture_%s_ee80' % stamp_dict['hst_ha_stamp_band']],
                                              color='red', line_width=2)

            plotting_tools.WCSPlottingTools.plot_coord_croshair(ax=ax_hst_ha_stamp,
                                              pos=SkyCoord(ra=ra_region*u.deg, dec=dec_region*u.deg),
                                            wcs=stamp_dict['cutout_dict_stamp']['%s_img_cutout' % stamp_dict['hst_ha_stamp_band']].wcs,
                                              rad=0.3, hair_length=0.3,
                                              color='red', line_width=2)
            ax_hst_ha_stamp.set_title(stamp_dict['hst_ha_stamp_band'].upper(), fontsize=fontsize_large, color='tab:red')
            self.arr_axis_params(ax=ax_hst_ha_stamp, ra_axis_label=' ', dec_axis_label=' ', fontsize=fontsize_small,
                                 labelsize=fontsize_small)

            ax_hst_ha_cont_sub_stamp = figure.add_axes([stamp_dict['stamp_left_align'] +
                                                        (hst_stamp_index + 4)*(stamp_dict['stamp_width'] +
                                                                               stamp_dict['stamp_space_vertical']),
                                                        stamp_dict['stamp_bottom_align'] +
                                                        stamp_row_count_down*(stamp_dict['stamp_height'] +
                                                                              stamp_dict['stamp_space_horizontal']),
                                                        stamp_dict['stamp_width'], stamp_dict['stamp_height']],
                                                       projection=
                                                       stamp_dict['cutout_dict_stamp']
                                                       ['%s_img_cutout' % stamp_dict['ha_cont_sub_stamp_band']].wcs)
            norm_hst_ha_cont_sub_stamp = helper_func.compute_cbar_norm(
                cutout_list=stamp_dict['cutout_dict_stamp']['%s_img_cutout' %
                                                            stamp_dict['ha_cont_sub_stamp_band']].data,
                log_scale=stamp_dict['cbar_log_scale'])
            ax_hst_ha_cont_sub_stamp.imshow(stamp_dict['cutout_dict_stamp']['%s_img_cutout' %
                                                                            stamp_dict['ha_cont_sub_stamp_band']].data,
                                            norm=norm_hst_ha_cont_sub_stamp, cmap='Greys')
            plotting_tools.WCSPlottingTools.plot_coord_croshair(ax=ax_hst_ha_cont_sub_stamp,
                                              pos=SkyCoord(ra=ra_region*u.deg, dec=dec_region*u.deg),
                                            wcs=stamp_dict['cutout_dict_stamp']['%s_img_cutout' % stamp_dict['ha_cont_sub_stamp_band']].wcs,
                                              rad=0.3, hair_length=0.3,
                                              color='red', line_width=2)
            ax_hst_ha_cont_sub_stamp.set_title(r'Cont. sub H$\alpha$', fontsize=fontsize_large,
                                               color='tab:red')
            self.arr_axis_params(ax=ax_hst_ha_cont_sub_stamp, ra_axis_label=' ', dec_axis_label=' ',
                                 fontsize=fontsize_small, labelsize=fontsize_small)

        nircam_stamp_index = 0
        stamp_row_count_down -= 1
        if nircam_zoom_in_dict is not None:
            ax_nircam_stamp_list = []
            for nircam_stamp_index, nircam_stamp_band in enumerate(stamp_dict['nircam_stamp_band_list']):
                if np.sum(stamp_dict['cutout_dict_stamp']['%s_img_cutout' % nircam_stamp_band].data) == 0:
                    continue
                ax_nircam_stamp_list.append(
                    figure.add_axes([stamp_dict['stamp_left_align'] +
                                     nircam_stamp_index*(stamp_dict['stamp_width'] +
                                                         stamp_dict['stamp_space_vertical']),
                                     stamp_dict['stamp_bottom_align'] +
                                     stamp_row_count_down*(stamp_dict['stamp_height'] +
                                                           stamp_dict['stamp_space_horizontal']),
                                     stamp_dict['stamp_width'], stamp_dict['stamp_height']],
                                    projection=stamp_dict['cutout_dict_stamp']['%s_img_cutout' %
                                                                               nircam_stamp_band].wcs))
                norm_nircam_stamp = helper_func.compute_cbar_norm(
                    cutout_list=stamp_dict['cutout_dict_stamp']['%s_img_cutout' % nircam_stamp_band].data,
                    log_scale=stamp_dict['cbar_log_scale'])
                ax_nircam_stamp_list[nircam_stamp_index].imshow(stamp_dict['cutout_dict_stamp']['%s_img_cutout' %
                                                                                                nircam_stamp_band].data,
                                                                norm=norm_nircam_stamp, cmap='Greys')
                if radius_dict is not None:
                    plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_nircam_stamp_list[nircam_stamp_index],
                                                  pos=SkyCoord(ra=ra_region*u.deg, dec=dec_region*u.deg),
                                                  rad=radius_dict['aperture_%s_ee50' % nircam_stamp_band],
                                                  color='cyan', line_width=2)
                    plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_nircam_stamp_list[nircam_stamp_index],
                                                  pos=SkyCoord(ra=ra_region*u.deg, dec=dec_region*u.deg),
                                                  rad=radius_dict['aperture_%s_ee80' % nircam_stamp_band],
                                                  color='red', line_width=2)
                plotting_tools.WCSPlottingTools.plot_coord_croshair(ax=ax_nircam_stamp_list[nircam_stamp_index],
                                              pos=SkyCoord(ra=ra_region*u.deg, dec=dec_region*u.deg),
                                            wcs=stamp_dict['cutout_dict_stamp']['%s_img_cutout' % nircam_stamp_band].wcs,
                                              rad=0.3, hair_length=0.3,
                                              color='red', line_width=2)
                ax_nircam_stamp_list[nircam_stamp_index].set_title(nircam_stamp_band.upper(), fontsize=fontsize_large,
                                                                   color='tab:green')
                if nircam_stamp_index == 0:
                    ra_tick_label, dec_tick_label = True, True
                else:
                    ra_tick_label, dec_tick_label = False, False
                self.arr_axis_params(ax=ax_nircam_stamp_list[nircam_stamp_index], ra_tick_label=ra_tick_label,
                                     dec_tick_label=dec_tick_label, ra_axis_label=' ', dec_axis_label=' ',
                                     fontsize=fontsize_small, labelsize=fontsize_small)

        if miri_zoom_in_dict is not None:
            # stamp_row_count_down -= 1
            ax_miri_stamp_list = []
            for miri_stamp_index, miri_stamp_band in enumerate(stamp_dict['miri_stamp_band_list']):
                if np.sum(stamp_dict['cutout_dict_stamp']['%s_img_cutout' % miri_stamp_band].data) == 0:
                    continue
                ax_miri_stamp_list.append(
                    figure.add_axes([stamp_dict['stamp_left_align'] +
                                     (nircam_stamp_index + miri_stamp_index + 2)*(stamp_dict['stamp_width'] +
                                                                                  stamp_dict['stamp_space_vertical']),
                                     stamp_dict['stamp_bottom_align'] +
                                     stamp_row_count_down*(stamp_dict['stamp_height'] +
                                                           stamp_dict['stamp_space_horizontal']),
                                     stamp_dict['stamp_width'], stamp_dict['stamp_height']],
                                    projection=stamp_dict['cutout_dict_stamp']['%s_img_cutout' % miri_stamp_band].wcs))
                norm_miri_stamp = helper_func.compute_cbar_norm(
                    cutout_list=stamp_dict['cutout_dict_stamp']['%s_img_cutout' % miri_stamp_band].data,
                    log_scale=stamp_dict['cbar_log_scale'])
                ax_miri_stamp_list[miri_stamp_index].imshow(
                    stamp_dict['cutout_dict_stamp']['%s_img_cutout' % miri_stamp_band].data,
                    norm=norm_miri_stamp, cmap='Greys')
                if radius_dict is not None:
                    plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_miri_stamp_list[miri_stamp_index],
                                                  pos=SkyCoord(ra=ra_region*u.deg, dec=dec_region*u.deg),
                                                  rad=radius_dict['aperture_%s_ee50' % miri_stamp_band],
                                                  color='cyan', line_width=2)
                    plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_miri_stamp_list[miri_stamp_index],
                                                  pos=SkyCoord(ra=ra_region*u.deg, dec=dec_region*u.deg),
                                                  rad=radius_dict['aperture_%s_ee80' % miri_stamp_band],
                                                  color='red', line_width=2)
                plotting_tools.WCSPlottingTools.plot_coord_croshair(ax=ax_miri_stamp_list[miri_stamp_index],
                                              pos=SkyCoord(ra=ra_region*u.deg, dec=dec_region*u.deg),
                                            wcs=stamp_dict['cutout_dict_stamp']['%s_img_cutout' % miri_stamp_band].wcs,
                                              rad=0.3, hair_length=0.3,
                                              color='red', line_width=2)
                ax_miri_stamp_list[miri_stamp_index].set_title(miri_stamp_band.upper(), fontsize=fontsize_large,
                                                               color='tab:purple')
                if miri_stamp_index == 0:
                    ra_tick_label, dec_tick_label = True, True
                else:
                    ra_tick_label, dec_tick_label = False, False
                self.arr_axis_params(ax=ax_miri_stamp_list[miri_stamp_index], ra_tick_label=ra_tick_label,
                                     dec_tick_label=dec_tick_label, ra_axis_label=' ', dec_axis_label=' ',
                                     fontsize=fontsize_small, labelsize=fontsize_small)

    def get_rgb_zoom_in(self, ra, dec, cutout_size, circle_rad, band_red, band_green, band_blue, ref_band='blue'):
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
        ref_band : str
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

        ref_wcs = cutout['%s_img_cutout' % eval('band_%s' % ref_band)].wcs

        if not (cutout['%s_img_cutout' % band_red].data.shape ==
                cutout['%s_img_cutout' % band_green].data.shape ==
                cutout['%s_img_cutout' % band_blue].data.shape):
            new_shape = cutout['%s_img_cutout' % eval('band_%s' % ref_band)].data.shape
            if ref_band == 'red':
                cutout_data_red = cutout['%s_img_cutout' % band_red].data
            else:
                cutout_data_red = helper_func.CoordTools.reproject_image(data=cutout['%s_img_cutout' % band_red].data,
                                                              wcs=cutout['%s_img_cutout' % band_red].wcs,
                                                              new_wcs=ref_wcs, new_shape=new_shape)
            if ref_band == 'green':
                cutout_data_green = cutout['%s_img_cutout' % band_green].data
            else:
                cutout_data_green = helper_func.CoordTools.reproject_image(data=cutout['%s_img_cutout' % band_green].data,
                                                                wcs=cutout['%s_img_cutout' % band_green].wcs,
                                                                new_wcs=ref_wcs, new_shape=new_shape)
            if ref_band == 'blue':
                cutout_data_blue = cutout['%s_img_cutout' % band_blue].data
            else:
                cutout_data_blue = helper_func.CoordTools.reproject_image(data=cutout['%s_img_cutout' % band_blue].data,
                                                                wcs=cutout['%s_img_cutout' % band_blue].wcs,
                                                                new_wcs=ref_wcs, new_shape=new_shape)
        else:
            cutout_data_red = cutout['%s_img_cutout' % band_red].data
            cutout_data_green = cutout['%s_img_cutout' % band_green].data
            cutout_data_blue = cutout['%s_img_cutout' % band_blue].data

        # get rgb image
        min_red_hst, max_red_hst = self.get_image_scale_with_circle(
            img_data=cutout_data_red,
            img_wcs=ref_wcs, ra=ra, dec=dec, circle_rad=circle_rad)
        min_green_hst, max_green_hst = self.get_image_scale_with_circle(
            img_data=cutout_data_green,
            img_wcs=ref_wcs, ra=ra, dec=dec, circle_rad=circle_rad)
        min_blue_hst, max_blue_hst = self.get_image_scale_with_circle(
            img_data=cutout_data_blue,
            img_wcs=ref_wcs, ra=ra, dec=dec, circle_rad=circle_rad)

        cutout_rgb_img = self.get_rgb_img(data_r=cutout_data_red,
                                   data_g=cutout_data_green,
                                   data_b=cutout_data_blue,
                                   min_max_r=(min_red_hst, max_red_hst),
                                   min_max_g=(min_green_hst, max_green_hst),
                                   min_max_b=(min_blue_hst, max_blue_hst),
                                   scaletype_r='abs', scaletype_g='abs', scaletype_b='abs')

        return cutout_rgb_img, ref_wcs

    def get_rgb_img_(self, ra, dec, cutout_size, band_red, band_green, band_blue, ref_band='blue',
                     color_r='#FF4433', color_g='#40E0D0', color_b='#1F51FF',
                     overview_img_size=(500, 500),
                     min_max_r=(0.3, 99.5), min_max_g=(0.3, 99.5), min_max_b=(0.3, 99.5),
                                   gamma_r=17.5, gamma_g=17.5, gamma_b=17.5,
                                   gamma_corr_r=17.5, gamma_corr_g=17.5, gamma_corr_b=17.5,
                                   combined_gamma=17.5):
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
        ref_band : str
            can be blue, green or red. In case the images are not the same size they get reprojected to one frame

        Returns
        -------
        rgb_image, wcs : ``numpy.ndarray``, ``astropy.wcs.WCS``
        """

        self.load_phangs_bands(band_list=[band_red, band_green, band_blue],
                               flux_unit='MJy/sr', load_err=False)

        cutout = self.get_band_cutout_dict(ra_cutout=ra, dec_cutout=dec, cutout_size=cutout_size,
                                           band_list=[band_red, band_green, band_blue])

        ref_cutout_shape = cutout['%s_img_cutout' % eval('band_%s' % ref_band)].data.shape

        ra_max = cutout['%s_img_cutout' % eval('band_%s' % ref_band)].wcs.pixel_to_world(
            0, 0).ra.value
        ra_min = cutout['%s_img_cutout' % eval('band_%s' % ref_band)].wcs.pixel_to_world(
            ref_cutout_shape[0], ref_cutout_shape[1]).ra.value
        dec_min = cutout['%s_img_cutout' % eval('band_%s' % ref_band)].wcs.pixel_to_world(
            0, 0).dec.value
        dec_max = cutout['%s_img_cutout' % eval('band_%s' % ref_band)].wcs.pixel_to_world(
            ref_cutout_shape[0], ref_cutout_shape[1]).dec.value

        new_wcs = helper_func.CoordTools.construct_wcs(ra_min=ra_min, ra_max=ra_max, dec_min=dec_max, dec_max=dec_min,
                                                       img_shape=overview_img_size, quadratic_image=False)


        img_data_red = helper_func.CoordTools.reproject_image(data=cutout['%s_img_cutout' % band_red].data,
                                                              wcs=cutout['%s_img_cutout' % band_red].wcs,
                                                              new_wcs=new_wcs, new_shape=overview_img_size)
        img_data_green = helper_func.CoordTools.reproject_image(data=cutout['%s_img_cutout' % band_green].data,
                                                                wcs=cutout['%s_img_cutout' % band_green].wcs,
                                                                new_wcs=new_wcs, new_shape=overview_img_size)
        img_data_blue = helper_func.CoordTools.reproject_image(data=cutout['%s_img_cutout' % band_blue].data,
                                                               wcs=cutout['%s_img_cutout' % band_blue].wcs,
                                                               new_wcs=new_wcs, new_shape=overview_img_size)

        img_data_red[img_data_red == 0] = np.nan
        img_data_green[img_data_green == 0] = np.nan
        img_data_blue[img_data_blue == 0] = np.nan

        rgb_img = self.get_rgb_img(data_r=img_data_red, data_g=img_data_green, data_b=img_data_blue,
                                   color_r=color_r, color_g=color_g, color_b=color_b,
                                   min_max_r=min_max_r, min_max_g=min_max_g, min_max_b=min_max_b,
                                   gamma_r=gamma_r, gamma_g=gamma_g, gamma_b=gamma_b,
                                   gamma_corr_r=gamma_corr_r, gamma_corr_g=gamma_corr_g, gamma_corr_b=gamma_corr_b,
                                   combined_gamma=combined_gamma)

        return rgb_img, new_wcs

    def get_target_overview_rgb_img(self, red_band, green_band, blue_band, ref_band='red',
                                    overview_img_size=(500, 500)):
        """
        Function to create an overview RGB image of PHANGS HST observations

        Parameters
        ----------
        red_band, green_band, blue_band : str
            Can be specified to any hst band
        ref_band: str
            Band which is used for the image limits. can be red green or blue
        overview_img_size : tuple
            denotes the shape of the new image

        Returns
        -------
        rgb_image, wcs : ``numpy.ndarray``, ``astropy.wcs.WCS``
        """

        # band list need to be loaded
        self.load_phangs_bands(band_list=[red_band, green_band, blue_band], flux_unit='MJy/sr', load_err=False)
        # get overview image
        non_zero_elements = np.where(self.hst_bands_data['%s_data_img' % eval('%s_band' % ref_band)] != 0)

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

        ra_max = self.hst_bands_data['%s_wcs_img' % eval('%s_band' % ref_band)].pixel_to_world(
            min_index_ra_axis_x_val, min_index_ra_axis_y_val).ra.value
        ra_min = self.hst_bands_data['%s_wcs_img' % eval('%s_band' % ref_band)].pixel_to_world(
            max_index_ra_axis_x_val, max_index_ra_axis_y_val).ra.value

        dec_min = self.hst_bands_data['%s_wcs_img' % eval('%s_band' % ref_band)].pixel_to_world(
            min_index_dec_axis_x_val, min_index_dec_axis_y_val).dec.value
        dec_max = self.hst_bands_data['%s_wcs_img' % eval('%s_band' % ref_band)].pixel_to_world(
            max_index_dec_axis_x_val, max_index_dec_axis_y_val).dec.value

        new_wcs = helper_func.CoordTools.construct_wcs(ra_min=ra_min, ra_max=ra_max, dec_min=dec_max, dec_max=dec_min,
                                                       img_shape=overview_img_size, quadratic_image=True)

        img_data_red = helper_func.CoordTools.reproject_image(data=self.hst_bands_data['%s_data_img' % red_band],
                                                              wcs=self.hst_bands_data['%s_wcs_img' % red_band],
                                                              new_wcs=new_wcs, new_shape=overview_img_size)
        img_data_green = helper_func.CoordTools.reproject_image(data=self.hst_bands_data['%s_data_img' % green_band],
                                                                wcs=self.hst_bands_data['%s_wcs_img' % green_band],
                                                                new_wcs=new_wcs, new_shape=overview_img_size)
        img_data_blue = helper_func.CoordTools.reproject_image(data=self.hst_bands_data['%s_data_img' % blue_band],
                                                               wcs=self.hst_bands_data['%s_wcs_img' % blue_band],
                                                               new_wcs=new_wcs, new_shape=overview_img_size)

        img_data_red[img_data_red == 0] = np.nan
        img_data_green[img_data_green == 0] = np.nan
        img_data_blue[img_data_blue == 0] = np.nan

        hst_rgb = self.get_rgb_img(data_r=img_data_red, data_g=img_data_green, data_b=img_data_blue,
                                   min_max_r=(0.3, 99.5), min_max_g=(0.3, 99.5), min_max_b=(0.3, 99.5),
                                   gamma_r=17.5, gamma_g=17.5, gamma_b=17.5,
                                   gamma_corr_r=17.5, gamma_corr_g=17.5, gamma_corr_b=17.5,
                                   combined_gamma=17.5)

        return hst_rgb, new_wcs

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

    @staticmethod
    def get_rgb_img(data_r, data_g, data_b, color_r='#FF4433', color_g='#0FFF50', color_b='#1F51FF',
                    min_max_r=None, min_max_g=None, min_max_b=None,
                    rescalefn='asinh',
                    scaletype_r='perc', scaletype_g='perc', scaletype_b='perc',
                    gamma_r=2.2, gamma_g=2.2, gamma_b=2.2,
                    gamma_corr_r=2.2, gamma_corr_g=2.2, gamma_corr_b=2.2, combined_gamma=2.2):
        """
        Function to create an RGB image

        Parameters
        ----------
        data_r, data_g, data_b : ``numpy.ndarray``, array
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
            min_max_r = [0., 100.]
        if min_max_g is None:
            min_max_g = [0., 100.]
        if min_max_b is None:
            min_max_b = [0., 100.]

        grey_r = mcf.greyRGBize_image(data_r, rescalefn=rescalefn, scaletype=scaletype_r, min_max=min_max_r,
                                      gamma=gamma_r)
        grey_g = mcf.greyRGBize_image(data_g, rescalefn=rescalefn, scaletype=scaletype_g, min_max=min_max_g,
                                      gamma=gamma_g)
        grey_b = mcf.greyRGBize_image(data_b, rescalefn=rescalefn, scaletype=scaletype_b, min_max=min_max_b,
                                      gamma=gamma_b)
        r = mcf.colorize_image(grey_r, color_r, colorintype='hex', gammacorr_color=gamma_corr_r)
        g = mcf.colorize_image(grey_g, color_g, colorintype='hex', gammacorr_color=gamma_corr_g)
        b = mcf.colorize_image(grey_b, color_b, colorintype='hex', gammacorr_color=gamma_corr_b)
        return mcf.combine_multicolor([r, g, b], gamma=combined_gamma)

    @staticmethod
    def plot_circle_on_wcs_img(ax, pos, rad, color, line_style='-', line_width=2., alpha=1., fill=False):
        """
        plots circle on image using coordinates and WCS to orientate

        Parameters
        ----------
        ax : ``astropy.visualization.wcsaxes.core.WCSAxes``
            axis for plotting
        pos : ``astropy.coordinates.SkyCoord``
            position in form of Sky coordinates
        rad : float
            radius in arc seconds of circle
        color : str
            matplotlib color
        line_style : str
            matplotlib line style
        line_width : float
        alpha : float
            matplotlib alpha factor
        fill : bool
            flag whether circle should be filled or not

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
            for pos_i, rad_i, color_i, line_style_i, line_width_i, alpha_i in zip(pos, rad, color, line_style,
                                                                                  line_width, alpha):
                circle = SphericalCircle(pos_i, rad_i * u.arcsec, edgecolor=color_i, facecolor=face_color,
                                         linewidth=line_width_i,
                                         linestyle=line_style_i, alpha=alpha_i, transform=ax.get_transform('fk5'))
                ax.add_patch(circle)
        else:
            circle = SphericalCircle(pos, rad * u.arcsec, edgecolor=color, facecolor=face_color, linewidth=line_width,
                                     linestyle=line_style, alpha=alpha, transform=ax.get_transform('fk5'))
            ax.add_patch(circle)


    # @staticmethod
    # def get_sdd_image():
    #
    #
    #