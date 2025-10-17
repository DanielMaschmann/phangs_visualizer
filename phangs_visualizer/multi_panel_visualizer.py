"""
Tool to visualize PHANGS products with different analysis
"""

from phangs_visualizer import plotting_tools
from phangs_visualizer.phot_visualizer import PhotVisualizer
from phangs_visualizer import plot_params
from phangs_data_access import spec_tools, phangs_info
import numpy as np


class MultiPanelVisualizer:
    """

    """
    @staticmethod
    def phangs_holistic_viewer1(ra, dec, target_name=None, phot_visual_access=None,
                                plot_rad_profile=False, plot_sed=False,
                                plot_muse=True, ppxf_fit_dict=None):
        """

        This method creates a holistic inspection plot for one coordinate.

        This is based on the phangs data access tools and therefore not universal for any objects.

        """

        # create figure
        fig = plotting_tools.AxisTools.init_fig(fig_dict=plot_params.holistic_viewer1_param_dic)

        if phot_visual_access is None:
            phot_visual_access = PhotVisualizer(target_name=target_name)

        # create the overview plot
        phot_visual_access.plot_hst_overview_panel(fig=fig, fig_dict=plot_params.holistic_viewer1_param_dic,
                                                   ra_box=ra, dec_box=dec)

        # plot environment zoom in panels
        phot_visual_access.plot_zoom_in_panel_group(fig=fig, fig_dict=plot_params.holistic_viewer1_param_dic,
                                                    ra=ra, dec=dec)

        # plot postage stamps
        phot_visual_access.plot_img_stamps(fig=fig, fig_dict=plot_params.holistic_viewer1_param_dic, ra=ra, dec=dec,
                                           plot_rad_profile=plot_rad_profile)

        # plot sed estimation
        if plot_sed:
            phot_visual_access.plot_sed_panel(fig=fig, fig_dict=plot_params.holistic_viewer1_param_dic, ra=ra, dec=dec)

        # # get_EW estimation
        # phot_visual_access.compute_ha_ew(fig=fig, fig_dict=plot_params.holistic_viewer1_param_dic, ra=ra, dec=dec)

        # # get MUSE spectrum from region
        if plot_muse:
            if ppxf_fit_dict is None:
                rad_arcsec = phangs_info.muse_obs_res_dict[phot_visual_access.spec_target_name]['copt_res'] / 2
                spec_dict = phot_visual_access.extract_muse_spec_circ_app(ra=ra, dec=dec,
                                                                          rad_arcsec=rad_arcsec, wave_range=None,
                                                                          res='copt')
                ppxf_fit_dict = spec_tools.SpecTools.fit_ppxf2spec(spec_dict=spec_dict,
                                                                   target=phot_visual_access.spec_target_name,
                                                                   sps_name='fsps', age_range=None, metal_range=None)
            phot_visual_access.plot_muse_spec(fig=fig, fig_dict=plot_params.holistic_viewer1_param_dic, ra=ra, dec=dec,
                                              ppxf_fit_dict=ppxf_fit_dict)

        return fig


    @staticmethod
    def phangs_holistic_viewer2(ra, dec, target_name=None, phot_visual_access=None,
                                plot_rad_profile=False, plot_sed=False):
        """

        This method creates a holistic inspection plot for one coordinate.

        This is based on the phangs data access tools and therefore not universal for any objects.

        """

        # create figure
        fig = plotting_tools.AxisTools.init_fig(fig_dict=plot_params.holistic_viewer2_param_dic)

        if phot_visual_access is None:
            phot_visual_access = PhotVisualizer(target_name=target_name)

        # create the overview plot
        phot_visual_access.plot_hst_overview_panel(fig=fig, fig_dict=plot_params.holistic_viewer2_param_dic,
                                                   ra_box=ra, dec_box=dec)

        # plot environment zoom in panels
        phot_visual_access.plot_zoom_in_panel_group_extra_nircam(fig=fig, fig_dict=plot_params.holistic_viewer2_param_dic,
                                                    ra=ra, dec=dec)

        # plot postage stamps
        phot_visual_access.plot_img_stamps_all(fig=fig, fig_dict=plot_params.holistic_viewer2_param_dic, ra=ra, dec=dec,
                                           plot_rad_profile=plot_rad_profile,
                                               individual_band_list=plot_params.holistic_viewer2_param_dic['individual_band_list'])

        # plot sed estimation
        if plot_sed:
            phot_visual_access.plot_sed_panel(fig=fig, fig_dict=plot_params.holistic_viewer2_param_dic, ra=ra, dec=dec,
                                              individual_band_list=plot_params.holistic_viewer2_param_dic['individual_band_list'])

        return fig



    @staticmethod
    def phangs_phot_viewer(ra, dec, target_name=None, phot_visual_access=None, return_flux_dict=False):
        """

        This method creates a holistic inspection plot for one coordinate.

        This is based on the phangs data access tools and therefore not universal for any objects.

        """

        # create figure
        fig = plotting_tools.AxisTools.init_fig(fig_dict=plot_params.phot_viewer_param_dic)

        # get photometry plotting access
        if phot_visual_access is None:
            phot_visual_access = PhotVisualizer(target_name=target_name)

        # create the overview plot
        phot_visual_access.plot_hst_overview_panel(fig=fig, fig_dict=plot_params.phot_viewer_param_dic,
                                                   ra_box=ra, dec_box=dec)
        #
        # # plot environment zoom in panels
        # phot_visual_access.plot_zoom_in_panel_group(fig=fig, fig_dict=plot_params.phot_viewer_param_dic,
        #                                             ra=ra, dec=dec)

        # plot postage stamps
        flux_dict = phot_visual_access.plot_phot_morph(fig=fig, fig_dict=plot_params.phot_viewer_param_dic, ra=ra, dec=dec,
                                           return_flux_dict=return_flux_dict)

        # # plot sed estimation
        # phot_visual_access.plot_sed_panel(fig=fig, fig_dict=plot_params.phot_viewer_param_dic, ra=ra, dec=dec)

        return fig, flux_dict


    @staticmethod
    def ism_phot_viewer(target_name, ra, dec):
        """

        This method creates an overview of MIRI photometry

        """

        # create figure
        fig = plotting_tools.AxisTools.init_fig(fig_dict=plot_params.ism_phot_viewer_param_dict)

        # get photometry plotting access
        phot_visual_access = PhotVisualizer(target_name=target_name)

        # plot_miri images
        phot_visual_access.plot_ism_cutout_and_bkg(fig=fig, fig_dict=plot_params.ism_phot_viewer_param_dict,
                                                   ra=ra, dec=dec)

        return fig

    @staticmethod
    def muse_spec_viewer(target_name, ra , dec, spec_rad=None):
        # create figure
        fig = plotting_tools.AxisTools.init_fig(fig_dict=plot_params.muse_spec_viewer_param_dict)

        # get photometry plotting access
        phot_visual_access = PhotVisualizer(target_name=target_name)

        # get MUSE spectrum from region
        spec_dict, ppxf_fit_dict, em_fit_dict = phot_visual_access.plot_muse_spec(
            fig=fig, fig_dict=plot_params.muse_spec_viewer_param_dict, ra=ra, dec=dec, n_nl_gauss=2)

        # plot hb and oiii
        phot_visual_access.plot_spec_features(fig=fig, fig_dict=plot_params.muse_spec_viewer_param_dict,
                                              ppxf_fit_dict=ppxf_fit_dict, em_fit_dict=em_fit_dict)
        return fig



    @staticmethod
    def fit_spec_viewer(target_name, ra , dec, rad_arcsec, spec_dict=None, ppxf_fit_dict=None, em_line_fit_dict=None,
                        phot_visual_access=None):
        # create figure
        fig = plotting_tools.AxisTools.init_fig(fig_dict=plot_params.fit_spec_viewer_param_dict)

        # get photometry plotting access
        if phot_visual_access is None:
            phot_visual_access = PhotVisualizer(target_name=target_name)

        if spec_dict is None:
            spec_dict = phot_visual_access.extract_muse_spec_circ_app(
                ra=ra, dec=dec, rad_arcsec=rad_arcsec, wave_range=None, res='copt')

        if ppxf_fit_dict is None:
            # redshift = spec_tools.SpecTools.get_target_ned_redshift(
            #     target=helper_func.FileTools.target_name_no_directions(target=phangs_spec.spec_target_name))
            # vsys = spec_tools.SpecTools.get_target_sys_vel(
            #     target=helper_func.FileTools.target_name_no_directions(target=phangs_spec.spec_target_name))
            #
            # ppxf fit
            ppxf_fit_dict = spec_tools.SpecTools.fit_ppxf2spec(spec_dict=spec_dict, target=target_name,
                                                           sps_name='fsps', age_range=None, metal_range=None)

        if em_line_fit_dict is None:
            em_line_fit_dict = spec_tools.SpecTools.fit_em_lines2spec(
                target=target_name, wave=ppxf_fit_dict['wave'],
                em_flux=ppxf_fit_dict['total_flux'] - ppxf_fit_dict['continuum_best_fit'],
                em_flux_err=ppxf_fit_dict['total_flux_err'],
                n_nl_gauss=2, n_nl_lorentz=0, n_bl_gauss=0,
                x_data_format='wave', instrument='muse', blue_limit=30., red_limit=30., search_outflow=False,
                outflow_shift='redshift', outflow_mu_offset=400, outflow_sig=1200,
                init_mu_nl_gauss=100, init_sig_nl_gauss=200)

        # get MUSE spectrum from region
        phot_visual_access.plot_muse_spec(
            fig=fig, fig_dict=plot_params.fit_spec_viewer_param_dict, ra=ra, dec=dec, ppxf_fit_dict=ppxf_fit_dict)

        # plot hb and oiii
        phot_visual_access.plot_spec_features(fig=fig, fig_dict=plot_params.fit_spec_viewer_param_dict,
                                              em_line_fit_dict=em_line_fit_dict)
        return fig






