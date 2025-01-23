"""
Tool to visualize PHANGS products with different analysis
"""

from phangs_visualizer import plotting_tools
from phangs_visualizer.phot_visualizer import PhotVisualizer
from phangs_visualizer import plot_params

class MultiPanelVisualizer:
    """

    """
    @staticmethod
    def phangs_holistic_viewer1(ra, dec, target_name=None, phot_visual_access=None,
                                plot_rad_profile=False, plot_sed=False):
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
        # phot_visual_access.plot_muse_spec(fig=fig, fig_dict=plot_params.holistic_viewer1_param_dic, ra=ra, dec=dec)

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
    def muse_spec_viwer(target_name, ra , dec, spec_rad=None):
        # create figure
        fig = plotting_tools.AxisTools.init_fig(fig_dict=plot_params.muse_spec_viwer_param_dict)

        # get photometry plotting access
        phot_visual_access = PhotVisualizer(target_name=target_name)

        # get MUSE spectrum from region
        spec_dict, ppxf_fit_dict, em_fit_dict = phot_visual_access.plot_muse_spec(
            fig=fig, fig_dict=plot_params.muse_spec_viwer_param_dict, ra=ra, dec=dec, n_nl_gauss=2)

        # plot hb and oiii
        phot_visual_access.plot_spec_features(fig=fig, fig_dict=plot_params.muse_spec_viwer_param_dict,
                                              ppxf_fit_dict=ppxf_fit_dict, em_fit_dict=em_fit_dict)
        return fig






