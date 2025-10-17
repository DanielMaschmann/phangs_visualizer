"""
This script will gather all reference parameters for consistent plotting
"""


phangs_obs_cov_plot_kwargs_dict = {
    'hst':
        {
            'F275W': {'color': 'deepskyblue',
                      'line_width': 7,
                      'line_style': '-'},
            'F336W': {'color': 'dodgerblue',
                      'line_width': 5,
                      'line_style': '-'},
            'F435W': {'color': 'blue',
                      'line_width': 3,
                      'line_style': '-'},
            'F438W': {'color': 'blue',
                      'line_width': 3,
                      'line_style': '-'},
            'F555W': {'color': 'blueviolet',
                      'line_width': 3,
                      'line_style': '--'},
            'F657N': {'color': 'magenta',
                      'line_width': 3,
                      'line_style': '-'},
            'F658N': {'color': 'magenta',
                      'line_width': 3,
                      'line_style': '-'},
            'F814W': {'color': 'violet',
                      'line_width': 2,
                      'line_style': ':'},
        },
    'nircam':
        {
            'F200W': {'color': 'deeppink',
                      'line_width': 7,
                      'line_style': '-'},
            'F300M': {'color': 'crimson',
                      'line_width': 5,
                      'line_style': '-'},
            'F335M': {'color': 'red',
                      'line_width': 3,
                      'line_style': '--'},
            'F360M': {'color': 'orangered',
                      'line_width': 3,
                      'line_style': ':'},
        },
    'miri':
        {
            'F770W': {'color': 'lime',
                      'line_width': 7,
                      'line_style': '-'},
            'F1000W': {'color': 'springgreen',
                       'line_width': 5,
                      'line_style': '-'},
            'F1130W': {'color': 'turquoise',
                       'line_width': 3,
                      'line_style': '--'},
            'F2100W': {'color': 'cyan',
                       'line_width': 3,
                      'line_style': ':'},
        },
    'astrosat': {'color': 'peru', 'line_width': 6,
                      'line_style': '-'},
    'alma': {'color': 'orange', 'line_width': 6,
                      'line_style': '-'},
    'muse': {'color': 'yellow', 'line_width': 6,
                      'line_style': '-'},
}



# parameters for holistic overview plots


holistic_viewer1_param_dic = {

    # image proportions:
    'env_cutout_size': (10, 10),

    # Figure and axis parameters
    'fig_size': (45, 60),


    # overview image
    # general parameters
    'overview_red_band': 'I',
    'overview_green_band': 'V',
    'overview_blue_band': 'B',
    'overview_img_pixel_size': (500, 500),
    # limits for the overview image
    'overview_width': 0.35,
    'overview_height': 0.35,
    'overview_left_align': 0.035,
    'overview_bottom_align': 0.68,
    # parameters for RGB image
    'overview_img_params': {
        'color_r': '#FF4433',
        'color_g': '#0FFF50',
        'color_b': '#1F51FF',
        'min_max_r': (0.3, 99.5),
        'min_max_g': (0.3, 99.5),
        'min_max_b': (0.3, 99.5),
        'gamma_r': 17.5,
        'gamma_g': 17.5,
        'gamma_b': 17.5,
        'gamma_corr_r': 17.5,
        'gamma_corr_g': 17.5,
        'gamma_corr_b': 17.5,
        'combined_gamma': 17.5
    },
    'overview_scale_bar_length': 1,

    # zoom in panels
    # limits for the zoom in panels
    'zoom_in_width': 0.15,
    'zoom_in_height': 0.15,
    'zoom_in_left_align': 0.42,
    'zoom_in_bottom_align': 0.72,
    'zoom_in_space_horizontal': -0.02,
    'zoom_in_space_vertical': 0.005,
    # hst broad band image
    'hst_broad_band_red_band': 'I',
    'hst_broad_band_green_band': 'V',
    'hst_broad_band_blue_band': 'B',
    # hst h-alpha image
    'hst_ha_red_band': 'Ha',
    'hst_ha_green_band': 'B',
    'hst_ha_blue_band': 'U',
    # Nircam
    'nircam_red_band': 'F300M',
    'nircam_green_band': 'F335M',
    'nircam_blue_band': 'F200W',
    # miri
    'miri_red_band': 'F1130W',
    'miri_green_band': 'F1000W',
    'miri_blue_band': 'F770W',
    # astrosat
    'astrosat_band': 'F148W',
    'astrosat_cbar_left_align': 0.92,
    'astrosat_cbar_bottom_align': 0.87,
    'astrosat_cbar_width': 0.02,
    'astrosat_cbar_height': 0.10,
    'astrosat_norm': 'log',
    'astrosat_cmap': 'Greys',

    # alma
    'alma_res': 150,
    'alma_alpha_co_method': 'B13_scaling',
    'alma_cbar_left_align': 0.92,
    'alma_cbar_bottom_align': 0.75,
    'alma_cbar_width': 0.02,
    'alma_cbar_height': 0.10,
    'alma_norm': 'log',
    'alma_cmap': 'inferno',
    'zoom_in_scale_bar_length_1': 100,
    'zoom_in_scale_bar_length_2': 5,



    # limits for the stamps
    'stamp_width': 0.1,
    'stamp_height': 0.1,
    'stamp_left_align': 0.035,
    'stamp_bottom_align': 0.35,
    'stamp_space_horizontal': 0.07,
    'stamp_space_vertical': 0.005,

    'rad_pro_width': 0.1,
    'rad_pro_height': 0.06,
    'rad_pro_left_align': 0.035,
    'rad_pro_bottom_align': 0.46,
    'rad_pro_space_horizontal': 0.11,
    'rad_pro_space_vertical': 0.005,

    'stamp_size': (2.5, 2.5),
    'stamp_scale_bar_length_1': 1,
    'stamp_scale_bar_length_2': 30,

    # limits for the stamps
    'ha_ew_width': 0.17,
    'ha_ew_height': 0.14,
    'ha_ew_left_align': 0.81,
    'ha_ew_bottom_align': 0.55,
    'ha_ew_ap_rad_pix_list': [1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12, 13, 14 ,15],
    'ha_ew_annulus_rad_in_pix': 16,
    'ha_ew_annulus_rad_out_pix': 17,

    # limits for the SED
    'sed_width': 0.75,
    'sed_height': 0.15,
    'sed_left_align': 0.035,
    'sed_bottom_align': 0.20,
    'sed_size_hst': (4, 4),
    'sed_size_nircam': (6, 6),
    'sed_size_miri': (7, 7),

    # 'hst_broad_band_ap_rad_pix_list': [1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12],
    # 'hst_broad_band_annulus_rad_in_pix': 13,
    # 'hst_broad_band_annulus_rad_out_pix': 15,
    #
    # 'hst_ha_ap_rad_pix_list': [1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12],
    # 'hst_ha_annulus_rad_in_pix': 13,
    # 'hst_ha_annulus_rad_out_pix': 15,
    #
    # 'nircam_ap_rad_pix_list': [1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12],
    # 'nircam_annulus_rad_in_pix': 13,
    # 'nircam_annulus_rad_out_pix': 15,
    #
    # 'miri_ap_rad_pix_list': [1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12],
    # 'miri_annulus_rad_in_pix': 13,
    # 'miri_annulus_rad_out_pix': 15,

    'muse_spec_width': 0.80,
    'muse_spec_height': 0.16,
    'muse_spec_left_align': 0.035,
    'muse_spec_bottom_align': 0.02,

    'muse_spec_x_lim': ('min', 9250),
    # 'muse_spec_x_lim': ('min', 'max'),
    'muse_spec_y_lim': 'cont',


    'muse_ha_zoom_in_width': 0.12,
    'muse_ha_zoom_in_height': 0.12,
    'muse_ha_zoom_in_left_align': 0.79,
    'muse_ha_zoom_in_bottom_align': 0.10,

    'muse_ha_zoom_in_size': (2.5, 2.5),
    'muse_ha_zoom_in_res': 'copt',
    'muse_ha_zoom_in_ssp_model': 'fiducial',
    'muse_ha_zoom_in_map_typ': 'HA6562_FLUX',
    'muse_scale_bar_length_1': 1,
    'muse_scale_bar_length_2': 30,

    # plotting parameters
    'hst_broad_band_color': 'tab:blue',
    'hst_ha_color': 'tab:red',
    'nircam_color': 'tab:green',
    'miri_color': 'tab:purple',

    'overview_title_font_size': 50,
    'overview_label_size': 40,
    'zoom_in_title_font_size': 40,
    'zoom_in_label_size': 40,
    'stamp_title_font_size': 40,
    'stamp_label_size': 30,
    'ha_ew_title_font_size': 40,
    'ha_ew_label_size': 30,
    'sed_title_font_size': 40,
    'sed_label_size': 30,
    'muse_spec_title_font_size': 35,
    'muse_spec_label_size': 30,


}


holistic_viewer2_param_dic = {

    # image proportions:
    'env_cutout_size': (10, 10),

    # Figure and axis parameters
    'fig_size': (45, 60),


    # overview image
    # general parameters
    'overview_red_band': 'I',
    'overview_green_band': 'V',
    'overview_blue_band': 'B',
    'overview_img_pixel_size': (500, 500),
    # limits for the overview image
    'overview_width': 0.34,
    'overview_height': 0.34,
    'overview_left_align': 0.035,
    'overview_bottom_align': 0.69,
    # parameters for RGB image
    'overview_img_params': {
        'color_r': '#FF4433',
        'color_g': '#0FFF50',
        'color_b': '#1F51FF',
        'min_max_r': (0.3, 99.5),
        'min_max_g': (0.3, 99.5),
        'min_max_b': (0.3, 99.5),
        'gamma_r': 17.5,
        'gamma_g': 17.5,
        'gamma_b': 17.5,
        'gamma_corr_r': 17.5,
        'gamma_corr_g': 17.5,
        'gamma_corr_b': 17.5,
        'combined_gamma': 17.5
    },
    'overview_scale_bar_length': 1,

    # zoom in panels
    # limits for the zoom in panels
    'zoom_in_width': 0.15,
    'zoom_in_height': 0.15,
    'zoom_in_left_align': 0.42,
    'zoom_in_bottom_align': 0.73,
    'zoom_in_space_horizontal': -0.03,
    'zoom_in_space_vertical': 0.005,
    # hst broad band image
    'hst_broad_band_red_band': 'I',
    'hst_broad_band_green_band': 'V',
    'hst_broad_band_blue_band': 'B',
    # hst h-alpha image
    'hst_ha_red_band': 'Ha',
    'hst_ha_green_band': 'B',
    'hst_ha_blue_band': 'U',
    # Nircam
    'nircam_red_band':  'F200W',
    'nircam_green_band': 'F164N',
    'nircam_blue_band': 'F150W',
    # Nircam2
    'nircam2_red_band': 'F200W',
    'nircam2_green_band': 'F187N',
    'nircam2_blue_band': 'F150W',
    # Nircam3
    'nircam3_red_band': 'F360M',
    'nircam3_green_band': 'F335M',
    'nircam3_blue_band': 'F300M',


    'individual_band_list': [
        # 'F275W', 'F336W', 'F555W', 'F814W',
                             'F658N',
                       'F150W', 'F164N', 'F187N', 'F200W',
                       #'F300M', 'F335M', 'F360M',
                        'F770W', 'F1000W', 'F1130W', 'F1500W', 'F1800W', 'F2100W'
                       ],


    # miri
    'miri_red_band': 'F1130W',
    'miri_green_band': 'F1000W',
    'miri_blue_band': 'F770W',
    # astrosat
    'astrosat_band': 'F148W',
    'astrosat_cbar_left_align': 0.92,
    'astrosat_cbar_bottom_align': 0.87,
    'astrosat_cbar_width': 0.02,
    'astrosat_cbar_height': 0.10,
    'astrosat_norm': 'log',
    'astrosat_cmap': 'Greys',

    # alma
    'alma_res': 150,
    'alma_alpha_co_method': 'B13_scaling',
    'alma_cbar_left_align': 0.92,
    'alma_cbar_bottom_align': 0.75,
    'alma_cbar_width': 0.02,
    'alma_cbar_height': 0.10,
    'alma_norm': 'log',
    'alma_cmap': 'inferno',
    'zoom_in_scale_bar_length_1': 100,
    'zoom_in_scale_bar_length_2': 5,


    # limits for the stamps
    'stamp_width': 0.1,
    'stamp_height': 0.1,
    'stamp_left_align': 0.035,
    'stamp_bottom_align': -0.01,
    'stamp_space_horizontal': 0.05,
    'stamp_space_vertical': 0.005,

    'rad_pro_width': 0.1,
    'rad_pro_height': 0.045,
    'rad_pro_left_align': 0.035,
    'rad_pro_bottom_align': 0.095,
    'rad_pro_space_horizontal': 0.105,
    'rad_pro_space_vertical': 0.005,

    'stamp_size': (2.5, 2.5),
    'stamp_scale_bar_length_1': 1,
    'stamp_scale_bar_length_2': 30,

    # limits for the SED
    'sed_width': 0.94,
    'sed_height': 0.13,
    'sed_left_align': 0.035,
    'sed_bottom_align': 0.02,
    'sed_size_hst': (4, 4),
    'sed_size_nircam': (6, 6),
    'sed_size_miri': (7, 7),


    # plotting parameters
    'hst_broad_band_color': 'tab:blue',
    'hst_ha_color': 'tab:red',
    'nircam_color': 'tab:green',
    'miri_color': 'tab:purple',

    'overview_title_font_size': 50,
    'overview_label_size': 40,
    'zoom_in_title_font_size': 40,
    'zoom_in_label_size': 40,
    'stamp_title_font_size': 40,
    'stamp_label_size': 30,
    'ha_ew_title_font_size': 40,
    'ha_ew_label_size': 30,
    'sed_title_font_size': 40,
    'sed_label_size': 30,


}

phot_viewer_param_dic = {

    # image proportions:
    'env_cutout_size': (10, 10),

    # Figure and axis parameters
    'fig_size': (45, 60),

    # overview image
    # general parameters
    'overview_red_band': 'I',
    'overview_green_band': 'V',
    'overview_blue_band': 'B',
    'overview_img_pixel_size': (500, 500),
    # limits for the overview image
    'overview_width': 0.35,
    'overview_height': 0.35,
    'overview_left_align': 0.035,
    'overview_bottom_align': 0.68,
    # parameters for RGB image
    'overview_img_params': {
        'color_r': '#FF4433',
        'color_g': '#0FFF50',
        'color_b': '#1F51FF',
        'min_max_r': (0.3, 99.5),
        'min_max_g': (0.3, 99.5),
        'min_max_b': (0.3, 99.5),
        'gamma_r': 17.5,
        'gamma_g': 17.5,
        'gamma_b': 17.5,
        'gamma_corr_r': 17.5,
        'gamma_corr_g': 17.5,
        'gamma_corr_b': 17.5,
        'combined_gamma': 17.5
    },
    'overview_scale_bar_length': 1,

    #
    # limits for the stamps and also BKG
    'stamp_width': 0.1,
    'stamp_height': 0.1,
    'stamp_left_align': 0.03,
    'stamp_bottom_align': 0.07,
    'stamp_space_horizontal': 0.08,
    'stamp_space_vertical': 0.015,

    'rad_pro_width': 0.1,
    'rad_pro_height': 0.06,
    'rad_pro_left_align': 0.03,
    'rad_pro_bottom_align': -0.01,
    'rad_pro_space_horizontal': 0.12,
    'rad_pro_space_vertical': 0.015,

    'cbar_width': 0.1,
    'cbar_height': 0.01,
    # 'cbar_left_offset': 0.01,
    'cbar_left_align': 0.03,
    'cbar_bottom_align': 0.07,
    'cbar_space_horizontal': 0.1,
    'cbar_space_vertical': 0.015,

    'stamp_size_hst': (2.5, 2.5),
    'stamp_size_nircam': (2.5, 2.5),
    'stamp_size_miri': (5, 5),

    'stamp_scale_bar_length_hst': 1,
    'stamp_scale_bar_length_nircam': 1,
    'stamp_scale_bar_length_miri': 2,

    'stamp_scale_bar_length_pc_hst': 30,
    'stamp_scale_bar_length_pc_nircam': 30,
    'stamp_scale_bar_length_pc_miri': 50,

    'max_rad_profile_hst_arcsec': 1,
    'max_rad_profile_nircam_arcsec': 1,
    'max_rad_profile_miri_arcsec': 3,
    'n_profile_slits': 12,

    'bkg_img_size_factor_hst': 40,
    'box_size_factor_hst': 2,
    'filter_size_factor_hst': 1,

    'bkg_img_size_factor_nircam': 40,
    'box_size_factor_nircam': 2,
    'filter_size_factor_nircam': 1,

    'bkg_img_size_factor_miri': 40,
    'box_size_factor_miri': 2,
    'filter_size_factor_miri': 1,

    # zoom in panels
    # limits for the zoom in panels
    'zoom_in_width': 0.15,
    'zoom_in_height': 0.15,
    'zoom_in_left_align': 0.42,
    'zoom_in_bottom_align': 0.72,
    'zoom_in_space_horizontal': -0.02,
    'zoom_in_space_vertical': 0.005,
    # hst broad band image
    'hst_broad_band_red_band': 'I',
    'hst_broad_band_green_band': 'V',
    'hst_broad_band_blue_band': 'B',
    # hst h-alpha image
    'hst_ha_red_band': 'Ha',
    'hst_ha_green_band': 'B',
    'hst_ha_blue_band': 'U',
    # Nircam
    'nircam_red_band': 'F300M',
    'nircam_green_band': 'F335M',
    'nircam_blue_band': 'F200W',
    # miri
    'miri_red_band': 'F1130W',
    'miri_green_band': 'F1000W',
    'miri_blue_band': 'F770W',
    # astrosat
    'astrosat_band': 'F148W',
    'astrosat_cbar_left_align': 0.92,
    'astrosat_cbar_bottom_align': 0.87,
    'astrosat_cbar_width': 0.02,
    'astrosat_cbar_height': 0.10,
    'astrosat_norm': 'log',
    'astrosat_cmap': 'Greys',

    # alma
    'alma_res': 150,
    'alma_alpha_co_method': 'S20_MUSEGPR',
    'alma_cbar_left_align': 0.92,
    'alma_cbar_bottom_align': 0.75,
    'alma_cbar_width': 0.02,
    'alma_cbar_height': 0.10,
    'alma_norm': 'log',
    'alma_cmap': 'inferno',
    'zoom_in_scale_bar_length_1': 100,
    'zoom_in_scale_bar_length_2': 5,




    # limits for the SED
    'sed_width': 0.9,
    'sed_height': 0.13,
    'sed_left_align': 0.035,
    'sed_bottom_align': 0.02,
    'sed_size': (20, 20),
    'hst_broad_band_ap_rad_pix_list': [1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12],
    'hst_broad_band_annulus_rad_in_pix': 13,
    'hst_broad_band_annulus_rad_out_pix': 15,

    'hst_ha_ap_rad_pix_list': [1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12],
    'hst_ha_annulus_rad_in_pix': 13,
    'hst_ha_annulus_rad_out_pix': 15,

    'nircam_ap_rad_pix_list': [1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12],
    'nircam_annulus_rad_in_pix': 13,
    'nircam_annulus_rad_out_pix': 15,

    'miri_ap_rad_pix_list': [1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12],
    'miri_annulus_rad_in_pix': 13,
    'miri_annulus_rad_out_pix': 15,


    # plotting parameters
    'hst_broad_band_color': 'tab:blue',
    'hst_ha_color': 'tab:red',
    'nircam_color': 'tab:green',
    'miri_color': 'tab:purple',

    'overview_title_font_size': 50,
    'overview_label_size': 40,
    'zoom_in_title_font_size': 40,
    'zoom_in_label_size': 40,
    'stamp_title_font_size': 40,
    'stamp_label_size': 30,
    'ha_ew_title_font_size': 40,
    'ha_ew_label_size': 30,
    'sed_title_font_size': 40,
    'sed_label_size': 30,


}

ism_phot_viewer_param_dict = {
    # Figure and axis parameters
    'fig_size': (40, 50),

    'bkg_cutout_size': (40, 40),
    'obj_cutout_size': (5, 5),

    # limits for the stamps
    'bkg_env_width': 0.23,
    'bkg_env_height': 0.23,
    'bkg_env_left_align': 0.04,
    'bkg_env_bottom_align': 0.39,
    'bkg_env_space_horizontal': -0.035,
    'bkg_env_space_vertical': 0.005,
    'bkg_env_title_font_size': 40,
    'bkg_env_label_size': 30,

    'bkg_1_box_size': (20, 20),
    'bkg_2_box_size': (5, 5),

    'zoom_in_scale_bar_length_1': 300,
    'zoom_in_scale_bar_length_2': 5,


    # limits for the stamps
    'stamp_width': 0.11,
    'stamp_height': 0.11,
    'rad_prof_height': 0.09,
    'stamp_left_align': 0.04,
    'stamp_bottom_align': 0.29,
    'rad_prof_bottom_align': 0.30,
    'stamp_space_horizontal': 0.005,
    'stamp_space_vertical': 0.005,

    'stamp_scale_bar_length_1': 1,

    # limits for the SED

    'sed_width': 0.9,
    'sed_height': 0.15,
    'sed_left_align': 0.06,
    'sed_bottom_align': 0.02,

    'miri_ap_rad_pix_list': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'miri_annulus_rad_in_pix': 13,
    'miri_annulus_rad_out_pix': 15,

    'miri_color': 'tab:purple',
    'zoom_in_title_font_size': 40,
    'zoom_in_label_size': 40,
    'stamp_title_font_size': 40,
    'stamp_label_size': 30,
    'sed_title_font_size': 40,
    'sed_label_size': 30,

}

muse_spec_viewer_param_dict = {
    # Figure and axis parameters
    'fig_size': (45, 30),


    'muse_spec_width': 0.80,
    'muse_spec_height': 0.25,
    'muse_spec_left_align': 0.035,
    'muse_spec_bottom_align': 0.05,

    'hb_oiii_width': 0.40,
    'hb_oiii_height': 0.25,
    'hb_oiii_left_align': 0.035,
    'hb_oiii_bottom_align': 0.73,

    'hb_oiii_res_width': 0.40,
    'hb_oiii_res_height': 0.05,
    'hb_oiii_res_left_align': 0.035,
    'hb_oiii_res_bottom_align': 0.675,

    'hb_oiii_left_lim_offst': 15,
    'hb_oiii_right_lim_offst': 15,

    'ha_nii_width': 0.40,
    'ha_nii_height': 0.25,
    'ha_nii_left_align': 0.035,
    'ha_nii_bottom_align': 0.4,

    'ha_nii_res_width': 0.40,
    'ha_nii_res_height': 0.05,
    'ha_nii_res_left_align': 0.035,
    'ha_nii_res_bottom_align': 0.345,

    'ha_nii_left_lim_offst': 15,
    'ha_nii_right_lim_offst': 15,

    'red_bump_width': 0.20,
    'red_bump_height': 0.25,
    'red_bump_left_align': 0.46,
    'red_bump_bottom_align': 0.73,

    'red_bump_left_lim_offst': 15,
    'red_bump_right_lim_offst': 15,

    'oi6302_width': 0.20,
    'oi6302_height': 0.25,
    'oi6302_left_align': 0.685,
    'oi6302_bottom_align': 0.73,

    'oi6302_res_width': 0.20,
    'oi6302_res_height': 0.05,
    'oi6302_res_left_align': 0.685,
    'oi6302_res_bottom_align': 0.675,

    'oi6302_left_lim_offst': 15,
    'oi6302_right_lim_offst': 15,

    'hei6680_width': 0.20,
    'hei6680_height': 0.25,
    'hei6680_left_align': 0.46,
    'hei6680_bottom_align': 0.4,

    'hei6680_left_lim_offst': 15,
    'hei6680_right_lim_offst': 15,

    'sii_width': 0.25,
    'sii_height': 0.25,
    'sii_left_align': 0.685,
    'sii_bottom_align': 0.4,

    'sii_res_width': 0.25,
    'sii_res_height': 0.05,
    'sii_res_left_align': 0.685,
    'sii_res_bottom_align': 0.345,

    'sii_left_lim_offst': 15,
    'sii_right_lim_offst': 15,

    # 'muse_spec_x_lim': ('min', 7100),
    'muse_spec_x_lim': ('min', 'max'),
    'muse_spec_y_lim': 'cont',


    'muse_ha_zoom_in_width': 0.12,
    'muse_ha_zoom_in_height': 0.12,
    'muse_ha_zoom_in_left_align': 0.81,
    'muse_ha_zoom_in_bottom_align': 0.21,

    'muse_ha_zoom_in_size': (2.5, 2.5),
    'muse_ha_zoom_in_res': 'copt',
    'muse_ha_zoom_in_ssp_model': 'fiducial',
    'muse_ha_zoom_in_map_typ': 'HA6562_FLUX',
    'muse_scale_bar_length_1': 1,
    'muse_scale_bar_length_2': 30,
    'muse_spec_title_font_size': 35,
    'muse_spec_label_size': 30,
    'em_title_font_size': 35,
    'em_label_size': 30,

}


fit_spec_viewer_param_dict = {
    # Figure and axis parameters
    'fig_size': (45, 30),


    'muse_spec_width': 0.80,
    'muse_spec_height': 0.25,
    'muse_spec_left_align': 0.035,
    'muse_spec_bottom_align': 0.05,



    'hb_oiii_width': 0.40,
    'hb_oiii_height': 0.25,
    'hb_oiii_left_align': 0.035,
    'hb_oiii_bottom_align': 0.73,

    'hb_oiii_res_width': 0.40,
    'hb_oiii_res_height': 0.05,
    'hb_oiii_res_left_align': 0.035,
    'hb_oiii_res_bottom_align': 0.675,

    'hb_oiii_left_lim_offst': 15,
    'hb_oiii_right_lim_offst': 15,


    'ha_nii_width': 0.40,
    'ha_nii_height': 0.25,
    'ha_nii_left_align': 0.48,
    'ha_nii_bottom_align': 0.73,

    'ha_nii_res_width': 0.40,
    'ha_nii_res_height': 0.05,
    'ha_nii_res_left_align': 0.48,
    'ha_nii_res_bottom_align': 0.675,

    'ha_nii_left_lim_offst': 15,
    'ha_nii_right_lim_offst': 15,



    'sii_width': 0.25,
    'sii_height': 0.25,
    'sii_left_align': 0.035,
    'sii_bottom_align': 0.4,

    'sii_res_width': 0.25,
    'sii_res_height': 0.05,
    'sii_res_left_align': 0.035,
    'sii_res_bottom_align': 0.345,

    'sii_left_lim_offst': 15,
    'sii_right_lim_offst': 15,



    'red_bump_width': 0.20,
    'red_bump_height': 0.25,
    'red_bump_left_align': 0.46,
    'red_bump_bottom_align': 0.73,

    'red_bump_left_lim_offst': 15,
    'red_bump_right_lim_offst': 15,

    'oi6302_width': 0.20,
    'oi6302_height': 0.25,
    'oi6302_left_align': 0.685,
    'oi6302_bottom_align': 0.73,

    'oi6302_res_width': 0.20,
    'oi6302_res_height': 0.05,
    'oi6302_res_left_align': 0.685,
    'oi6302_res_bottom_align': 0.675,

    'oi6302_left_lim_offst': 15,
    'oi6302_right_lim_offst': 15,

    'hei6680_width': 0.20,
    'hei6680_height': 0.25,
    'hei6680_left_align': 0.46,
    'hei6680_bottom_align': 0.4,

    'hei6680_left_lim_offst': 15,
    'hei6680_right_lim_offst': 15,


    # 'muse_spec_x_lim': ('min', 7100),
    'muse_spec_x_lim': ('min', 'max'),
    'muse_spec_y_lim': 'cont',


    'muse_ha_zoom_in_width': 0.12,
    'muse_ha_zoom_in_height': 0.12,
    'muse_ha_zoom_in_left_align': 0.81,
    'muse_ha_zoom_in_bottom_align': 0.21,

    'muse_ha_zoom_in_size': (2.5, 2.5),
    'muse_ha_zoom_in_res': 'copt',
    'muse_ha_zoom_in_ssp_model': 'fiducial',
    'muse_ha_zoom_in_map_typ': 'HA6562_FLUX',
    'muse_scale_bar_length_1': 1,
    'muse_scale_bar_length_2': 30,
    'muse_spec_title_font_size': 35,
    'muse_spec_label_size': 30,
    'em_title_font_size': 35,
    'em_label_size': 30,

}