from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/media/w/dataset/got10k_lmdb'
    settings.got10k_path = '/media/w/dataset/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/media/w/dataset/itb'
    settings.lasot_extension_subset_path_path = '/media/w/dataset/lasot_extension_subset'
    settings.lasot_lmdb_path = '/media/w/dataset/lasot_lmdb'
    settings.lasot_path = '/home/w/hgg/lasot/lasot'
    settings.network_path = '/media/w/dataset/Experiments/SMAT/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/media/w/dataset/nfs'
    settings.otb_path = '/media/w/dataset/otb'
    settings.prj_dir = '/media/w/dataset/Experiments/SMAT'
    settings.result_plot_path = '/media/w/dataset/Experiments/SMAT/output/test/result_plots'
    settings.results_path = '/media/w/dataset/Experiments/SMAT/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/media/w/dataset/Experiments/SMAT/output'
    settings.segmentation_path = '/media/w/dataset/SMAT/output/test/segmentation_results'
    settings.tc128_path = '/media/w/dataset/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/media/w/dataset/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/media/w/dataset/trackingnet'
    settings.uav_path = '/media/w/dataset/uav'
    settings.vot18_path = '/media/w/dataset/vot2018'
    settings.vot22_path = '/media/w/dataset/vot2022'
    settings.vot_path = '/media/w/dataset/VOT2019'
    settings.youtubevos_dir = ''
    settings.wildlife2024_path = '/media/w/dataset/Wildlife2024_test'
    settings.watb_path = r'/media/w/719A549756118C56/datasets/WATB/WATB'
    settings.add_path = r'/media/w/dataset/test'
    settings.tlp_path = '/media/w/dataset/TLP/seq'

    return settings

