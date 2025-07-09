class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/media/w/dataset/SMAT'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/media/w/dataset/SMAT/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/media/w/dataset/SMAT/pretrained_networks'
        self.lasot_dir = '/home/w/hgg/lasot/lasot'
        self.got10k_dir = '/home/w/hgg/GOT-10k/train'
        self.got10k_val_dir = '/home/w/hgg/GOT-10k'
        self.lasot_lmdb_dir = ''
        self.got10k_lmdb_dir = ''
        self.trackingnet_dir = '/media/w/719A549756118C56/datasets/TrackingNet'
        self.trackingnet_lmdb_dir = ''
        self.coco_dir = '/home/w/hgg/CoCo'
        self.coco_lmdb_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''