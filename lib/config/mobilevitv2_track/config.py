from easydict import EasyDict as edict
import yaml

"""
Add default config for MobileViT-V2-Track.
This config is now updated to support Domain Adaptation.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.PRETRAIN_FILE = "mobilevitv2-1.0.pt"
cfg.MODEL.EXTRA_MERGER = False

cfg.MODEL.RETURN_INTER = False
cfg.MODEL.RETURN_STAGES = []

# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "mobilevitv2"
cfg.MODEL.BACKBONE.STRIDE = 16
cfg.MODEL.BACKBONE.MID_PE = False
cfg.MODEL.BACKBONE.SEP_SEG = False
cfg.MODEL.BACKBONE.CAT_MODE = 'direct'
cfg.MODEL.BACKBONE.MERGE_LAYER = 0
cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = False
cfg.MODEL.BACKBONE.CLS_TOKEN_USE_MODE = 'ignore'
cfg.MODEL.BACKBONE.MIXED_ATTN = True

# MODEL.NECK
cfg.MODEL.NECK = edict()
cfg.MODEL.NECK.TYPE = 'BN_FEATURE_FUSOR_LIGHTTRACK'
cfg.MODEL.NECK.NUM_CHANNS_POST_XCORR = 64

# MODEL.HEAD
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = "CENTER"
cfg.MODEL.HEAD.NUM_CHANNELS = 256

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.FREEZE_LAYERS = [0, ]
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.AMP = False
cfg.TRAIN.DOMAIN_WEIGHT = 0.1  # <-- 新增：领域损失权重的默认值

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1

# DATA
cfg.DATA = edict()
cfg.DATA.SAMPLER_MODE = "causal"  # sampling methods
cfg.DATA.MEAN = [0.0, 0.0, 0.0]
cfg.DATA.STD = [1.0, 1.0, 1.0]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200

# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000

# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["GOT10K_votval"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000

# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 320
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
cfg.DATA.SEARCH.NUMBER = 1

# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# --- 新增：为 YAML 文件中使用的所有数据集添加默认配置块 ---
cfg.DATA.TrackingNet = edict()
cfg.DATA.TrackingNet.ROOT = ''
cfg.DATA.TrackingNet.ANNO = ''
cfg.DATA.TrackingNet.FRAME_RANGE = 100
cfg.DATA.TrackingNet.NUM_USE = 100000
cfg.DATA.TrackingNet.DOMAIN = 'source'

cfg.DATA.VID = edict()
cfg.DATA.VID.ROOT = ''
cfg.DATA.VID.ANNO = ''
cfg.DATA.VID.FRAME_RANGE = 100
cfg.DATA.VID.NUM_USE = 100000
cfg.DATA.VID.DOMAIN = 'target'

cfg.DATA.WATB400_1 = edict()
cfg.DATA.WATB400_1.ROOT = ''
cfg.DATA.WATB400_1.ANNO = ''
cfg.DATA.WATB400_1.NUM_USE = -1
cfg.DATA.WATB400_1.DOMAIN = 'target'

cfg.DATA.WATB400_2 = edict()
cfg.DATA.WATB400_2.ROOT = ''
cfg.DATA.WATB400_2.ANNO = ''
cfg.DATA.WATB400_2.NUM_USE = -1
cfg.DATA.WATB400_2.DOMAIN = 'target' # Default to source, can be overridden by yaml

cfg.DATA.GOT10K_official_val = edict()
# Add necessary fields for validation dataset if any
# This is just a placeholder to prevent errors.
cfg.DATA.GOT10K_official_val.ROOT = ''
cfg.DATA.GOT10K_official_val.ANNO = ''

# TEST
cfg.TEST = edict()
cfg.TEST.DEVICE = "cpu"
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.SEARCH_FACTOR = 5.0
cfg.TEST.SEARCH_SIZE = 320
cfg.TEST.EPOCH = 500


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename, base_cfg=None):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            _update_config(cfg, exp_config)
