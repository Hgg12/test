import torch
import random
from torch.utils.data.distributed import DistributedSampler

from lib.train.data.sampler import TrackingSampler

class TrackingDataset(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches.
    Modified to support domain adaptation by providing paired source and target samples.
    """
    def __init__(self, datasets, p_datasets, samples_per_epoch, settings, cfg, processing_fn):
        self.datasets = datasets
        self.p_datasets = p_datasets
        self.samples_per_epoch = samples_per_epoch
        self.settings = settings
        self.cfg = cfg
        self.processing_fn = processing_fn

        self.source_list = []
        self.target_list = []

        for i, d in enumerate(self.datasets):
            domain = getattr(self.cfg.DATA, d.name, {}).get('DOMAIN', 'source')
            samples = [(i, sz) for sz in range(d.get_num_sequences())]
            
            if domain == 'source':
                self.source_list.extend(samples)
            elif domain == 'target':
                self.target_list.extend(samples)
            else:
                self.source_list.extend(samples)

        if not self.target_list:
            print("Warning: No target domain datasets found. Using source domain data for the target branch.")
            self.target_list = self.source_list
        
        self.num_samples = len(self.source_list) if self.source_list else 0
        print(f"Dataset initialized: {len(self.source_list)} source samples, {len(self.target_list)} target samples.")

        self.sampler = TrackingSampler(datasets=self.datasets, p_datasets=self.p_datasets,
                                       samples_per_epoch=self.samples_per_epoch)

    def __len__(self):
        return self.num_samples

    def _get_sample(self, dataset_idx, seq_idx):
        dataset = self.datasets[dataset_idx]
        seq_info = dataset.get_sequence_info(seq_idx)
        
        template_frame_ids, search_frame_ids = self.sampler._sample_frames(seq_info['valid'], seq_info['visible'])
        
        template_frame, template_anno, _ = dataset.get_frame(seq_idx, template_frame_ids[0])
        search_frame, search_anno, _ = dataset.get_frame(seq_idx, search_frame_ids[0])

        data = {}
        data['template_images'], data['template_anno'] = self.processing_fn(template_frame, template_anno, 'template')
        data['search_images'], data['search_anno'] = self.processing_fn(search_frame, search_anno, 'search')

        return data

    def __getitem__(self, index):
        s_dataset_idx, s_seq_idx = self.source_list[index % len(self.source_list)]
        source_data = self._get_sample(s_dataset_idx, s_seq_idx)

        t_index = random.randint(0, len(self.target_list) - 1)
        t_dataset_idx, t_seq_idx = self.target_list[t_index]
        target_data = self._get_sample(t_dataset_idx, t_seq_idx)

        output = {
            'template_s': source_data['template_images'],
            'search_s': source_data['search_images'],
            'template_anno_s': source_data['template_anno'],
            'search_anno_s': source_data['search_anno'],
            'template_t': target_data['template_images'],
            'search_t': target_data['search_images'],
        }
        return output


class LTRLoader(torch.utils.data.DataLoader):
    def __init__(self, name, dataset, training=True, batch_size=1, shuffle=False,
                 num_workers=0, drop_last=False, stack_dim=0, pin_memory=True):
        
        super(LTRLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                        num_workers=num_workers, drop_last=drop_last, pin_memory=pin_memory)
        self.name = name
        self.stack_dim = stack_dim

    def __iter__(self):
        for d in super().__iter__():
            yield d


def create_loader(mode, settings, cfg, Processing_class):
    """ Creates a data loader. """
    from lib.train.dataset import Lasot, Got10k, TrackingNet, MSCOCO, Got10k_lmdb, Lasot_lmdb, ImagenetVID, \
        CocoSeq, ImagenetVID_lmdb, CocoSeq_lmdb, TrackingNet_lmdb, WATB400_1, WATB400_2, GOT10K_official_val

    dataset_classes = {
        'LASOT': Lasot, 'GOT10K': Got10k, 'TrackingNet': TrackingNet, 'MSCOCO': MSCOCO,
        'GOT10K_LMDB': Got10k_lmdb, 'LASOT_LMDB': Lasot_lmdb, 'VID': ImagenetVID,
        'IMAGENETVID': ImagenetVID, 'COCOSEQ': CocoSeq, 'IMAGENETVID_LMDB': ImagenetVID_lmdb,
        'COCOSEQ_LMDB': CocoSeq_lmdb, 'TRACKINGNET_LMDB': TrackingNet_lmdb,
        'WATB400_1': WATB400_1, 'WATB400_2': WATB400_2, 'GOT10K_official_val': GOT10K_official_val,
    }

    proc_fn = Processing_class(
        search_area_factor={'template': cfg.DATA.TEMPLATE.FACTOR, 'search': cfg.DATA.SEARCH.FACTOR},
        output_sz={'template': cfg.DATA.TEMPLATE.SIZE, 'search': cfg.DATA.SEARCH.SIZE},
        center_jitter_factor={'template': cfg.DATA.TEMPLATE.CENTER_JITTER, 'search': cfg.DATA.SEARCH.CENTER_JITTER},
        scale_jitter_factor={'template': cfg.DATA.TEMPLATE.SCALE_JITTER, 'search': cfg.DATA.SEARCH.SCALE_JITTER},
        mode='sequence',
        settings=settings)

    dataset_names = cfg.DATA.TRAIN.DATASETS_NAME
    dataset_ratios = cfg.DATA.TRAIN.DATASETS_RATIO
    
    datasets = []
    for name in dataset_names:
        if name not in dataset_classes:
            raise KeyError(f"Dataset '{name}' is specified in config, but not defined in loader.py's dataset_classes.")
        
        dataset_class = dataset_classes[name]
        
        # --- 已修正：同时获取 ROOT 和 ANNO 路径 ---
        dataset_cfg_node = getattr(cfg.DATA, name)
        root_path = dataset_cfg_node.ROOT
        anno_path = dataset_cfg_node.ANNO
        
        # --- 已修正：将 anno_path 传递给数据集构造函数 ---
        datasets.append(dataset_class(root=root_path, anno_path=anno_path))

    dataset = TrackingDataset(datasets, dataset_ratios, cfg.TRAIN.SAMPLE_PER_EPOCH, settings, cfg, proc_fn)

    loader = LTRLoader(cfg.DATA.TRAIN.DATASETS_NAME, dataset, training=True,
                       batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.TRAIN.NUM_WORKERS,
                       shuffle=True, drop_last=True, stack_dim=0)
    
    return [loader]
