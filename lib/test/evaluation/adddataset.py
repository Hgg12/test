import os
import numpy as np
import json

from typing import Dict, Any

from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class AddDataset(BaseDataset):
    """ OTB-2015 dataset
    Publication:
        Object Tracking Benchmark
        Wu, Yi, Jongwoo Lim, and Ming-hsuan Yan
        TPAMI, 2015
        http://faculty.ucmerced.edu/mhyang/papers/pami15_tracking_benchmark.pdf
    Download the dataset from http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.add_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path, 
        sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        # NOTE: OTB has some weird annos which panda cannot handle
        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')

        # ground_truth_rect[:, 2] = ground_truth_rect[:, 2] - ground_truth_rect[:, 0]
        # ground_truth_rect[:, 3] = ground_truth_rect[:, 3] - ground_truth_rect[:, 1]

        return Sequence(sequence_info['name'], frames, 'test', ground_truth_rect[init_omit:,:],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {"name": "000104", "path": "000104/img", "startFrame": 1, "endFrame": 4044, "nz": 6,
             "ext": "jpg", "anno_path": "000104/groundtruth.txt",
             "object_class": "other"},
            {"name": "000105", "path": "000105/img", "startFrame": 1, "endFrame": 4970, "nz": 6,
             "ext": "jpg", "anno_path": "000105/groundtruth.txt",
             "object_class": "other"},
            {"name": "000106", "path": "000106/img", "startFrame": 1, "endFrame": 595, "nz": 6,
             "ext": "jpg", "anno_path": "000106/groundtruth.txt",
             "object_class": "other"},
            {"name": "000107", "path": "000107/img", "startFrame": 1, "endFrame": 696, "nz": 6,
             "ext": "jpg", "anno_path": "000107/groundtruth.txt",
             "object_class": "other"},
            {"name": "000108", "path": "000108/img", "startFrame": 1, "endFrame": 364, "nz": 6,
             "ext": "jpg", "anno_path": "000108/groundtruth.txt",
             "object_class": "other"},
            {"name": "000109", "path": "000109/img", "startFrame": 1, "endFrame": 496, "nz": 6,
             "ext": "jpg", "anno_path": "000109/groundtruth.txt",
             "object_class": "other"},
            {"name": "000110", "path": "000110/img", "startFrame": 1, "endFrame": 2440, "nz": 6,
             "ext": "jpg", "anno_path": "000110/groundtruth.txt",
             "object_class": "other"}
        ]
    
        return sequence_info_list