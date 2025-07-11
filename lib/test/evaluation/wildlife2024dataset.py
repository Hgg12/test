import os
import numpy as np
import json

from typing import Dict, Any

from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class Wildlife2024Dataset(BaseDataset):
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
        self.base_path = self.env_settings.wildlife2024_path
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

        return Sequence(sequence_info['name'], frames, 'Wildlife2024', ground_truth_rect[init_omit:,:],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {"name": "000001", "path": "000001/img", "startFrame": 1,
             "endFrame": 486, "nz": 6, "ext": "jpg",
             "anno_path": "000001/groundtruth.txt",
             "object_class": "other"},
            {"name": "000002", "path": "000002/img", "startFrame": 1,
             "endFrame": 414, "nz": 6, "ext": "jpg",
             "anno_path": "000002/groundtruth.txt",
             "object_class": "other"},
            {"name": "000003", "path": "000003/img", "startFrame": 1, "endFrame": 644, "nz": 6,
             "ext": "jpg", "anno_path": "000003/groundtruth.txt",
             "object_class": "other"},
            {"name": "000004", "path": "000004/img", "startFrame": 1,
             "endFrame": 550, "nz": 6, "ext": "jpg",
             "anno_path": "000004/groundtruth.txt",
             "object_class": "other"},
            {"name": "000005", "path": "000005/img", "startFrame": 1, "endFrame": 744,
             "nz": 6, "ext": "jpg", "anno_path": "000005/groundtruth.txt",
             "object_class": "other"},
            {"name": "000006", "path": "000006/img", "startFrame": 1, "endFrame": 666, "nz": 6,
             "ext": "jpg", "anno_path": "000006/groundtruth.txt",
             "object_class": "other"},
            {"name": "000007", "path": "000007/img", "startFrame": 1, "endFrame": 857,
             "nz": 6, "ext": "jpg", "anno_path": "000007/groundtruth.txt",
             "object_class": "other"},
            {"name": "000008", "path": "000008/img", "startFrame": 1, "endFrame": 392, "nz": 6,
             "ext": "jpg", "anno_path": "000008/groundtruth.txt",
             "object_class": "other"},
            {"name": "000009", "path": "000009/img", "startFrame": 1, "endFrame": 1000, "nz": 6,
             "ext": "jpg", "anno_path": "000009/groundtruth.txt",
             "object_class": "other"},
            {"name": "000010", "path": "000010/img", "startFrame": 1, "endFrame": 875, "nz": 6,
             "ext": "jpg", "anno_path": "000010/groundtruth.txt",
             "object_class": "other"},
            {"name": "000011", "path": "000011/img", "startFrame": 1, "endFrame": 599, "nz": 6,
             "ext": "jpg", "anno_path": "000011/groundtruth.txt",
             "object_class": "other"},
            {"name": "000012", "path": "000012/img", "startFrame": 1, "endFrame": 485, "nz": 6,
             "ext": "jpg", "anno_path": "000012/groundtruth.txt",
             "object_class": "other"},
            {"name": "000013", "path": "000013/img", "startFrame": 1, "endFrame": 598, "nz": 6,
             "ext": "jpg", "anno_path": "000013/groundtruth.txt",
             "object_class": "other"},
            {"name": "000014", "path": "000014/img", "startFrame": 1, "endFrame": 812, "nz": 6,
             "ext": "jpg", "anno_path": "000014/groundtruth.txt",
             "object_class": "other"},
            {"name": "000015", "path": "000015/img", "startFrame": 1, "endFrame": 819, "nz": 6,
             "ext": "jpg", "anno_path": "000015/groundtruth.txt",
             "object_class": "other"},
            {"name": "000016", "path": "000016/img", "startFrame": 1, "endFrame": 492, "nz": 6,
             "ext": "jpg", "anno_path": "000016/groundtruth.txt",
             "object_class": "other"},
            {"name": "000017", "path": "000017/img", "startFrame": 1, "endFrame": 420, "nz": 6,
             "ext": "jpg", "anno_path": "000017/groundtruth.txt",
             "object_class": "other"},
            {"name": "000018", "path": "000018/img", "startFrame": 1, "endFrame": 788, "nz": 6,
             "ext": "jpg", "anno_path": "000018/groundtruth.txt",
             "object_class": "other"},
            {"name": "000019", "path": "000019/img", "startFrame": 1, "endFrame": 501, "nz": 6,
             "ext": "jpg", "anno_path": "000019/groundtruth.txt",
             "object_class": "other"},
            {"name": "000020", "path": "000020/img", "startFrame": 1, "endFrame": 1594, "nz": 6,
             "ext": "jpg", "anno_path": "000020/groundtruth.txt",
             "object_class": "other"},
            {"name": "000021", "path": "000021/img", "startFrame": 1, "endFrame": 1313, "nz": 6,
             "ext": "jpg", "anno_path": "000021/groundtruth.txt",
             "object_class": "other"},
            {"name": "000022", "path": "000022/img", "startFrame": 1, "endFrame": 1130, "nz": 6,
             "ext": "jpg", "anno_path": "000022/groundtruth.txt",
             "object_class": "other"},
            {"name": "000023", "path": "000023/img", "startFrame": 1, "endFrame": 503, "nz": 6,
             "ext": "jpg", "anno_path": "000023/groundtruth.txt",
             "object_class": "other"},
            {"name": "000024", "path": "000024/img", "startFrame": 1, "endFrame": 440, "nz": 6,
             "ext": "jpg", "anno_path": "000024/groundtruth.txt",
             "object_class": "other"},
            {"name": "000025", "path": "000025/img", "startFrame": 1, "endFrame": 934, "nz": 6,
             "ext": "jpg", "anno_path": "000025/groundtruth.txt",
             "object_class": "other"},
            {"name": "000026", "path": "000026/img", "startFrame": 1, "endFrame": 1238, "nz": 6,
             "ext": "jpg", "anno_path": "000026/groundtruth.txt",
             "object_class": "other"},
            {"name": "000027", "path": "000027/img", "startFrame": 1, "endFrame": 993, "nz": 6,
             "ext": "jpg", "anno_path": "000027/groundtruth.txt",
             "object_class": "other"},
            {"name": "000028", "path": "000028/img", "startFrame": 1, "endFrame": 600, "nz": 6,
             "ext": "jpg", "anno_path": "000028/groundtruth.txt",
             "object_class": "other"},
            {"name": "000029", "path": "000029/img", "startFrame": 1, "endFrame": 872, "nz": 6,
             "ext": "jpg", "anno_path": "000029/groundtruth.txt",
             "object_class": "other"},
            {"name": "000030", "path": "000030/img", "startFrame": 1, "endFrame": 1207, "nz": 6,
             "ext": "jpg", "anno_path": "000030/groundtruth.txt",
             "object_class": "other"},
            {"name": "000031", "path": "000031/img", "startFrame": 1, "endFrame": 549, "nz": 6,
             "ext": "jpg", "anno_path": "000031/groundtruth.txt",
             "object_class": "other"},
            {"name": "000032", "path": "000032/img", "startFrame": 1, "endFrame": 259, "nz": 6,
             "ext": "jpg", "anno_path": "000032/groundtruth.txt",
             "object_class": "other"},
            {"name": "000033", "path": "000033/img", "startFrame": 1, "endFrame": 865, "nz": 6,
             "ext": "jpg", "anno_path": "000033/groundtruth.txt",
             "object_class": "other"},
            {"name": "000034", "path": "000034/img", "startFrame": 1, "endFrame": 386, "nz": 6,
             "ext": "jpg", "anno_path": "000034/groundtruth.txt",
             "object_class": "other"},
            {"name": "000035", "path": "000035/img", "startFrame": 1, "endFrame": 794, "nz": 6,
             "ext": "jpg", "anno_path": "000035/groundtruth.txt",
             "object_class": "other"},
            {"name": "000036", "path": "000036/img", "startFrame": 1, "endFrame": 1133, "nz": 6,
             "ext": "jpg", "anno_path": "000036/groundtruth.txt",
             "object_class": "other"},
            {"name": "000037", "path": "000037/img", "startFrame": 1, "endFrame": 3892, "nz": 6,
             "ext": "jpg", "anno_path": "000037/groundtruth.txt",
             "object_class": "other"},
            {"name": "000038", "path": "000038/img", "startFrame": 1, "endFrame": 1695, "nz": 6,
             "ext": "jpg", "anno_path": "000038/groundtruth.txt",
             "object_class": "other"},
            {"name": "000039", "path": "000039/img", "startFrame": 1, "endFrame": 1338, "nz": 6,
             "ext": "jpg", "anno_path": "000039/groundtruth.txt",
             "object_class": "other"},
            {"name": "000040", "path": "000040/img", "startFrame": 1, "endFrame": 818, "nz": 6,
             "ext": "jpg", "anno_path": "000040/groundtruth.txt",
             "object_class": "other"},
            {"name": "000041", "path": "000041/img", "startFrame": 1, "endFrame": 840, "nz": 6,
             "ext": "jpg", "anno_path": "000041/groundtruth.txt",
             "object_class": "other"},
            {"name": "000042", "path": "000042/img", "startFrame": 1, "endFrame": 2228, "nz": 6,
             "ext": "jpg", "anno_path": "000042/groundtruth.txt",
             "object_class": "other"},
            {"name": "000043", "path": "000043/img", "startFrame": 1, "endFrame": 1611, "nz": 6,
             "ext": "jpg", "anno_path": "000043/groundtruth.txt",
             "object_class": "other"},
            {"name": "000044", "path": "000044/img", "startFrame": 1, "endFrame": 767, "nz": 6,
             "ext": "jpg", "anno_path": "000044/groundtruth.txt",
             "object_class": "other"},
            {"name": "000045", "path": "000045/img", "startFrame": 1, "endFrame": 1047, "nz": 6,
             "ext": "jpg", "anno_path": "000045/groundtruth.txt",
             "object_class": "other"},
            {"name": "000046", "path": "000046/img", "startFrame": 1, "endFrame": 558, "nz": 6,
             "ext": "jpg", "anno_path": "000046/groundtruth.txt",
             "object_class": "other"},
            {"name": "000047", "path": "000047/img", "startFrame": 1, "endFrame": 758, "nz": 6,
             "ext": "jpg", "anno_path": "000047/groundtruth.txt",
             "object_class": "other"},
            {"name": "000048", "path": "000048/img", "startFrame": 1, "endFrame": 510, "nz": 6,
             "ext": "jpg", "anno_path": "000048/groundtruth.txt",
             "object_class": "other"},
            {"name": "000049", "path": "000049/img", "startFrame": 1, "endFrame": 435, "nz": 6,
             "ext": "jpg", "anno_path": "000049/groundtruth.txt",
             "object_class": "other"},
            {"name": "000050", "path": "000050/img", "startFrame": 1, "endFrame": 831, "nz": 6,
             "ext": "jpg", "anno_path": "000050/groundtruth.txt",
             "object_class": "other"},
            {"name": "000051", "path": "000051/img", "startFrame": 1, "endFrame": 935, "nz": 6,
             "ext": "jpg", "anno_path": "000051/groundtruth.txt",
             "object_class": "other"},
            {"name": "000052", "path": "000052/img", "startFrame": 1, "endFrame": 405, "nz": 6,
             "ext": "jpg", "anno_path": "000052/groundtruth.txt",
             "object_class": "other"},
            {"name": "000053", "path": "000053/img", "startFrame": 1, "endFrame": 465, "nz": 6,
             "ext": "jpg", "anno_path": "000053/groundtruth.txt",
             "object_class": "other"},
            {"name": "000054", "path": "000054/img", "startFrame": 1, "endFrame": 570, "nz": 6,
             "ext": "jpg", "anno_path": "000054/groundtruth.txt",
             "object_class": "other"},
            {"name": "000055", "path": "000055/img", "startFrame": 1, "endFrame": 887, "nz": 6,
             "ext": "jpg", "anno_path": "000055/groundtruth.txt",
             "object_class": "other"},
            {"name": "000056", "path": "000056/img", "startFrame": 1, "endFrame": 753, "nz": 6,
             "ext": "jpg", "anno_path": "000056/groundtruth.txt",
             "object_class": "other"},
            {"name": "000057", "path": "000057/img", "startFrame": 1, "endFrame": 768, "nz": 6,
             "ext": "jpg", "anno_path": "000057/groundtruth.txt",
             "object_class": "other"},
            {"name": "000058", "path": "000058/img", "startFrame": 1, "endFrame": 647, "nz": 6,
             "ext": "jpg", "anno_path": "000058/groundtruth.txt",
             "object_class": "other"},
            {"name": "000059", "path": "000059/img", "startFrame": 1, "endFrame": 505, "nz": 6,
             "ext": "jpg", "anno_path": "000059/groundtruth.txt",
             "object_class": "other"},
            {"name": "000060", "path": "000060/img", "startFrame": 1, "endFrame": 1413, "nz": 6,
             "ext": "jpg", "anno_path": "000060/groundtruth.txt",
             "object_class": "other"},
            {"name": "000061", "path": "000061/img", "startFrame": 1, "endFrame": 431, "nz": 6,
             "ext": "jpg", "anno_path": "000061/groundtruth.txt",
             "object_class": "other"},
            {"name": "000062", "path": "000062/img", "startFrame": 1, "endFrame": 2477, "nz": 6,
             "ext": "jpg", "anno_path": "000062/groundtruth.txt",
             "object_class": "other"},
            {"name": "000063", "path": "000063/img", "startFrame": 1, "endFrame": 570, "nz": 6,
             "ext": "jpg", "anno_path": "000063/groundtruth.txt",
             "object_class": "other"},
            {"name": "000064", "path": "000064/img", "startFrame": 1, "endFrame": 499, "nz": 6,
             "ext": "jpg", "anno_path": "000064/groundtruth.txt",
             "object_class": "other"},
            {"name": "000065", "path": "000065/img", "startFrame": 1, "endFrame": 1171, "nz": 6,
             "ext": "jpg", "anno_path": "000065/groundtruth.txt",
             "object_class": "other"},
            {"name": "000066", "path": "000066/img", "startFrame": 1, "endFrame": 586, "nz": 6,
             "ext": "jpg", "anno_path": "000066/groundtruth.txt",
             "object_class": "other"},
            {"name": "000067", "path": "000067/img", "startFrame": 1, "endFrame": 888, "nz": 6,
             "ext": "jpg", "anno_path": "000067/groundtruth.txt",
             "object_class": "other"},
            {"name": "000068", "path": "000068/img", "startFrame": 1, "endFrame": 767, "nz": 6,
             "ext": "jpg", "anno_path": "000068/groundtruth.txt",
             "object_class": "other"},
            {"name": "000069", "path": "000069/img", "startFrame": 1, "endFrame": 788, "nz": 6,
             "ext": "jpg", "anno_path": "000069/groundtruth.txt",
             "object_class": "other"},
            {"name": "000070", "path": "000070/img", "startFrame": 1, "endFrame": 454, "nz": 6,
             "ext": "jpg", "anno_path": "000070/groundtruth.txt",
             "object_class": "other"},
            {"name": "000071", "path": "000071/img", "startFrame": 1, "endFrame": 899, "nz": 6,
             "ext": "jpg", "anno_path": "000071/groundtruth.txt",
             "object_class": "other"},
            {"name": "000072", "path": "000072/img", "startFrame": 1, "endFrame": 392, "nz": 6,
             "ext": "jpg", "anno_path": "000072/groundtruth.txt",
             "object_class": "other"},
            {"name": "000073", "path": "000073/img", "startFrame": 1, "endFrame": 521, "nz": 6,
             "ext": "jpg", "anno_path": "000073/groundtruth.txt",
             "object_class": "other"},
            {"name": "000074", "path": "000074/img", "startFrame": 1, "endFrame": 729, "nz": 6,
             "ext": "jpg", "anno_path": "000074/groundtruth.txt",
             "object_class": "other"},
            {"name": "000075", "path": "000075/img", "startFrame": 1, "endFrame": 1011, "nz": 6,
             "ext": "jpg", "anno_path": "000075/groundtruth.txt",
             "object_class": "other"},
            {"name": "000076", "path": "000076/img", "startFrame": 1, "endFrame": 837, "nz": 6,
             "ext": "jpg", "anno_path": "000076/groundtruth.txt",
             "object_class": "other"},
            {"name": "000077", "path": "000077/img", "startFrame": 1, "endFrame": 573, "nz": 6,
             "ext": "jpg", "anno_path": "000077/groundtruth.txt",
             "object_class": "other"},
            {"name": "000078", "path": "000078/img", "startFrame": 1, "endFrame": 608, "nz": 6,
             "ext": "jpg", "anno_path": "000078/groundtruth.txt",
             "object_class": "other"},
            {"name": "000079", "path": "000079/img", "startFrame": 1, "endFrame": 719, "nz": 6,
             "ext": "jpg", "anno_path": "000079/groundtruth.txt",
             "object_class": "other"},
            {"name": "000080", "path": "000080/img", "startFrame": 1, "endFrame": 719, "nz": 6,
             "ext": "jpg", "anno_path": "000080/groundtruth.txt",
             "object_class": "other"},
            {"name": "000081", "path": "000081/img", "startFrame": 1, "endFrame": 719, "nz": 6,
             "ext": "jpg", "anno_path": "000081/groundtruth.txt",
             "object_class": "other"},
            {"name": "000082", "path": "000082/img", "startFrame": 1, "endFrame": 719, "nz": 6,
             "ext": "jpg", "anno_path": "000082/groundtruth.txt",
             "object_class": "other"},
            {"name": "000083", "path": "000083/img", "startFrame": 1, "endFrame": 679, "nz": 6,
             "ext": "jpg", "anno_path": "000083/groundtruth.txt",
             "object_class": "other"},
            {"name": "000084", "path": "000084/img", "startFrame": 1, "endFrame": 1244, "nz": 6,
             "ext": "jpg", "anno_path": "000084/groundtruth.txt",
             "object_class": "other"},
            {"name": "000085", "path": "000085/img", "startFrame": 1, "endFrame": 320, "nz": 6,
             "ext": "jpg", "anno_path": "000085/groundtruth.txt",
             "object_class": "other"},
            {"name": "000086", "path": "000086/img", "startFrame": 1, "endFrame": 491, "nz": 6,
             "ext": "jpg", "anno_path": "000086/groundtruth.txt",
             "object_class": "other"},
            {"name": "000087", "path": "000087/img", "startFrame": 1, "endFrame": 648, "nz": 6,
             "ext": "jpg", "anno_path": "000087/groundtruth.txt",
             "object_class": "other"},
            {"name": "000088", "path": "000088/img", "startFrame": 1, "endFrame": 1394, "nz": 6,
             "ext": "jpg", "anno_path": "000088/groundtruth.txt",
             "object_class": "other"},
            {"name": "000089", "path": "000089/img", "startFrame": 1, "endFrame": 609, "nz": 6,
             "ext": "jpg", "anno_path": "000089/groundtruth.txt",
             "object_class": "other"},
            {"name": "000090", "path": "000090/img", "startFrame": 1, "endFrame": 367, "nz": 6,
             "ext": "jpg", "anno_path": "000090/groundtruth.txt",
             "object_class": "other"},
            {"name": "000091", "path": "000091/img", "startFrame": 1, "endFrame": 2560, "nz": 6,
             "ext": "jpg", "anno_path": "000091/groundtruth.txt",
             "object_class": "other"},
            {"name": "000092", "path": "000092/img", "startFrame": 1, "endFrame": 1466, "nz": 6,
             "ext": "jpg", "anno_path": "000092/groundtruth.txt",
             "object_class": "other"},
            {"name": "000093", "path": "000093/img", "startFrame": 1, "endFrame": 520, "nz": 6,
             "ext": "jpg", "anno_path": "000093/groundtruth.txt",
             "object_class": "other"},
            {"name": "000094", "path": "000094/img", "startFrame": 1, "endFrame": 764, "nz": 6,
             "ext": "jpg", "anno_path": "000094/groundtruth.txt",
             "object_class": "other"},
            {"name": "000095", "path": "000095/img", "startFrame": 1, "endFrame": 721, "nz": 6,
             "ext": "jpg", "anno_path": "000095/groundtruth.txt",
             "object_class": "other"},
            {"name": "000096", "path": "000096/img", "startFrame": 1, "endFrame": 425, "nz": 6,
             "ext": "jpg", "anno_path": "000096/groundtruth.txt",
             "object_class": "other"},
            {"name": "000097", "path": "000097/img", "startFrame": 1, "endFrame": 850, "nz": 6,
             "ext": "jpg", "anno_path": "000097/groundtruth.txt",
             "object_class": "other"},
            {"name": "000098", "path": "000098/img", "startFrame": 1, "endFrame": 1153, "nz": 6,
             "ext": "jpg", "anno_path": "000098/groundtruth.txt",
             "object_class": "other"},
            {"name": "000099", "path": "000099/img", "startFrame": 1, "endFrame": 822, "nz": 6,
             "ext": "jpg", "anno_path": "000099/groundtruth.txt",
             "object_class": "other"},
            {"name": "000100", "path": "000100/img", "startFrame": 1, "endFrame": 626, "nz": 6,
             "ext": "jpg", "anno_path": "000100/groundtruth.txt",
             "object_class": "other"},
            {"name": "000101", "path": "000101/img", "startFrame": 1, "endFrame": 781, "nz": 6,
             "ext": "jpg", "anno_path": "000101/groundtruth.txt",
             "object_class": "other"},
            {"name": "000102", "path": "000102/img", "startFrame": 1, "endFrame": 1559, "nz": 6,
             "ext": "jpg", "anno_path": "000102/groundtruth.txt",
             "object_class": "other"},
            {"name": "000103", "path": "000103/img", "startFrame": 1, "endFrame": 633, "nz": 6,
             "ext": "jpg", "anno_path": "000103/groundtruth.txt",
             "object_class": "other"}
        ]
    
        return sequence_info_list