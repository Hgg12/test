import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class TLPDataset(BaseDataset):
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
        self.base_path = self.env_settings.tlp_path
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

        return Sequence(sequence_info['name'], frames, 'tlp', ground_truth_rect[init_omit:,:],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {"name": "Alladin", "path": "Alladin/img", "startFrame": 1, "endFrame": 8992, "nz": 5, "ext": "jpg", "anno_path": "../anno/Alladin.txt",
             "object_class": "person"},
            {"name": "Aquarium1", "path": "Aquarium1/img", "startFrame": 1, "endFrame": 7337, "nz": 5, "ext": "jpg", "anno_path": "../anno/Aquarium1.txt",
             "object_class": "person head"},
            {"name": "Aquarium2", "path": "Aquarium2/img", "startFrame": 1, "endFrame": 8182, "nz": 5, "ext": "jpg", "anno_path": "../anno/Aquarium2.txt",
             "object_class": "bird"},
            {"name": "Badminton1", "path": "Badminton1/img", "startFrame": 1, "endFrame": 15240, "nz": 5, "ext": "jpg", "anno_path": "../anno/Badminton1.txt",
             "object_class": "bird"},
            {"name": "Badminton2", "path": "Badminton2/img", "startFrame": 1, "endFrame": 16920, "nz": 5, "ext": "jpg", "anno_path": "../anno/Badminton2.txt",
             "object_class": "person"},
            {"name": "Basketball", "path": "Basketball/img", "startFrame": 1, "endFrame": 17970, "nz": 5, "ext": "jpg", "anno_path": "../anno/Basketball.txt",
             "object_class": "car"},
            {"name": "Bharatanatyam", "path": "Bharatanatyam/img", "startFrame": 1, "endFrame": 15936, "nz": 5, "ext": "jpg", "anno_path": "../anno/Bharatanatyam.txt",
             "object_class": "car"},
            {"name": "Bike", "path": "Bike/img", "startFrame": 1, "endFrame": 4196, "nz": 5, "ext": "jpg", "anno_path": "../anno/Bike.txt",
             "object_class": "car"},
            {"name": "Billiards1", "path": "Billiards1/img", "startFrame": 1, "endFrame": 20375, "nz": 5, "ext": "jpg", "anno_path": "../anno/Billiards1.txt",
             "object_class": "car"},
            {"name": "Billiards2", "path": "Billiards2/img", "startFrame": 1, "endFrame": 20070, "nz": 5, "ext": "jpg", "anno_path": "../anno/Billiards2.txt",
             "object_class": "face"},
            {"name": "Boat", "path": "Boat/img", "startFrame": 1, "endFrame": 6234, "nz": 5, "ext": "jpg", "anno_path": "../anno/Boat.txt",
             "object_class": "other"},
            {"name": "Boxing1", "path": "Boxing1/img", "startFrame": 1, "endFrame": 20670, "nz": 5, "ext": "jpg", "anno_path": "../anno/Boxing1.txt",
             "object_class": "other"},
            {"name": "Boxing2", "path": "Boxing2/img", "startFrame": 1, "endFrame": 21180, "nz": 5, "ext": "jpg", "anno_path": "../anno/Boxing2.txt",
             "object_class": "person"},
            {"name": "Boxing3", "path": "Boxing3/img", "startFrame": 1, "endFrame": 19590, "nz": 5, "ext": "jpg", "anno_path": "../anno/Boxing3.txt",
             "object_class": "person"},
            {"name": "BreakfastClub", "path": "BreakfastClub/img", "startFrame": 1, "endFrame": 22600, "nz": 5, "ext": "jpg", "anno_path": "../anno/BreakfastClub.txt",
             "object_class": "other"},
            {"name": "CarChase1", "path": "CarChase1/img", "startFrame": 1, "endFrame": 8932, "nz": 5, "ext": "jpg", "anno_path": "../anno/CarChase1.txt",
             "object_class": "face"},
            {"name": "CarChase2", "path": "CarChase2/img", "startFrame": 1, "endFrame": 14010, "nz": 5, "ext": "jpg", "anno_path": "../anno/CarChase2.txt",
             "object_class": "car"},
            {"name": "CarChase3", "path": "CarChase3/img", "startFrame": 1, "endFrame": 22860, "nz": 5, "ext": "jpg", "anno_path": "../anno/CarChase3.txt",
             "object_class": "car"},
            {"name": "Dashcam", "path": "Dashcam/img", "startFrame": 1, "endFrame": 10260, "nz": 5, "ext": "jpg", "anno_path": "../anno/Dashcam.txt",
             "object_class": "car"},
            {"name": "DriftCar1", "path": "DriftCar1/img", "startFrame": 1, "endFrame": 10130, "nz": 5, "ext": "jpg", "anno_path": "../anno/DriftCar1.txt",
             "object_class": "car"},
            {"name": "DriftCar2", "path": "DriftCar2/img", "startFrame": 1, "endFrame": 8572, "nz": 5, "ext": "jpg", "anno_path": "../anno/DriftCar2.txt",
             "object_class": "car"},
            {"name": "Drone1", "path": "Drone1/img", "startFrame": 1, "endFrame": 4320, "nz": 5, "ext": "jpg", "anno_path": "../anno/Drone1.txt",
             "object_class": "car"},
            {"name": "Drone2", "path": "Drone2/img", "startFrame": 1, "endFrame": 8812, "nz": 5, "ext": "jpg", "anno_path": "../anno/Drone2.txt",
             "object_class": "other"},
            {"name": "Drone3", "path": "Drone3/img", "startFrame": 1, "endFrame": 6594, "nz": 5, "ext": "jpg", "anno_path": "../anno/Drone3.txt",
             "object_class": "other"},
            {"name": "Elephants", "path": "Elephants/img", "startFrame": 1, "endFrame": 4376, "nz": 5, "ext": "jpg", "anno_path": "../anno/Elephants.txt",
             "object_class": "person"},
            {"name": "Helicopter", "path": "Helicopter/img", "startFrame": 1, "endFrame": 17053, "nz": 5, "ext": "jpg", "anno_path": "../anno/Helicopter.txt",
             "object_class": "other"},
            {"name": "Hideaway", "path": "Hideaway/img", "startFrame": 1, "endFrame": 5900, "nz": 5, "ext": "jpg", "anno_path": "../anno/Hideaway.txt",
             "object_class": "person"},
            {"name": "IceSkating", "path": "IceSkating/img", "startFrame": 1, "endFrame": 8125, "nz": 5, "ext": "jpg", "anno_path": "../anno/IceSkating.txt",
             "object_class": "person"},
            {"name": "ISS", "path": "ISS/img", "startFrame": 1, "endFrame": 28562, "nz": 5, "ext": "jpg", "anno_path": "../anno/ISS.txt",
             "object_class": "person"},
            {"name": "Jet1", "path": "Jet1/img", "startFrame": 1, "endFrame": 7403, "nz": 5, "ext": "jpg", "anno_path": "../anno/Jet1.txt",
             "object_class": "person"},
            {"name": "Jet2", "path": "Jet2/img", "startFrame": 1, "endFrame": 18882, "nz": 5, "ext": "jpg", "anno_path": "../anno/Jet2.txt",
             "object_class": "face"},
            {"name": "Jet3", "path": "Jet3/img", "startFrame": 1, "endFrame": 17953, "nz": 5, "ext": "jpg", "anno_path": "../anno/Jet3.txt",
             "object_class": "face"},
            {"name": "Jet4", "path": "Jet4/img", "startFrame": 1, "endFrame": 10160, "nz": 5, "ext": "jpg", "anno_path": "../anno/Jet4.txt",
             "object_class": "person"},
            {"name": "Jet5", "path": "Jet5/img", "startFrame": 1, "endFrame": 13675, "nz": 5, "ext": "jpg", "anno_path": "../anno/Jet5.txt",
             "object_class": "mammal"},
            {"name": "KinBall1", "path": "KinBall1/img", "startFrame": 1, "endFrame": 20230, "nz": 5, "ext": "jpg", "anno_path": "../anno/KinBall1.txt",
             "object_class": "person"},
            {"name": "KinBall2", "path": "KinBall2/img", "startFrame": 1, "endFrame": 13575, "nz": 5, "ext": "jpg", "anno_path": "../anno/KinBall2.txt",
             "object_class": "dog"},
            {"name": "KinBall3", "path": "KinBall3/img", "startFrame": 1, "endFrame": 14940, "nz": 5, "ext": "jpg", "anno_path": "../anno/KinBall3.txt",
             "object_class": "dog"},
            {"name": "Lion", "path": "Lion/img", "startFrame": 1, "endFrame": 6570, "nz": 5, "ext": "jpg", "anno_path": "../anno/Lion.txt",
             "object_class": "other"},
            {"name": "Mohiniyattam", "path": "Mohiniyattam/img", "startFrame": 1, "endFrame": 15456, "nz": 5, "ext": "jpg", "anno_path": "../anno/Mohiniyattam.txt",
             "object_class": "face"},
            {"name": "MotorcycleChase", "path": "MotorcycleChase/img", "startFrame": 1, "endFrame": 5550, "nz": 5, "ext": "jpg", "anno_path": "../anno/MotorcycleChase.txt",
             "object_class": "face"},
            {"name": "Parakeet", "path": "Parakeet/img", "startFrame": 1, "endFrame": 21609, "nz": 5, "ext": "jpg", "anno_path": "../anno/Parakeet.txt",
             "object_class": "face"},
            {"name": "PolarBear1", "path": "PolarBear1/img", "startFrame": 1, "endFrame": 9501, "nz": 5, "ext": "jpg", "anno_path": "../anno/PolarBear1.txt",
             "object_class": "face"},
            {"name": "PolarBear2", "path": "PolarBear2/img", "startFrame": 1, "endFrame": 27153, "nz": 5, "ext": "jpg", "anno_path": "../anno/PolarBear2.txt",
             "object_class": "other"},
            {"name": "PolarBear3", "path": "PolarBear3/img", "startFrame": 1, "endFrame": 9531, "nz": 5, "ext": "jpg", "anno_path": "../anno/PolarBear3.txt",
             "object_class": "face"},
            {"name": "Puppies1", "path": "Puppies1/img", "startFrame": 1, "endFrame": 17730, "nz": 5, "ext": "jpg", "anno_path": "../anno/Puppies1.txt",
             "object_class": "person head"},
            {"name": "Puppies2", "path": "Puppies2/img", "startFrame": 1, "endFrame": 22620, "nz": 5, "ext": "jpg", "anno_path": "../anno/Puppies2.txt",
             "object_class": "face"},
            {"name": "Rope", "path": "Rope/img", "startFrame": 1, "endFrame": 17503, "nz": 5, "ext": "jpg", "anno_path": "../anno/Rope.txt",
             "object_class": "face"},
            {"name": "Sam", "path": "Sam/img", "startFrame": 1, "endFrame": 4628, "nz": 5, "ext": "jpg", "anno_path": "../anno/Sam.txt",
             "object_class": "face"},
            {"name": "Violinist", "path": "Violinist/img", "startFrame": 1, "endFrame": 6844, "nz": 5, "ext": "jpg", "anno_path": "../anno/Violinist.txt",
             "object_class": "face"},
            {"name": "ZebraFish", "path": "ZebraFish/img", "startFrame": 1, "endFrame": 10920, "nz": 5, "ext": "jpg", "anno_path": "../anno/ZebraFish.txt",
             "object_class": "face"}
        ]
    
        return sequence_info_list