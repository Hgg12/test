import os
from .base_video_dataset import BaseVideoDataset
from lib.train.data.image_loader import jpeg4py_loader

class WATB400_1(BaseVideoDataset):
    """
    WATB400-1 dataset.
    This is a placeholder class for your custom dataset.
    You can modify this as needed.
    """
    def __init__(self, root, image_loader=jpeg4py_loader):
        """
        args:
            root - path to the dataset root folder.
            image_loader - The function to read the images.
        """
        super().__init__('WATB400_1', root, image_loader)
        self.sequence_list = self._get_sequence_list()

    def _get_sequence_list(self):
        # This should return a list of sequence names in your dataset
        # For now, it assumes sequences are directories in the root folder.
        return [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]

    def get_name(self):
        return 'watb400_1'
