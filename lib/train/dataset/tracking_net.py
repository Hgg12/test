import os
import json
from .base_video_dataset import BaseVideoDataset
from lib.train.data.image_loader import jpeg4py_loader
import time

class TrackingNet(BaseVideoDataset):
    """
    TrackingNet dataset.

    This class has been modified to load data from a JSON annotation file,
    similar to the WATS-DA / SAM-DA implementation, and to correctly
    interpret the provided JSON structure.
    """
    def __init__(self, root, image_loader=jpeg4py_loader, anno_path=None):
        """
        args:
            root - path to the dataset root folder where video folders are stored.
            image_loader - The function to read the images.
            anno_path - path to the annotation file (e.g., train.json).
        """
        super().__init__('TrackingNet', root, image_loader)

        if anno_path is None:
            raise ValueError("Annotation file path (anno_path) is required for TrackingNet.")
        
        if not os.path.exists(anno_path):
            raise FileNotFoundError(f"Annotation file not found at: {anno_path}")
            
        # --- 已修正：添加进度打印 ---
        print(f"Loading annotation file: {anno_path}...")
        start_time = time.time()
        with open(anno_path, 'r') as f:
            anno_data_raw = json.load(f)
        print(f"Annotation file loaded in {time.time() - start_time:.2f} seconds. Processing data...")

        # Preprocess the annotation data
        start_time = time.time()
        self.anno_data = self._process_anno(anno_data_raw)
        print(f"Annotation data processed in {time.time() - start_time:.2f} seconds.")

        self.sequence_list = list(self.anno_data.keys())
        
        # Optional: Filter out sequences that are not present in the root directory
        print("Checking sequence directories on disk...")
        start_time = time.time()
        self.sequence_list = [seq for seq in self.sequence_list if os.path.isdir(os.path.join(self.root, seq))]
        print(f"Disk check complete in {time.time() - start_time:.2f} seconds. Found {len(self.sequence_list)} valid sequences.")
        
        if not self.sequence_list:
            raise Exception("No valid sequences found in the root directory that match the annotation file.")

    def _process_anno(self, anno_data):
        """
        Processes the raw annotation data. It filters out invalid bounding boxes
        and adds a 'frames' key with a sorted list of frame names to each track.
        """
        processed_anno = {}
        for video_name, video_info in anno_data.items():
            new_video_info = {}
            for track_id, track_info_raw in video_info.items():
                new_track_info = {}
                frame_keys = []
                # Sort frames numerically to ensure correct order
                for frame_key in sorted(track_info_raw.keys(), key=int):
                    bbox = track_info_raw[frame_key]
                    # Bbox format is [x1, y1, x2, y2]
                    # Check for valid width and height
                    if (bbox[2] - bbox[0]) > 0 and (bbox[3] - bbox[1]) > 0:
                        new_track_info[frame_key] = bbox
                        frame_keys.append(frame_key)
                
                if frame_keys:
                    new_track_info['frames'] = frame_keys
                    new_video_info[track_id] = new_track_info
            
            if new_video_info:
                processed_anno[video_name] = new_video_info
        return processed_anno

    def get_name(self):
        return 'TrackingNet'

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_sequence_info(self, seq_id):
        video_name = self.sequence_list[seq_id]
        video_info = self.anno_data[video_name]
        
        first_track_id = list(video_info.keys())[0]
        track_info = video_info[first_track_id]
        
        frames = track_info['frames']
        
        bbox_list = [track_info[f] for f in frames]
        # A bounding box is valid if its width and height are positive.
        valid = [( (bbox[2] - bbox[0]) > 0 and (bbox[3] - bbox[1]) > 0 ) for bbox in bbox_list]
        visible = valid  # Assume all valid frames are visible

        return {'bbox': bbox_list, 'valid': valid, 'visible': visible}

    def get_frame(self, seq_id, frame_id):
        """
        Gets a frame from the dataset.
        args:
            seq_id - index of the sequence.
            frame_id - index of the frame in the sequence.
        """
        video_name = self.sequence_list[seq_id]
        video_info = self.anno_data[video_name]
        
        first_track_id = list(video_info.keys())[0]
        track_info = video_info[first_track_id]
        
        frame_key = track_info['frames'][frame_id]
        
        # Construct image path. Assumes frames are like '000001.jpg' or similar.
        image_path = os.path.join(self.root, video_name, f"{frame_key}.jpg")
        
        anno = track_info[frame_key]
        img = self.image_loader(image_path)
        
        # The BaseVideoDataset expects a mask, we return None if not available.
        return img, anno, None
