import numpy as np
import torch
from lib.utils.bbox import corner2center, Center

class PointTarget:
    def __init__(self, cfg=None):
        # In this version, cfg is not used but kept for potential future use.
        pass

    def __call__(self, target_bbox, size, neg=False, stride=16, neg_radius=2.5):
        """
        Args:
            target_bbox: Corner bbox
            size: output size of score map
            neg: whether is negative sample
        """
        center_bbox = corner2center(target_bbox)
        x, y, w, h = center_bbox.x, center_bbox.y, center_bbox.w, center_bbox.h

        cls = np.zeros((size, size), dtype=np.int64)
        delta = np.zeros((4, size, size), dtype=np.float32)

        if not neg:
            s = 1 / stride
            x, y, w, h = x * s, y * s, w * s, h * s
            
            x1 = int(np.round(x - w/2))
            y1 = int(np.round(y - h/2))
            x2 = int(np.round(x + w/2))
            y2 = int(np.round(y + h/2))

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(size-1, x2)
            y2 = min(size-1, y2)
            
            cls[y1:y2+1, x1:x2+1] = 1

            # Generate delta
            for i in range(size):
                for j in range(size):
                    if cls[i, j] == 1:
                        delta[0, i, j] = j - (x - w/2)
                        delta[1, i, j] = i - (y - h/2)
                        delta[2, i, j] = (x + w/2) - j
                        delta[3, i, j] = (y + h/2) - i
        
        return cls, delta
