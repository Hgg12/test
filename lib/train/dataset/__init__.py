# Standard datasets
from .lasot import Lasot
from .got10k import Got10k
from .tracking_net import TrackingNet
from .coco import MSCOCO
# --- 已修正：使用别名导入正确的类名 MSCOCOSeq ---
from .coco_seq import MSCOCOSeq as CocoSeq
from .imagenetvid import ImagenetVID

# LMDB variants
from .got10k_lmdb import Got10k_lmdb
from .lasot_lmdb import Lasot_lmdb
from .imagenetvid_lmdb import ImagenetVID_lmdb
from .coco_seq_lmdb import MSCOCOSeq_lmdb as CocoSeq_lmdb
from .tracking_net_lmdb import TrackingNet_lmdb

# --- 新增：导出您的自定义数据集类 ---
# Note: Ensure you have created the corresponding .py files for these classes in this directory.
# For example, watb400_1.py, watb400_2.py, etc.
try:
    from .watb400_1 import WATB400_1
    from .watb400_2 import WATB400_2
    from .got10k_official_val import GOT10K_official_val
except ImportError:
    print("Warning: Custom dataset files (e.g., watb400_1.py) not found. Please create them if needed.")
    # Define placeholder classes to avoid further import errors if the files don't exist yet
    class WATB400_1: pass
    class WATB400_2: pass
    class GOT10K_official_val: pass


