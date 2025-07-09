import torch
from torch import nn

# 导入所有必要的模块
from lib.models.mobilevit_track.mobilevitv2_track import build_mobilevitv2_track
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.utils.misc import NestedTensor
from lib.utils.ce_utils import generate_mask_cond, adjust_keep_rate
from . import BaseActor

# 新增：导入判别器
from lib.models.discriminator import TransformerDiscriminator

class MobileViTTrackActor(BaseActor):
    """ Actor for training the MobileViT-Track model.
    This actor is modified to support Domain-Adversarial training.
    """
    def __init__(self, net, objective, loss_weight, settings, cfg):
        """
        Initializes the MobileViTTrackActor.
        Args:
            net: The main network model.
            objective: The loss function for the tracking task.
            loss_weight: A dictionary of weights for different loss components.
            settings: The training settings (for env paths, device, etc.).
            cfg: The experiment configuration (for hyperparameters).
        """
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.cfg = cfg # 保存配置对象

        # --- 已修正：使用正确的 cfg 对象访问 BATCH_SIZE ---
        self.bs = self.cfg.TRAIN.BATCH_SIZE  # batch size

        # --- 新增：初始化判别器 ---
        # 注意：这里的参数需要根据您的模型输出进行精确调整。
        # channels: 应等于主干网络输出给 neck 的特征通道数。
        # img_size: 应等于 neck 输出给 head 的特征图的空间尺寸。
        
        # 这是一个示例，假设判别器接收来自 backbone 最后一个阶段的特征。
        discriminator_channels = net.backbone.model_conf_dict['layer4']['out']
        # --- 已修正：使用 cfg 对象访问配置 ---
        feature_map_size = self.cfg.DATA.SEARCH.SIZE // self.cfg.MODEL.BACKBONE.STRIDE 
        
        self.discriminator = TransformerDiscriminator(
            channels=discriminator_channels, 
            img_size=feature_map_size
        )
        # 将判别器移动到正确的设备 (从 settings 获取)
        self.discriminator.to(self.settings.device)


    def __call__(self, data):
        """
        Runs a forward pass for training.
        Args:
            data: The input data, a dict with the following keys:
                  'template_s', 'search_s', 'template_anno_s', 'search_anno_s',
                  'template_t', 'search_t'
        Returns:
            loss: The training loss (only tracking loss).
            stats: A dict of statistics, including discriminator predictions.
        """
        # --- 跟踪任务前向传播 (源域) ---
        # 将源域数据传递给主网络
        out_dict_s = self.net(template=data['template_s'], search=data['search_s'])

        # 计算跟踪损失
        loss, stats = self.compute_losses(out_dict_s, data['search_anno_s'])

        # --- 领域自适应任务 ---
        # 1. 提取用于判别的特征
        # 我们使用 backbone 输出的最后一个阶段的特征作为判别器的输入
        source_features = self.net.forward_features(data['search_s'])[-1] # 假设返回一个元组，取最后一个
        
        # 目标域的特征提取不需要计算梯度
        with torch.no_grad():
            target_features = self.net.forward_features(data['search_t'])[-1] # 假设返回一个元组，取最后一个

        # 2. 将特征送入判别器
        domain_pred_s = self.discriminator(source_features)
        domain_pred_t = self.discriminator(target_features)

        # 3. 将判别器输出添加到 stats 字典中，以便训练脚本使用
        stats['domain_pred_s'] = domain_pred_s
        stats['domain_pred_t'] = domain_pred_t

        return loss, stats

    def compute_losses(self, pred_dict, gt_bbox, return_status=False):
        """
        Computes the tracking loss.
        This function is kept from the original implementation.
        """
        # Get groundtruth boxes
        gt_bbox = gt_bbox.to(pred_dict['pred_boxes'].device)
        # Get iou prediction
        pred_iou = pred_dict.get('pred_iou') # Use .get for safety
        # Get predicted boxes
        pred_boxes = pred_dict['pred_boxes']
        
        # box prediction loss
        loss_box = self.objective['box'](pred_boxes, gt_bbox)
        
        # iou prediction loss (if iou head exists)
        loss_iou = torch.tensor(0.0, device=gt_bbox.device)
        if pred_iou is not None and 'iou' in self.objective and 'iou' in self.loss_weight:
            # The target for iou prediction is the iou between predicted box and gt box
            # We don't want to backprop through the detached iou value
            iou_target = loss_box.detach().mean(dim=1, keepdim=True) # A simplified target
            loss_iou = self.objective['iou'](pred_iou, iou_target)

        # weighted sum
        loss = self.loss_weight.get('box', 1.0) * loss_box + self.loss_weight.get('iou', 0.0) * loss_iou

        # status for log
        mean_iou = loss_box.detach().mean()
        stats = {"Loss/box": loss_box.item(), "IoU": mean_iou.item()}
        if pred_iou is not None and 'iou' in self.loss_weight:
            stats["Loss/iou"] = loss_iou.item()
            
        if return_status:
            return loss, stats
        else:
            return loss, stats
