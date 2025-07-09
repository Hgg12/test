"""
Basic MobileViT-Track model.
This file has been modified to support domain adaptation.
"""
import math
import os
from typing import List, Union, Tuple, Optional

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.mobilevit_track.layers.conv_layer import Conv2d
from .layers.neck import build_neck, build_feature_fusor
from .layers.head import build_box_head

from lib.models.mobilevit_track.mobilevit_v2 import MobileViTv2_backbone
from lib.utils.box_ops import box_xyxy_to_cxcywh
from easydict import EasyDict as edict


class MobileViTv2_Track(nn.Module):
    """ This is the base class for MobileViTv2-Track """

    def __init__(self, backbone, neck, feature_fusor, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = backbone
        if neck is not None:
            self.neck = neck
        if feature_fusor is not None:
            self.feature_fusor = feature_fusor
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or "CENTER" in head_type:
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor, search: torch.Tensor):
        # Standard forward pass for tracking
        x, z = self.backbone(x=search, z=template)

        # Forward neck
        if hasattr(self, 'neck'):
            x, z = self.neck(x, z)

        # Forward feature fusor
        if hasattr(self, 'feature_fusor'):
            feat_fused = self.feature_fusor(z, x)
        else:
            feat_fused = x 

        # Forward head
        out = self.forward_head(feat_fused, None)

        return out

    def forward_features(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """ 
        --- 新增的辅助函数 ---
        Helper function to extract features for a single image stack (e.g., search regions).
        This is required for the domain adaptation task.
        The backbone's forward method can handle a single input 'x'.
        """
        return self.backbone(x=x)

    def forward_head(self, backbone_feature, gt_score_map=None):
        """
        backbone_feature: output embeddings of the backbone for search region
        """
        opt_feat = backbone_feature.contiguous()
        bs, _, _, _ = opt_feat.size()
        
        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif "CENTER" in self.head_type:
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_mobilevitv2_track(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if "mobilevitv2" in cfg.MODEL.BACKBONE.TYPE:
        # --- 已修正的逻辑：安全地获取宽度乘数 ---
        backbone_type_parts = cfg.MODEL.BACKBONE.TYPE.split('-')
        # 检查是否在类型名中指定了宽度乘数 (例如, 'mobilevitv2-0.75')
        if len(backbone_type_parts) > 1 and backbone_type_parts[-1].replace('.', '', 1).isdigit():
            width_multiplier = float(backbone_type_parts[-1])
        else:
            # 如果没有指定，则默认为 1.0
            width_multiplier = 1.0
        
        print(f"Building MobileViT V2 with width multiplier: {width_multiplier}")
        
        backbone = create_mobilevitv2_backbone(pretrained, width_multiplier, has_mixed_attn=cfg.MODEL.BACKBONE.MIXED_ATTN)
        if cfg.MODEL.BACKBONE.MIXED_ATTN is True:
            backbone.mixed_attn = True
        else:
            backbone.mixed_attn = False
        hidden_dim = backbone.model_conf_dict['layer4']['out']
        patch_start_index = 1
    else:
        raise NotImplementedError

    # build neck module
    if cfg.MODEL.NECK:
        neck = build_neck(cfg=cfg, hidden_dim=hidden_dim)
    else:
        neck = None

    # build feature fusor module
    if cfg.MODEL.NECK and cfg.MODEL.NECK.TYPE == "BN_PWXCORR":
        feature_fusor = build_feature_fusor(cfg=cfg, in_features=hidden_dim,
                                              xcorr_out_features=cfg.MODEL.NECK.NUM_CHANNS_POST_XCORR)
    elif cfg.MODEL.NECK and (cfg.MODEL.NECK.TYPE == "BN_SSAT" or cfg.MODEL.NECK.TYPE == "BN_HSSAT"):
        feature_fusor = build_feature_fusor(cfg=cfg, in_features=hidden_dim,
                                              xcorr_out_features=None)
    else:
        feature_fusor = None

    # build head module
    box_head = build_box_head(cfg, cfg.MODEL.HEAD.NUM_CHANNELS)

    model = MobileViTv2_Track(
        backbone=backbone,
        neck=neck,
        feature_fusor=feature_fusor,
        box_head=box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'mobilevit_track' in cfg.MODEL.PRETRAIN_FILE and training:
        if not os.path.exists(cfg.MODEL.PRETRAIN_FILE):
             print(f"Warning: Pretrained model path does not exist: {cfg.MODEL.PRETRAIN_FILE}")
        else:
            checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            if missing_keys or unexpected_keys:
                print("Warning: Mismatch in loading pretrained model weights.")
                print("Missing keys:", missing_keys)
                print("Unexpected keys:", unexpected_keys)
            else:
                print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model


def create_mobilevitv2_backbone(pretrained, width_multiplier, has_mixed_attn):
    """
    function to create an instance of MobileViT backbone
    """
    opts = {}
    opts['mode'] = width_multiplier
    opts['head_dim'] = None
    opts['number_heads'] = 4
    opts['conv_layer_normalization_name'] = 'batch_norm'
    opts['conv_layer_activation_name'] = 'relu'
    opts['mixed_attn'] = has_mixed_attn
    model = MobileViTv2_backbone(opts)

    if pretrained:
        if not os.path.exists(pretrained):
            print(f"Warning: Pretrained backbone path does not exist: {pretrained}")
            return model
            
        checkpoint = torch.load(pretrained, map_location="cpu")
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys or unexpected_keys:
            print("Warning: Mismatch in loading pretrained backbone weights.")
            # Filter out keys that are expected to be missing (like classifier head)
            missing_keys_filtered = [k for k in missing_keys if 'classifier' not in k]
            if missing_keys_filtered:
                 print("Missing keys (filtered):", missing_keys_filtered)
            if unexpected_keys:
                 print("Unexpected keys:", unexpected_keys)
        else:
            print('Load pretrained backbone from: ' + pretrained)

    return model
