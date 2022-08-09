# ------------------------------------------------------------------------
# GRIT: Faster and Better Image captioning Transformer
# Licensed under the Creative Commons Attribution.
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.swin_model import *
from models.detection.od_losses import SetCriterion, build_matcher, PostProcess
from utils.misc import nested_tensor_from_tensor_list, NestedTensor

from models.detection.det_module import build_det_module_with_config
from models.detection.heads import AttrHead


class Detector(nn.Module):

    def __init__(self,
                 backbone,
                 det_module,
                 hidden_dim=256,
                 has_attr_head=False,
                 num_attr_classes=400,
                 num_od_classes=1849):
        super().__init__()
        self.backbone = backbone
        self.det_module = det_module
        self.has_attr_head = has_attr_head
        if self.has_attr_head:
            self.attr_head = AttrHead(hidden_dim, num_attr_classes, num_od_classes)
        # self.vl_module = vl_module

        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(backbone.num_channels[i], hidden_dim, kernel_size=1),  # linear layer
                nn.GroupNorm(32, hidden_dim),
            ) for i in range(len(backbone.num_channels))
        ])
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        x = samples.tensors  # RGB input # [B, 3, H, W]
        mask = samples.mask  # padding mask [B, H, W]

        # return multi-scale [PATCH] tokens along with final [DET] tokens and their pos encodings
        features = self.backbone(x)
        # features = [[B, C1, H1, W1], [B, C2, H2, W2], [B, C3, H3, W3], [B, C4, H4, W4]]
        # C1 -> C3 = embed_dim*2(i) i = 1->3
        # C4 = query_pos_dim
        # H1 = H/8, H2 = H/16, H3 = H/32, H4 = H/64

        # srcs = [[B, D, Hi, Wi]]
        srcs = [self.input_proj[l](src) for l, src in enumerate(features)]
        masks = [
            F.interpolate(mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0] for l, src in enumerate(srcs)
        ]  # masks [[B, Hi, Wi]]

        hs, init_reference, inter_references = self.det_module(srcs, masks)
        outputs = self.det_module.detection_head(hs, init_reference, inter_references)
        if self.has_attr_head:
            attr_outputs = self.attr_head(hs[-1], outputs['pred_logits'])
            outputs.update(attr_outputs)
        return outputs

    def forward_features(self, batch):
        samples = batch['image']
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        x = samples.tensors  # RGB input # [B, 3, H, W]
        mask = samples.mask  # padding mask [B, H, W]

        features = self.backbone(x)
        # features = [[B, C1, H1, W1], [B, C2, H2, W2], [B, C3, H3, W3], [B, C4, H4, W4]]
        # C1 -> C3 = embed_dim*2(i) i = 1->3
        # C4 = query_pos_dim
        # H1 = H/8, H2 = H/16, H3 = H/32, H4 = H/64

        # srcs = [[B, D, Hi, Wi]]
        srcs = [self.input_proj[l](src) for l, src in enumerate(features)]
        masks = [
            F.interpolate(mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0] for l, src in enumerate(srcs)
        ]  # masks [[B, Hi, Wi]]

        out = {'img': srcs[-1], 'img_mask': masks[-1]}

        if self.det_module:
            hs, init_reference, inter_references = self.det_module(srcs, masks)
            out['det_queries'] = hs[-1]

        return out


def build_backbone_with_config(config):
    # config at backbone level
    if config.backbone_name == 'swin_nano':
        backbone, hidden_dim = swin_nano(pretrained=config.pre_trained)
    elif config.backbone_name == 'swin_tiny':
        backbone, hidden_dim = swin_tiny(pretrained=config.pre_trained)
    elif config.backbone_name == 'swin_small':
        backbone, hidden_dim = swin_small(pretrained=config.pre_trained)
    elif config.backbone_name == 'swin_base_win7_224_22k':
        backbone, hidden_dim = swin_base_win7_224(pretrained=config.pre_trained)
    elif config.backbone_name == 'swin_large_win7_224_22k':
        backbone, hidden_dim = swin_large_win7_224(pretrained=config.pre_trained)
    elif config.backbone_name == 'swin_base_win7_384_22k':
        backbone, hidden_dim = swin_base_win7_384(pretrained=config.pre_trained)
    elif config.backbone_name == 'swin_large_win7_384_22k':
        backbone, hidden_dim = swin_large_win7_384(pretrained=config.pre_trained)
    else:
        raise ValueError(f'backbone {config.backbone_name} not supported')
    return backbone


def build_detector(config):
    backbone = build_backbone_with_config(config.model.backbone)
    det_cfg = config.model.det_module
    det_module = build_det_module_with_config(det_cfg)
    has_attr_head = getattr(config.model, 'has_attr_head', False)
    model = Detector(backbone, det_module, hidden_dim=det_cfg.reduced_dim, has_attr_head=has_attr_head)

    matcher = build_matcher(det_cfg.matcher)

    loss_cfg = det_cfg.loss
    weight_dict = {
        'loss_ce': loss_cfg.cls_loss_coef,
        'loss_bbox': loss_cfg.bbox_loss_coef,
        'loss_giou': loss_cfg.giou_loss_coef,
        'loss_attr': loss_cfg.attr_loss_coef,
    }

    # aux decoding loss
    if det_cfg.aux_loss:
        aux_weight_dict = {}
        for i in range(det_cfg.num_layers - 1 + 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']

    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(det_cfg.num_classes, matcher, weight_dict, losses, focal_alpha=loss_cfg.focal_alpha)
    postprocessors = {'bbox': PostProcess()}  # todo: hardcode here

    return model, criterion, postprocessors
