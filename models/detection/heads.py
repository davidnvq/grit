# ------------------------------------------------------------------------
# GRIT: Faster and Better Image captioning Transformer
# Licensed under the Creative Commons Attribution.
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import os
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import inverse_sigmoid


class BoxRegressionHead(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class AttrHead(nn.Module):

    def __init__(self, d_model, num_attr_classes, num_od_classes):
        super().__init__()
        self.od_cls_embed = nn.Embedding(num_od_classes, d_model)
        self.attr_linear1 = nn.Linear(d_model + d_model, d_model)
        self.attr_linear2 = nn.Linear(d_model, num_attr_classes)

    def forward(self, obj_h, pred_logits):
        # pred_logits: [B, num_queries, num_classes]
        # obj_h: [B, num_queries, d_model]
        pred_scores = torch.sigmoid(pred_logits)
        best_labels = torch.argmax(pred_scores, dim=-1)  # [B, num_queries]
        cls_embed = self.od_cls_embed(best_labels)  # [B, num_queries, d_model]

        attr = torch.cat([obj_h, cls_embed], dim=-1)
        attr = self.attr_linear1(attr)
        attr_logits = self.attr_linear2(F.relu(attr))
        return {'attr_logits': attr_logits}  # [B, num_queries, num_attr_classes]


class BBoxHeads(nn.Module):

    def __init__(self, d_model=768, num_aux_layers=6, num_od_classes=1849):
        super().__init__()
        self.num_aux_layers = num_aux_layers

        self.bbox_cls = nn.Linear(d_model, num_od_classes)
        self.bbox_reg = BoxRegressionHead(d_model, d_model, 4, 3)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.bbox_cls.bias.data = torch.ones(num_od_classes) * bias_value
        nn.init.constant_(self.bbox_reg.layers[-1].bias.data[2:], -2.0)

        self.bbox_cls = nn.ModuleList([self.bbox_cls for _ in range(num_aux_layers)])
        self.bbox_reg = nn.ModuleList([self.bbox_reg for _ in range(num_aux_layers)])

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def forward(self, obj_hs, inter_references):
        outputs_classes = []
        outputs_coords = []

        for aux_idx in range(obj_hs.shape[0]):
            reference = inter_references[aux_idx]  # [B, num_queries, 2]

            reference = inverse_sigmoid(reference)  # [B, num_queries, 2]
            outputs_class = self.bbox_cls[aux_idx](obj_hs[aux_idx])  # [B, num_queries, num_classes]

            tmp = self.bbox_reg[aux_idx](obj_hs[aux_idx])  # [B, num_queries, 4]
            assert reference.shape[-1] == 2
            tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()  # [B, num_queries, 4]
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)  # [num_aux_layers, B, num_queries, num_classes]
        outputs_coord = torch.stack(outputs_coords)  # [num_aux_layers, B, num_queries, 4]

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.num_aux_layers > 1:
            # We have: len(out['aux_outputs']) : num_aux_layers - 1
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out
