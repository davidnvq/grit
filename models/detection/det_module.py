# ------------------------------------------------------------------------
# GRIT: Faster and Better Image captioning Transformer
# Licensed under the Creative Commons Attribution.
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import normal_

from utils.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn

from timm.models.layers import DropPath


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DetectionModule(nn.Module):
    """ A Deformable Transformer for the neck in a detector

    Parameters:
        d_model: the channel dimension for attention [default=256]
        nhead: the number of heads [default=8]
        num_decoder_layers: the number of decoding layers [default=6]
        dim_feedforward: the channel dim of point-wise FFNs [default=1024]
        dropout: the degree of dropout used in FFNs [default=0.1]
        activation: An activation function to use [default='relu']
        return_intermediate_dec: whether to return all the indermediate outputs [default=True]
        num_feature_levels: the number of scales for extracted features [default=4]
        dec_n_points: the number of reference points for deformable attention [default=4]
        drop_path: the ratio of stochastic depth for decoding layers [default=0.0]
        
        num_classes: number of object classes
        num_queries: number of object queries (i.e., det tokens). This is the maximal number of objects
                        DETR can detect in a single image. For COCO, we recommend 100 queries.
        aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        with_box_refine: iterative bounding box refinement

    """

    def bbox_refine(self, bbox_embed, output, reference_points):
        # hack implementation for iterative bounding box refinement
        if bbox_embed is not None:
            tmp = bbox_embed(output)
            if reference_points.shape[-1] == 4:
                new_reference_points = tmp + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
            else:
                assert reference_points.shape[-1] == 2
                new_reference_points = tmp
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
            reference_points = new_reference_points.detach()
        return reference_points

    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=4,
        dec_n_points=4,
        drop_path=0.,
        num_classes=81,
        aux_loss=True,
        with_box_refine=True,
        num_queries=100,
    ):
        super().__init__()
        # two essential techniques used [default use]
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine

        self.d_model = d_model
        self.nhead = nhead
        decoder_layer = DeformableTransformerDecoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            dec_n_points,
            drop_path=drop_path,
        )

        # for decoder:
        self.decoder_layers = _get_clones(decoder_layer, num_decoder_layers)
        self.num_decoder_layers = num_decoder_layers
        self.return_intermediate = return_intermediate_dec
        self.reference_points = nn.Linear(d_model, 2)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.class_embed = nn.Linear(d_model, num_classes)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        self.query_embed = nn.Embedding(num_queries, d_model * 2)

        # initialize detection head: box regression and classification
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # the prediction is made for each decoding layers + the standalone detector (Swin with RAM)
        num_pred = num_decoder_layers + 1

        # set up all required nn.Module for additional techniques
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def prepare_od_inputs(self, srcs, masks):
        query_embed = self.query_embed.weight  # [num_queries, C*2]
        query_pos, query_tgt = torch.split(query_embed, query_embed.shape[1] // 2, dim=1)  # -> 2 [num_queries, C]
        query_pos = query_pos.unsqueeze(0).expand(srcs[0].shape[0], -1, -1)  # [B, num_queries, C]
        query_tgt = query_tgt.unsqueeze(0).expand(srcs[0].shape[0], -1, -1)  # [B, num_queries, C]

        # prepare input for the Transformer decoder
        src_flatten = []  # [B, all_tokens, C]
        mask_flatten = []
        spatial_shapes = []
        for lvl, (src, mask) in enumerate(zip(srcs, masks)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)  # [B, H1*W1 + h2*w2 +...+h4*w4, C] has 4 levels
        mask_flatten = torch.cat(mask_flatten, 1)  # [B, H1*W1 + h2*w2 +...+h4*w4, C] has 4 levels
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long,
                                         device=src_flatten.device)  #[[h1, w1], [h2, w2],...[h4,w4]]
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))  # [0, h1*w1, h2*w2, h3*w3]
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)  # [B, num_levels, 2]
        # ^ [[for img0 - [wratio_lvl0, hratio_lvl0],[wratio_lvl1, hratio_lvl1],...],
        #    [for img1 - [wratio_lvl0, hratio_lvl0],[wratio_lvl1, hratio_lvl1],...], for batch]

        bs, _, c = src_flatten.shape
        query_pos = query_pos.expand(bs, -1, -1)  # [B, num_queries, C]

        # reference points for deformable attention
        reference_points = self.reference_points(query_pos).sigmoid()  # [B, num_queries, 2]
        reference_points = self.bbox_refine(self.bbox_embed[0], query_tgt, reference_points)

        return {
            'tgt': query_tgt,
            'src': src_flatten,
            'src_spatial_shapes': spatial_shapes,
            'src_level_start_index': level_start_index,
            'src_valid_ratios': valid_ratios,
            'src_padding_mask': mask_flatten,
            'query_pos': query_pos,
            'reference_points': reference_points
        }

    def forward(self, srcs, masks):
        """ The forward step of the decoder

        Parameters:
            srcs: [Patch] tokens
            masks: input padding mask
            query_embed: [DET] tokens - [B, num_queries, C*2] (tgt, pos)

        Returns:
            hs: calibrated [DET] tokens
            init_reference_out: init reference points
            inter_references_out: intermediate reference points for box refinement
            enc_token_class_unflat: info. for token labeling
        """

        od_inputs = self.prepare_od_inputs(srcs, masks)
        init_reference_out = od_inputs['reference_points']

        intermediate = []
        intermediate_reference_points = []
        if self.return_intermediate:
            intermediate.append(od_inputs['tgt'])
            intermediate_reference_points.append(init_reference_out)

        for lid, layer in enumerate(self.decoder_layers):
            # deformable operation
            od_inputs['tgt'] = layer(**od_inputs)

            # recompute reference_points
            bbox_embed = self.bbox_embed[lid + 1] if self.bbox_embed is not None else None
            od_inputs['reference_points'] = self.bbox_refine(bbox_embed, od_inputs['tgt'],
                                                             od_inputs['reference_points'])

            if self.return_intermediate:
                intermediate.append(od_inputs['tgt'])
                intermediate_reference_points.append(od_inputs['reference_points'])

        if self.return_intermediate:
            # decoder: hs = [n_levels, B, num_queries, C], inter_references = [B, queries, 4]
            hs = torch.stack(intermediate)
            inter_references = torch.stack(intermediate_reference_points)
        else:
            hs = od_inputs['tgt']
            inter_references = od_inputs['reference_points']

        inter_references_out = inter_references
        return hs, init_reference_out, inter_references_out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.

        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def detection_head(self, hs, init_reference, inter_references):
        # perform predictions via the detection head
        # hs: [num_outputs, B, num_queries, C]
        # init_reference: [B, num_queries, 2]
        # inter_references: [num_outputs, B, num_queries, 2]
        outputs_classes = []
        outputs_coords = []
        if self.training:
            for lvl in range(hs.shape[0]):
                reference = init_reference if lvl == 0 else inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)

                outputs_class = self.class_embed[lvl](hs[lvl])
                ## bbox output + reference
                tmp = self.bbox_embed[lvl](hs[lvl])
                if reference.shape[-1] == 4:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 2
                    tmp[..., :2] += reference

                outputs_coord = tmp.sigmoid()
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)

            # stack all predictions made from each decoding layers
            outputs_class = torch.stack(outputs_classes)  # [1 + n_decoder_layers, num_queries, num_classes]
            outputs_coord = torch.stack(outputs_coords)  # [1 + n_decoder_layers, num_queries, 4]

            # final prediction is made the last decoding layer
            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

            # aux loss is defined by using the rest predictions
            if self.aux_loss and self.num_decoder_layers > 0:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

            return out
        else:  # Run the below.
            reference = inter_references[-2]
            reference = inverse_sigmoid(reference)
            last_layer = -1
            outputs_class = self.class_embed[last_layer](hs[last_layer])
            ## bbox output + reference
            tmp = self.bbox_embed[last_layer](hs[last_layer])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            # final prediction is made the last decoding layer
            out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
            return out


class DeformableTransformerDecoderLayer(nn.Module):
    """ A decoder layer.

    Parameters:
        d_model: the channel dimension for attention [default=256]
        d_ffn: the channel dim of point-wise FFNs [default=1024]
        dropout: the degree of dropout used in FFNs [default=0.1]
        activation: An activation function to use [default='relu']
        n_levels: the number of scales for extracted features [default=4]
        n_heads: the number of heads [default=8]
        n_points: the number of reference points for deformable attention [default=4]
        drop_path: the ratio of stochastic depth for decoding layers [default=0.0]
    """

    def __init__(self,
                 d_model=256,
                 d_ffn=1024,
                 dropout=0.1,
                 activation="relu",
                 n_levels=4,
                 n_heads=8,
                 n_points=4,
                 drop_path=0.):
        super().__init__()

        # [DET x PATCH] deformable cross-attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # [DET x DET] self-attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn for multi-heaed
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self,
                tgt,
                query_pos,
                reference_points,
                src,
                src_spatial_shapes,
                src_level_start_index,
                src_valid_ratios,
                src_padding_mask=None):

        if reference_points.shape[-1] == 4:
            reference_points = reference_points[:, :, None] \
                                        * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
        else:
            assert reference_points.shape[-1] == 2
            reference_points = reference_points[:, :, None] * src_valid_ratios[:, None]

        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos), reference_points, src, src_spatial_shapes,
                               src_level_start_index, src_padding_mask)

        if self.drop_path is None:
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            # ffn
            tgt = self.forward_ffn(tgt)
        else:
            tgt = tgt + self.drop_path(self.dropout1(tgt2))
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
            tgt = tgt + self.drop_path(self.dropout4(tgt2))
            tgt = self.norm3(tgt)

        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""

    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_det_module_with_config(cfg):
    return DetectionModule(
        d_model=cfg.d_model,
        nhead=cfg.num_heads,
        num_decoder_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        activation=cfg.activation,
        num_classes=cfg.num_classes,
        num_feature_levels=cfg.num_levels,
        dec_n_points=cfg.num_points,
        num_queries=cfg.num_queries,
        return_intermediate_dec=cfg.return_intermediate,
        aux_loss=getattr(cfg, 'aux_loss', False),
        with_box_refine=cfg.with_box_refine,
    )
