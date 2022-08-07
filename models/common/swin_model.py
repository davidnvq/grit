# --------------------------------------------------------
# Modified from Swin Transformer
# Copyright (c) 2021 Microsoft
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------

from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import os


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def masked_sin_pos_encoding(x, mask, num_pos_feats, temperature=10000, scale=2 * math.pi):
    """ Masked Sinusoidal Positional Encoding

    Parameters:
        x: [PATCH] tokens
        mask: the padding mask for [PATCH] tokens
        num_pos_feats: the size of channel dimension
        temperature: the temperature value
        scale: the normalization scale

    Returns:
        pos: Sinusoidal positional encodings
    """

    num_pos_feats = num_pos_feats // 2
    not_mask = ~mask

    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)

    eps = 1e-6
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t

    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3)

    return pos


def window_partition(x, window_size):
    """
    Parameters:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Parameters:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim,
                                    window_size=to_2tuple(self.window_size),
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attn_drop=attn_drop,
                                    proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Parameters:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, expand=True, pos_dim=768):
        super().__init__()
        self.dim = dim

        # if expand is True, the channel size will be expanded, otherwise, return 256 size of channel
        expand_dim = 2 * dim if expand else pos_dim
        self.reduction = nn.Linear(4 * dim, expand_dim, bias=False)
        self.norm = norm_layer(4 * dim)

        # added for detection token [please ignore, not used for training]
        # not implemented yet.
        self.expansion = nn.Linear(dim, expand_dim, bias=False)
        self.norm2 = norm_layer(dim)

    def forward(self, x, H, W):
        """ Forward function.

        Parameters:
            x: Input feature, tensor size (B, H*W, C), i.e., binded [PATCH, DET] tokens
            H, W: Spatial resolution of the input feature.

        Returns:
            x
        """

        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Parameters:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 last=False,
                 use_checkpoint=False):

        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim,
                                 num_heads=num_heads,
                                 window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 qk_scale=qk_scale,
                                 drop=drop,
                                 attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer) for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, expand=(not last))
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.

        Parameters:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        B = x.shape[0]

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # mask for cyclic shift
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for n_blk, blk in enumerate(self.blocks):
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        # reduce the number of patch tokens, but maintaining a fixed-scale det tokens
        # meanwhile, the channel dim increases by a factor of 2
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Parameters:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""

        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class SwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Parameters:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            pretrain_img_size=224,
            patch_size=4,
            in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            out_indices=[1, 2, 3],  # not used in the current version, please ignore.
            frozen_stages=-1,
            use_checkpoint=False,
            pos_dim=768):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim  # dim in the first patch_embed
        self.ape = ape  # False
        self.patch_norm = patch_norm  # True
        self.out_indices = out_indices  # [1,2,3] <- but not used
        self.frozen_stages = frozen_stages  # -1 Not frozen???

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,  # 4
            in_chans=in_chans,  # 3 RGB
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)  # LayerNorm

        # absolute position embedding: Not used in ViDT
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)  # drop_rate=0.0 for swin_nano

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):  # has 4 stages/layers
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                # Adding the downsample (H/64, W/64)
                downsample=partial(PatchMerging, pos_dim=pos_dim) if (i_layer < self.num_layers) else None,
                last=None if (i_layer < self.num_layers - 1) else True,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        # Not used in the current version -> please ignore. this error will be fixed later
        # we leave this lines to load the pre-trained model ...
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

        self.pos_dim = pos_dim
        self.num_channels = [self.num_features[i + 1] for i in range(len(self.num_features) - 1)]
        self.num_channels.append(self.pos_dim)  # H/64, W/64

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x):
        """ Forward function.

            Parameters:
                x: input rgb images

            Returns:
                patch_outs: multi-scale [PATCH] tokens (four scales are used)
                    these tokens are the first input of the neck decoder
        """

        # original input shape
        B, ori_H, ori_W = x.shape[0], x.shape[2], x.shape[3]

        # patch embedding
        x = self.patch_embed(x)  #[B, C=48, Wh=ori_H/4, Ww=ori_W/4]

        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)  # [B, C, Wh*Ww]

        outs = []
        for stage in range(self.num_layers):
            layer = self.layers[stage]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            # Aggregate intermediate outputs
            if stage > 0:
                patch_out = x_out.view(B, H, W, -1).permute(0, 3, 1, 2)
                outs.append(patch_out)

        # patch token reduced from last stage output
        outs.append(x.view(B, Wh, Ww, -1).permute(0, 3, 1, 2))

        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()

    # not working in the current version
    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2**self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


def swin_nano(pretrained=None, pos_dim=384, **kwargs):
    pos_dim = 384 if pos_dim is None else pos_dim
    model = SwinTransformer(pretrain_img_size=[224, 224],
                            embed_dim=48,
                            depths=[2, 2, 6, 2],
                            num_heads=[3, 6, 12, 24],
                            window_size=7,
                            drop_path_rate=0.0,
                            pos_dim=pos_dim,
                            **kwargs)

    if pretrained is None or pretrained == 'none':
        return model, 384

    if pretrained is not None:
        if pretrained == 'imagenet':
            path = 'pretrained_weights/swin_nano_patch4_window7_224.pth'
            path = os.path.join(os.environ['OUTPUT'], 'checkpoints/eccv', path)
            if not os.path.exists(path):
                torch.hub._download_url_to_file(
                    url="https://github.com/naver-ai/vidt/releases/download/v0.1-swin/swin_nano_patch4_window7_224.pth",
                    dst=path)
            checkpoint = torch.load(path, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone pretrained on ImageNet 1K')

        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone in the given path')

    return model, 384


def swin_tiny(pretrained=None, pos_dim=768, **kwargs):
    pos_dim = 768 if pos_dim is None else pos_dim
    model = SwinTransformer(pretrain_img_size=[224, 224],
                            embed_dim=96,
                            depths=[2, 2, 6, 2],
                            num_heads=[3, 6, 12, 24],
                            window_size=7,
                            drop_path_rate=0.2,
                            pos_dim=pos_dim,
                            **kwargs)

    if pretrained is None or pretrained == 'none':
        return model, 768

    if pretrained is not None:
        if pretrained == 'imagenet':
            path = 'pretrained_weights/swin_tiny_patch4_window7_224.pth'
            path = os.path.join(os.environ['OUTPUT'], 'checkpoints/eccv', path)
            if not os.path.exists(path):
                torch.hub._download_url_to_file(
                    url=
                    "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
                    dst=path)
            checkpoint = torch.load(path, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone pretrained on ImageNet 1K')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone in the given path')
    return model, 768


def swin_small(pretrained=None, pos_dim=768, **kwargs):
    pos_dim = 768 if pos_dim is None else pos_dim
    model = SwinTransformer(pretrain_img_size=[224, 224],
                            embed_dim=96,
                            depths=[2, 2, 18, 2],
                            num_heads=[3, 6, 12, 24],
                            window_size=7,
                            drop_path_rate=0.3,
                            pos_dim=pos_dim,
                            **kwargs)

    if pretrained is None or pretrained == 'none':
        return model, 768

    if pretrained is not None:
        if pretrained == 'imagenet':

            path = "pretrained_weights/swin_small_patch4_window7_224.pth"
            path = os.path.join(os.environ['OUTPUT'], 'checkpoints/eccv', path)
            if not os.path.exists(path):
                torch.hub._download_url_to_file(
                    url=
                    "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth",
                    dst=path)
            checkpoint = torch.load(path, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone pretrained on ImageNet 1K')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone pretrained on ImageNet 1K')
    return model, 768


def swin_base_win7_224(pretrained=None, pos_dim=1024, **kwargs):
    pos_dim = 1024 if pos_dim is None else pos_dim
    model = SwinTransformer(pretrain_img_size=[224, 224],
                            embed_dim=128,
                            depths=[2, 2, 18, 2],
                            num_heads=[4, 8, 16, 32],
                            window_size=7,
                            drop_path_rate=0.3,
                            pos_dim=pos_dim,
                            **kwargs)

    if pretrained is None or pretrained == 'none':
        return model, 1024

    if pretrained is not None:
        if pretrained == 'imagenet':
            path = "pretrained_weights/swin_base_patch4_window7_224_22k.pth"
            path = os.path.join(os.environ['OUTPUT'], 'checkpoints/eccv', path)
            if not os.path.exists(path):
                torch.hub._download_url_to_file(
                    url=
                    "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth",
                    dst=path)
            checkpoint = torch.load(path, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone pretrained on ImageNet 22K')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone in the given path')
    return model, 1024


def swin_large_win7_224(pretrained=None, pos_dim=1024, **kwargs):
    pos_dim = 1024 if pos_dim is None else pos_dim
    model = SwinTransformer(pretrain_img_size=[224, 224],
                            embed_dim=192,
                            depths=[2, 2, 18, 2],
                            num_heads=[6, 12, 24, 48],
                            window_size=7,
                            drop_path_rate=0.3,
                            pos_dim=pos_dim,
                            **kwargs)

    if pretrained is None or pretrained == 'none':
        return model, 1024

    if pretrained is not None:
        if pretrained == 'imagenet':
            path = "pretrained_weights/swin_large_patch4_window7_224_22k.pth"
            path = os.path.join(os.environ['OUTPUT'], 'checkpoints/eccv', path)
            torch.hub._download_url_to_file(
                url=
                "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth",
                dst=path)
            checkpoint = torch.load(path, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone pretrained on ImageNet 22K')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone in the given path')
    return model, 1024


def swin_base_win7_384(pretrained=None, pos_dim=1024, **kwargs):
    pos_dim = 1024 if pos_dim is None else pos_dim
    model = SwinTransformer(pretrain_img_size=[384, 384],
                            embed_dim=128,
                            depths=[2, 2, 18, 2],
                            num_heads=[4, 8, 16, 32],
                            window_size=12,
                            drop_path_rate=0.3,
                            pos_dim=pos_dim,
                            **kwargs)

    if pretrained is None or pretrained == 'none':
        return model, 1024

    if pretrained is not None:
        if pretrained == 'imagenet':
            path = "pretrained_weights/swin_base_patch4_window7_384_22k.pth"
            path = os.path.join(os.environ['OUTPUT'], 'checkpoints/eccv', path)
            if not os.path.exists(path):
                torch.hub._download_url_to_file(
                    url=
                    "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth",
                    dst=path)
            checkpoint = torch.load(path, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone pretrained on ImageNet 22K')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone in the given path')
    return model, 1024


def swin_large_win7_384(pretrained=None, pos_dim=1024, **kwargs):
    pos_dim = 1024 if pos_dim is None else pos_dim
    model = SwinTransformer(pretrain_img_size=[384, 384],
                            embed_dim=192,
                            depths=[2, 2, 18, 2],
                            num_heads=[6, 12, 24, 48],
                            window_size=7,
                            drop_path_rate=0.3,
                            pos_dim=pos_dim,
                            **kwargs)

    if pretrained is None or pretrained == 'none':
        return model, 1024

    if pretrained is not None:
        if pretrained == 'imagenet':
            path = "pretrained_weights/swin_large_patch4_window7_384_22k.pth"
            path = os.path.join(os.environ['OUTPUT'], 'checkpoints/eccv', path)
            if not os.path.exists(path):
                torch.hub._download_url_to_file(
                    url=
                    "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_384_22k.pth",
                    dst=path)
            checkpoint = torch.load(path, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone pretrained on ImageNet 22K')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone in the given path')
    return model, 1024


def build_backbone(backbone_name='swin_nano', frozen_stages=2, pre_trained='imagenet', pos_dim=None):
    # config at backbone level
    if backbone_name == 'swin_nano':
        backbone, hidden_dim = swin_nano(pretrained=pre_trained, frozen_stages=frozen_stages, pos_dim=pos_dim)
    elif backbone_name == 'swin_tiny':
        backbone, hidden_dim = swin_tiny(pretrained=pre_trained, frozen_stages=frozen_stages, pos_dim=pos_dim)
    elif backbone_name == 'swin_small':
        backbone, hidden_dim = swin_small(pretrained=pre_trained, frozen_stages=frozen_stages, pos_dim=pos_dim)
    elif backbone_name == 'swin_base_win7_224_22k':
        backbone, hidden_dim = swin_base_win7_224(pretrained=pre_trained, frozen_stages=frozen_stages, pos_dim=pos_dim)
    elif backbone_name == 'swin_large_win7_224_22k':
        backbone, hidden_dim = swin_large_win7_224(pretrained=pre_trained, frozen_stages=frozen_stages, pos_dim=pos_dim)
    elif backbone_name == 'swin_base_win7_384_22k':
        backbone, hidden_dim = swin_base_win7_384(pretrained=pre_trained, frozen_stages=frozen_stages, pos_dim=pos_dim)
    elif backbone_name == 'swin_large_win7_384_22k':
        backbone, hidden_dim = swin_large_win7_384(pretrained=pre_trained, frozen_stages=frozen_stages, pos_dim=pos_dim)
    else:
        raise ValueError(f'backbone {backbone_name} not supported')
    return backbone


class NullDetector(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone


def build_backbone_null_detector(backbone_name, frozen_stages, pre_trained='imagenet', checkpoint_path=None):
    backbone = build_backbone(backbone_name, frozen_stages=frozen_stages, pre_trained=pre_trained)
    detector = NullDetector(backbone)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        detector.load_state_dict(checkpoint['model'], strict=False)
    return detector.backbone
