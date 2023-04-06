import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from models.common.swin_model import *
from utils.misc import nested_tensor_from_tensor_list, NestedTensor
from models.detection.det_module import build_det_module_with_config


class Detector(nn.Module):

    def __init__(
        self,
        backbone,
        det_module=None,
        use_gri_feat=True,
        use_reg_feat=True,
        hidden_dim=256,
    ):
        super().__init__()
        self.backbone = backbone
        self.use_gri_feat = use_gri_feat
        self.use_reg_feat = use_reg_feat

        if self.use_reg_feat:
            self.det_module = det_module
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[i], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ) for i in range(len(backbone.num_channels))
            ])

    def forward(self, images: NestedTensor):
        # - images.tensor: batched images, of shape [batch_size x 3 x H x W]
        # - images.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        if isinstance(images, (list, torch.Tensor)):
            samples = [img for img in images]
            samples = nested_tensor_from_tensor_list(samples)

        x = images.tensors  # RGB input # [B, 3, H, W]
        mask = images.mask  # padding mask [B, H, W]

        # features = [[B, C1, H1, W1], [B, C2, H2, W2], [B, C3, H3, W3], [B, C4, H4, W4]]
        features = self.backbone(x)

        masks = [
            F.interpolate(mask[None].float(), size=f.shape[-2:]).to(torch.bool)[0] for l, f in enumerate(features)
        ]  # masks [[B, Hi, Wi]]

        outputs = {}
        outputs['gri_feat'] = rearrange(features[-1], 'b c h w -> b (h w) c')
        outputs['gri_mask'] = repeat(masks[-1], 'b h w -> b 1 1 (h w)')

        if self.use_reg_feat:
            srcs = [self.input_proj[l](src) for l, src in enumerate(features)]
            hs, _, _ = self.det_module(srcs, masks)
            outputs['reg_feat'] = hs[-1]
            outputs['reg_mask'] = hs[-1].data.new_full((hs[-1].shape[0], 1, 1, hs[-1].shape[1]), 0).bool()
        return outputs


def build_detector(config):
    pos_dim = getattr(config.model.detector, 'pos_dim', None)
    backbone, _ = swin_base_win7_384(
        frozen_stages=config.model.frozen_stages,
        pos_dim=pos_dim,
    )
    det_cfg = config.model.detector
    det_module = build_det_module_with_config(det_cfg) if config.model.use_reg_feat else None
    detector = Detector(
        backbone,
        det_module=det_module,
        hidden_dim=config.model.d_model,
        use_gri_feat=config.model.use_gri_feat,
        use_reg_feat=config.model.use_reg_feat,
    )
    if os.path.exists(config.model.detector.checkpoint):
        checkpoint = torch.load(config.model.detector.checkpoint, map_location='cpu')
        missing, unexpected = detector.load_state_dict(checkpoint['model'], strict=False)
        print(f"Loading weights for detector: missing: {len(missing)}, unexpected: {len(unexpected)}.")
    return detector
