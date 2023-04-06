import os
import hydra
import random
import numpy as np
from omegaconf import DictConfig

from datasets.caption.field import TextField
from datasets.caption.coco import build_coco_dataloaders
from models.caption import Transformer
from models.caption.detector import build_detector

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from engine.caption_engine import *

import torch

# model
from models.common.attention import MemoryAttention
from models.caption.detector import build_detector
from models.caption import Transformer, GridFeatureNetwork, CaptionGenerator

# dataset
from PIL import Image
from datasets.caption.field import TextField
from datasets.caption.transforms import get_transform
from engine.utils import nested_tensor_from_tensor_list


@hydra.main(config_path="configs/caption", config_name="coco_config")
def run_main(config: DictConfig) -> None:
    device = torch.device(f"cuda:0")
    detector = build_detector(config).to(device)
    model = Transformer(detector=detector, config=config)
    model = model.to(device)

    # load checkpoint
    if os.path.exists(config.exp.checkpoint):
        checkpoint = torch.load(config.exp.checkpoint, map_location='cpu')
        missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"model missing:{len(missing)} model unexpected:{len(unexpected)}")

    model.cached_features = False

    # prepare utils
    transform = get_transform(config.dataset.transform_cfg)['valid']
    text_field = TextField(vocab_path=config.vocab_path if 'vocab_path' in config else config.dataset.vocab_path)

    # load image
    rgb_image = Image.open(config.img_path).convert('RGB')
    image = transform(rgb_image)
    images = nested_tensor_from_tensor_list([image]).to(device)

    # inference and decode
    with torch.no_grad():
        out, _ = model(
            images,
            seq=None,
            use_beam_search=True,
            max_len=config.model.beam_len,
            eos_idx=config.model.eos_idx,
            beam_size=config.model.beam_size,
            out_size=1,
            return_probs=False,
        )
        caption = text_field.decode(out, join_words=True)[0]
        print(caption)


if __name__ == "__main__":
    run_main()