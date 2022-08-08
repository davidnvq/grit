import os
import hydra
import random
import numpy as np
from omegaconf import DictConfig

from datasets.caption.field import TextField
from datasets.caption.coco import build_coco_dataloaders
from models.caption import Transformer, GridFeatureNetwork, CaptionGenerator

from models.caption.detector import build_detector
from models.common.attention import MemoryAttention

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from engine.caption_engine import *


def main(gpu, config):
    # dist init
    torch.backends.cudnn.enabled = False
    dist.init_process_group('nccl', 'env://', rank=0, world_size=1)

    torch.manual_seed(config.exp.seed)
    np.random.seed(config.exp.seed)
    random.seed(config.exp.seed)

    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(gpu)

    # extract reg features + initial grid features
    detector = build_detector(config).to(device)

    # wrap it into DDP to easily load the checkpoint
    detector = DDP(detector, device_ids=[gpu])

    grit_net = GridFeatureNetwork(
        pad_idx=config.model.pad_idx,
        d_in=config.model.grid_feat_dim,
        dropout=config.model.dropout,
        attn_dropout=config.model.attn_dropout,
        attention_module=MemoryAttention,
        **config.model.grit_net,
    )
    cap_generator = CaptionGenerator(
        vocab_size=config.model.vocab_size,
        max_len=config.model.max_len,
        pad_idx=config.model.pad_idx,
        cfg=config.model.cap_generator,
        dropout=config.model.dropout,
        attn_dropout=config.model.attn_dropout,
        **config.model.cap_generator,
    )
    model = Transformer(
        grit_net,
        cap_generator,
        detector=detector.module,
        use_gri_feat=config.model.use_gri_feat,
        use_reg_feat=config.model.use_reg_feat,
        config=config,
    )
    model = model.to(device)

    if os.path.exists(config.exp.checkpoint):
        checkpoint = torch.load(config.exp.checkpoint, map_location='cpu')
        missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("model missing:", len(missing))
        print("model unexpected:", len(unexpected))

    model = DDP(model, device_ids=[gpu], find_unused_parameters=True, broadcast_buffers=False)
    model.module.cached_features = False

    dataloaders, samplers = build_coco_dataloaders(config, mode='finetune', device=device)

    text_field = TextField(vocab_path=config.dataset.vocab_path)

    split = config.split
    print(f"Evaluating on split: {split}")
    scores = evaluate_metrics(
        model,
        optimizers=None,
        dataloader=dataloaders[f'{split}_dict'],
        text_field=text_field,
        epoch=-1,
        split=f'{split}',
        config=config,
        train_res=[],
        writer=None,
        best_cider=None,
        which='ft_sc',
        scheduler=None,
        log_and_save=False,
    )


@hydra.main(config_path="configs/caption", config_name="coco_config", version_base=None)
def run_main(config: DictConfig) -> None:
    mp.spawn(main, nprocs=1, args=(config,))


if __name__ == "__main__":
    if os.environ["USER"] == 'quang':
        os.environ["DATA_ROOT"] = "/home/quang/datasets/coco_caption"

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6688"
    run_main()