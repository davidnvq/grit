import os
import hydra
import random
import numpy as np
import multiprocessing
from omegaconf import DictConfig

from datasets.caption.field import TextField
from datasets.caption.coco import build_coco_dataloaders
from datasets.caption.metrics import PTBTokenizer, Cider
from models.caption import Transformer, GridFeatureNetwork, CaptionGenerator
from models.caption.detector import build_detector
from models.common.attention import MemoryAttention
from tools.extract_features import extract_vis_features
from utils.cap_scheduler import CosineLRScheduler

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from engine.caption_engine import *


def main(gpu, config):
    # dist init
    torch.backends.cudnn.enabled = False
    rank = config.exp.rank * config.exp.ngpus_per_node + gpu
    dist.init_process_group('nccl', 'env://', rank=rank, world_size=config.exp.world_size)

    torch.manual_seed(config.exp.seed)
    np.random.seed(config.exp.seed)
    random.seed(config.exp.seed)

    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(gpu)

    # extract features
    detector = build_detector(config).to(device)

    if os.path.exists(config.model.detector.checkpoint):
        checkpoint = torch.load(config.model.detector.checkpoint, map_location='cpu')
        missing, unexpected = detector.load_state_dict(checkpoint['model'], strict=False)
        print("det missing:", len(missing))
        print("det unexpected:", len(unexpected))

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
        dropout=config.model.dropout,
        attn_dropout=config.model.attn_dropout,
        cfg=config.model.cap_generator,
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

    start_epoch = 0
    best_cider_val = 0.0
    best_cider_test = 0.0
    if os.path.exists(config.exp.checkpoint):
        checkpoint = torch.load(config.exp.checkpoint, map_location='cpu')
        missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("model missing:", len(missing))
        print("model unexpected:", len(unexpected))
        if 'backbone' in checkpoint:
            model.detector.backbone.load_state_dict(checkpoint['backbone'], strict=False)
        if config.exp.resume:
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resume model at epoch {start_epoch} from checkpoint {config.exp.checkpoint}")
            if 'best_ciders' in checkpoint:
                best_cider_val, best_cider_test = checkpoint['best_ciders']

    if not config.dataset.overfit and start_epoch < config.optimizer.freezing_xe_epochs:
        if getattr(config.optimizer, 'freeze_backbone', False):
            for p, n in model.named_parameters():
                if 'backbone' in n:
                    p.requires_grad = False

        if getattr(config.optimizer, 'freeze_detector', False):
            for p, n in model.named_parameters():
                if 'detector' in n:
                    p.requires_grad = False
        else:
            extract_vis_features(detector, config, device, rank)

    model = DDP(model, device_ids=[gpu], find_unused_parameters=True, broadcast_buffers=False)
    optimizers = build_optimizers(model, config, mode='xe')

    # tensorboard:
    writer = SummaryWriter(log_dir='tensorboard') if rank == 0 or rank == 1 else None

    # train with freezing xe
    if not config.dataset.overfit and start_epoch < config.optimizer.freezing_xe_epochs \
        and not getattr(config.optimizer, 'freeze_backbone', False):
        model.module.cached_features = True
        dataloaders, samplers = build_coco_dataloaders(config, mode='freezing', device=device)
    else:
        model.module.cached_features = False
        dataloaders, samplers = build_coco_dataloaders(config, mode='finetune', device=device)

    text_field = TextField(vocab_path=config.dataset.vocab_path)
    train_dataset = dataloaders['train'].dataset
    cider = Cider(PTBTokenizer.tokenize([e.text for e in train_dataset.examples]))
    tokenizer = multiprocessing.Pool(8)  #config.optimizer.num_workers)

    scheduler = CosineLRScheduler(
        optimizers['model'],
        num_epochs=config.optimizer.freezing_xe_epochs + config.optimizer.finetune_xe_epochs,
        num_its_per_epoch=len(dataloaders['train']),
        init_lr=config.optimizer.xe_lr,
        min_lr=config.optimizer.min_lr,
        warmup_init_lr=config.optimizer.warmup_init_lr,
    )

    fr_xe_epochs = config.optimizer.freezing_xe_epochs  # 10
    fr_sc_epochs = fr_xe_epochs + config.optimizer.freezing_sc_epochs  # 15
    ft_xe_epochs = fr_sc_epochs + config.optimizer.finetune_xe_epochs  # 20
    ft_sc_epochs = ft_xe_epochs + config.optimizer.finetune_sc_epochs  # 20
    total_epochs = ft_sc_epochs
    for epoch in range(max(0, start_epoch), total_epochs):
        if epoch < fr_xe_epochs:
            phase = 'fr_xe'
        if fr_xe_epochs <= epoch < fr_sc_epochs:
            phase = 'fr_sc'
        if fr_sc_epochs <= epoch < ft_xe_epochs:
            phase = 'ft_xe'
        if ft_xe_epochs <= epoch < ft_sc_epochs:
            phase = 'ft_sc'

        if (phase == 'ft_sc' or phase == 'ft_xe') and dataloaders['train'].dataset.image_field.use_hdf5_feat:
            model.module.cached_features = False
            dataloaders, samplers = build_coco_dataloaders(config, mode='finetune', device=device)

        if (phase == 'fr_sc' or phase == 'ft_sc') and optimizers['mode'] == 'xe':
            optimizers = build_optimizers(model, config, mode='sc')

        if (phase == 'fr_xe' or phase == 'ft_xe') and optimizers['mode'] == 'sc':
            optimizers = build_optimizers(model, config, mode='xe')

        print(f"Train: rank={rank}, epoch={epoch}, phase={phase}")
        if phase == 'fr_xe' or phase == 'ft_xe':
            train_res = train_xe(
                model,
                dataloaders,
                optimizers=optimizers,
                text_field=text_field,
                epoch=epoch,
                rank=rank,
                config=config,
                scheduler=scheduler,
                writer=writer,
            )
            samplers['train'].set_epoch(epoch)

        elif phase == 'fr_sc' or phase == 'ft_sc':
            checkpoint = torch.load('checkpoint_best_valid.pth', map_location='cpu')
            missing, unexpected = model.module.load_state_dict(checkpoint['state_dict'], strict=False)
            print(f"Start self-critical optimization: missing={len(missing)}, unexpected={len(unexpected)}")
            train_res = train_sc(
                model,
                dataloaders,
                optimizers=optimizers,
                cider=cider,
                text_field=text_field,
                tokenizer_pool=tokenizer,
                device=device,
                epoch=epoch,
                rank=rank,
                config=config,
                writer=writer,
            )
            samplers['train_dict'].set_epoch(epoch)

        if rank == 0:
            best_cider_val = evaluate_metrics(
                model,
                optimizers,
                dataloader=dataloaders['valid_dict'],
                text_field=text_field,
                epoch=epoch,
                split='valid',
                config=config,
                train_res=train_res,
                writer=writer,
                best_cider=best_cider_val,
                which=phase,
                scheduler=scheduler,
            )

        if rank == 1:
            best_cider_test = evaluate_metrics(
                model,
                optimizers,
                dataloader=dataloaders['test_dict'],
                text_field=text_field,
                epoch=epoch,
                split='test',
                config=config,
                train_res=train_res,
                writer=writer,
                best_cider=best_cider_test,
                which=phase,
                scheduler=scheduler,
            )

        if rank == 0:
            save_checkpoint(
                model,
                optimizers,
                epoch=epoch,
                scores=[],
                best_ciders=[0, 0],
                config=config,
                filename=f'checkpoint_{phase}.pth',
                scheduler=scheduler,
            )
            if epoch >= 15:
                save_checkpoint(
                    model,
                    optimizers,
                    epoch=epoch,
                    scores=[],
                    best_ciders=[0, 0],
                    config=config,
                    filename=f'checkpoint_{epoch}.pth',
                    scheduler=scheduler,
                )

        torch.distributed.barrier()


@hydra.main(config_path="configs/caption", config_name="coco_config", version_base=None)
def run_main(config: DictConfig) -> None:
    mp.spawn(main, nprocs=config.exp.ngpus_per_node, args=(config,))


if __name__ == "__main__":
    # os.environ["DATA_ROOT"] = "/home/quang/datasets/coco_caption"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6688"
    run_main()