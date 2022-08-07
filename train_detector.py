import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.detection.detector import build_detector

from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import MultiStepLR
from utils import misc
from datasets.detection import build_train_dataset
from datasets.detection import build_valid_dataset

import torch.distributed as dist
import torch.multiprocessing as mp
from engine.utils import get_rank
from engine.hooks import *
from engine.det_solver import Trainer, Valider


def build_optimizers_schedulers(model, config):
    if hasattr(model.backbone, 'no_weight_decay'):
        skip = model.backbone.no_weight_decay()
    else:
        skip = ['query_embed']
    head = []
    det_no_decay = []
    backbone_decay = []
    backbone_no_decay = []
    sp_params = []
    sp_names = getattr(config.optimizer, 'sp_names', [])

    for name, param in model.named_parameters():
        if ("backbone" not in name and param.requires_grad) and not any(ns in name for ns in sp_names):
            if len(param.shape) == 1 or name.endswith(".bias") or name.split('.')[-1] in skip:
                det_no_decay.append(param)
            else:
                head.append(param)
        if "backbone" in name and param.requires_grad and not any(ns in name for ns in sp_names):
            if len(param.shape) == 1 or name.endswith(".bias") or name.split('.')[-1] in skip:
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)

        if param.requires_grad and any(ns in name for ns in sp_names):
            sp_params.append(param)

    param_dicts = [
        {
            "params": head
        },
        {
            "params": det_no_decay,
            "weight_decay": 0.,
            "lr": config.optimizer.lr,
        },
        {
            "params": backbone_no_decay,
            "weight_decay": 0.,
            "lr": config.optimizer.lr_backbone
        },
        {
            "params": backbone_decay,
            "lr": config.optimizer.lr_backbone
        },
    ]

    # print the total number of trainable params.
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('num of total trainable prams:' + str(n_parameters))

    optimizers = [torch.optim.AdamW(param_dicts, lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)]
    lr_schedulers = [
        MultiStepLR(optimizers[0], config.optimizer.lr_drop_epochs, verbose=True, gamma=config.optimizer.decay_rate)
    ]
    if len(sp_params) > 0:
        sp_optimizer = torch.optim.AdamW(sp_params,
                                         weight_decay=config.optimizer.weight_decay,
                                         lr=config.optimizer.sp_lr)
        optimizers.append(sp_optimizer)
        lr_schedulers.append(
            MultiStepLR(sp_optimizer,
                        config.optimizer.sp_lr_drop_epochs,
                        verbose=True,
                        gamma=config.optimizer.decay_rate))
    return optimizers, lr_schedulers


def main(gpu, config, overrides):
    # gpu: the rank of gpu in the node
    rank = config.exp.rank * config.exp.ngpus_per_node + gpu
    proj_dir = os.path.join(os.environ['OUTPUT'], 'workspace/ecaptioner')

    if gpu == 0:
        script_path = os.path.join(proj_dir, config.exp.script)
        os.system(f"rsync -av {script_path} run_{config.exp.rank}.sh")

        with open(os.path.join(proj_dir, config.exp.git_file), 'r') as f:
            git_info = f.read()
        with open(f'./run_{config.exp.rank}.sh', 'a') as f:
            f.write(f'{git_info}')
        os.system(f"rm {os.path.join(proj_dir, config.exp.git_file)}")

    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://',
                                         rank=rank,
                                         world_size=config.exp.world_size)
    print(f"Initialize: {rank}/{dist.get_world_size()}.")

    device = "cuda"
    torch.cuda.set_device(gpu)

    # fix the seed for reproducibility
    seed = config.exp.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # create model
    print("create model")
    model, criterion, postprocessors = build_detector(config)
    model.to(device)
    criterion.to(device)

    print("create dist model")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
    model_without_ddp = model.module
    optimizers, schedulers = build_optimizers_schedulers(model_without_ddp, config)

    start_epoch = 0
    if config.exp.checkpoint != "" and os.path.exists(config.exp.checkpoint):
        checkpoint = torch.load(config.exp.checkpoint, map_location=device)

        def create_new_dict(d):
            from collections import OrderedDict
            new_d = OrderedDict()
            for k, v in d.items():
                if 'query_embed' in k and 'query_embed' in config.optimizer.sp_names:
                    v = v[:config.model.det_module.num_queries]
                new_d[k] = v
            return new_d

        missing, unexpected = model_without_ddp.load_state_dict(
            create_new_dict(checkpoint['model']),
            strict=False,
        )
        if len(missing) > 0 and rank == 0:
            print('Missing Keys: {}'.format(len(missing)))
        if len(unexpected) > 0 and rank == 0:
            print('Unexpected Keys: {}'.format(len(unexpected)))
        if getattr(config.exp, 'resume', False):
            start_epoch = checkpoint['epoch'] + 1
            if 'optimizer' in checkpoint and not isinstance(optimizers, list):
                optimizers.load_state_dict(checkpoint['optimizer'])

        print(f"loading from the checkpoint: {config.exp.checkpoint}.")
        print(f"start at epoch: {start_epoch}.")

    # for datasets, dataloaders
    train_dataset = build_train_dataset(config.dataset)
    valid_datasets = build_valid_dataset(config.dataset_val)

    print("create dataloaders")
    train_sampler = DistributedSampler(train_dataset)
    valid_samplers = {k: DistributedSampler(v, shuffle=False) for k, v in valid_datasets.items()}

    batch_train_sampler = torch.utils.data.BatchSampler(train_sampler, config.optimizer.batch_size, drop_last=True)
    train_loader = DataLoader(train_dataset,
                              batch_sampler=batch_train_sampler,
                              prefetch_factor=2,
                              collate_fn=misc.collate_fn,
                              num_workers=config.optimizer.num_workers,
                              pin_memory=True)
    valid_loaders = {
        k: DataLoader(dataset,
                      config.optimizer.batch_size,
                      sampler=valid_samplers[k],
                      prefetch_factor=2,
                      drop_last=False,
                      collate_fn=misc.collate_fn,
                      num_workers=config.optimizer.num_workers,
                      pin_memory=True) for k, dataset in valid_datasets.items()
    }
    print("create trainers")
    trainer = Trainer(model,
                      train_loader,
                      optimizers,
                      criterion,
                      device=device,
                      max_norm=config.optimizer.clip_max_norm,
                      eval_every_iters=config.exp.eval_every_iters)
    validers = {
        data_name: Valider(
            model,
            valid_loaders[data_name],
            optimizers,
            criterion,
            postprocessors,
            device=device,
            rank=rank,
            data_name=data_name,
        ) for data_name in valid_loaders
    }

    excluded_keys = ["bbox"]
    for name in ["loss_bbox", "loss_giou", "loss_ce"]:
        for idx in range(6):
            excluded_keys.append(f"{name}_{idx}")

    train_hooks = [ProgressHook(name='train', excluded_keys=excluded_keys)]
    valid_hooks = {data_name: [ProgressHook(name=data_name, excluded_keys=excluded_keys)] for data_name in validers}

    if rank == 0:
        train_hooks += [
            TensorboardHook(name='train', save_dir='./', log_every_step=100),
            TextLoggingHook(name='train', save_dir='./'),
            CheckpointHook(save_every_epochs=getattr(config.exp, 'save_every_epochs', 1),
                           save_every_iters=-1,
                           save_dir='./',
                           args=config),
        ]
        for data_name in valid_hooks:
            valid_hooks[data_name] += [
                TensorboardHook(name=data_name, save_dir='./', log_every_step=100),
                TextLoggingHook(name=data_name, save_dir='./'),
            ]

    trainer.set_validers(validers)
    trainer.register_hooks(train_hooks)
    for data_name, valider in validers.items():
        valider.register_hooks(valid_hooks[data_name])

    if rank == 0:
        trainer.hooks[trainer.hook_name2idx["TextLoggingHook"]].write(OmegaConf.to_yaml(config))
    if getattr(config.exp, 'eval', False):
        for data_name, valider in validers.items():
            print(f"Evaluate {data_name}...")
            valider.run_epoch(0)
        return

    print("start training..")
    for lr_scheduler in schedulers:
        lr_scheduler.step()
    for epoch in range(start_epoch, config.optimizer.num_epochs):
        train_sampler.set_epoch(epoch)
        trainer.run_epoch(epoch)
        for data_name, valider in validers.items():
            print(f"Evaluate {data_name:<10} at epoch {epoch} ...")
            valider.run_epoch(epoch)

        for lr_scheduler in schedulers:
            lr_scheduler.step()


@hydra.main(config_path="configs/detection", config_name="train_config")
def run_main(config: DictConfig) -> None:
    overrides = HydraConfig.get().overrides.task
    mp.spawn(main, nprocs=config.exp.ngpus_per_node, args=(config, overrides))


if __name__ == "__main__":
    run_main()
