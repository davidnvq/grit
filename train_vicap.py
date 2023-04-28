import os
import hydra
import random
import numpy as np
import multiprocessing
from omegaconf import DictConfig

from datasets.caption.field import TextField
from datasets.caption.coco import build_coco_dataloaders
from datasets.caption.metrics import PTBTokenizer, Cider
from models.caption import Transformer
from models.caption.detector import build_detector
from tools.extract_features import extract_vis_features
from utils.cap_scheduler import CosineLRScheduler

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from engine.caption_engine import *
from vicap_dataset import *

import os
import time
import json
import torch
import itertools
import numpy as np
from tqdm import tqdm
from datasets.caption import metrics
from torch.nn import NLLLoss
import torch.distributed as dist
from engine.utils import NestedTensor


def evaluate_metrics(
    model,
    optimizers,
    dataloader,
    epoch=0,
    split='test',
    config=None,
    which='ft_xe',
    scheduler=None,
):

    model.eval()
    pred_captions = {}
    gt_captions = {}
    vocab = dataloader.dataset.vocab
    for it, batch in enumerate(iter(dataloader)):
        with torch.no_grad():
            out, _ = model(
                batch['samples'],
                seq=None,
                use_beam_search=True,
                max_len=config.model.beam_len,
                eos_idx=config.model.eos_idx,
                beam_size=config.model.beam_size,
                out_size=1,
                return_probs=False,
            )

        torch.cuda.synchronize()

        # decode and compute scores
        out = out.cpu().numpy()

        batch_predictions = []
        for token_ids in out:
            caption = []
            for token_id in token_ids:
                token = vocab.itos[token_id]
                if token == "<eos>":
                    break
                if token in ["<pad>", "<unk>", "<sos>", "eos"]:
                    continue
                caption.append(token)
            batch_predictions.append(" ".join(caption))

        bs = batch['samples'].tensors.shape[0]

        for i, (gts_i, pred_i) in enumerate(zip(batch['captions'], batch_predictions)):
            pred_captions[f'{it}_{i}'] = [pred_i]
            gt_captions[f'{it}_{i}'] = gts_i

    scores = metrics.compute_scores(gt_captions, pred_captions)[0]
    print(f'Epoch {epoch}: {split} scores: ' + str(scores) + '\n')
    return scores


def evaluate_loss(model, dataloader, loss_fn, epoch):
    vocab = dataloader.dataset.vocab

    model.eval()

    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % epoch, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, batch in enumerate(dataloader):
                out = model(batch['samples'], batch['captions'])

                captions_gt = batch['captions'][:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, out.shape[-1]), captions_gt.view(-1))

                loss = gather_result(loss)
                running_loss += loss.item()

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss


def train_xe(
    model,
    dataloaders,
    optimizers,
    epoch,
    rank=0,
    config=None,
    scheduler=None,
):
    vocab = dataloaders['train'].dataset.vocab

    model.train()
    loss_fn = NLLLoss(ignore_index=vocab.stoi['<pad>'])
    if scheduler is not None:
        scheduler.step()
    running_loss = .0
    with tqdm(desc=f'Epoch {epoch} - train', unit='it', total=len(dataloaders['train'])) as pbar:
        for it, batch in enumerate(dataloaders['train']):
            out = model(batch['samples'], batch['captions'])
            optimizers['model'].zero_grad()
            optimizers['backbone'].zero_grad()

            captions_gt = batch['captions'][:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            loss = loss_fn(out.view(-1, out.shape[-1]), captions_gt.view(-1))
            loss.backward()

            optimizers['model'].step()
            optimizers['backbone'].step()

            loss = gather_result(loss)
            running_loss += loss.item()

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()

            if scheduler is not None:
                lr = scheduler.step()

    train_loss = running_loss / len(dataloaders['train'])
    val_loss = evaluate_loss(model, dataloaders['valid'], loss_fn, epoch)
    torch.distributed.barrier()
    return train_loss, val_loss


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

    # build dataloaders
    print("Building dataloaders...")
    from vicap_dataset import get_dataloaders
    samplers, dataloaders = get_dataloaders(device=device)

    # build models
    print("Building models...")
    detector = build_detector(config).to(device)

    model = Transformer(detector=detector, config=config)
    # load checkpoint
    if os.path.exists(config.exp.checkpoint):
        checkpoint = torch.load(config.exp.checkpoint, map_location='cpu')
        missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"det missing:{len(missing)} det unexpected:{len(unexpected)}")

    model.cached_features = False
    model = model.to(device)
    if config.optimizer.freeze_detector:
        for param in model.detector.parameters():
            param.requires_grad = False

    model = DDP(model, device_ids=[gpu], find_unused_parameters=True, broadcast_buffers=False)
    optimizers = build_optimizers(model, config, mode='xe')

    scheduler = CosineLRScheduler(
        optimizers['model'],
        num_epochs=config.optimizer.finetune_xe_epochs,
        num_its_per_epoch=len(dataloaders['train']),
        init_lr=config.optimizer.xe_lr,
        min_lr=config.optimizer.min_lr,
        warmup_init_lr=config.optimizer.warmup_init_lr,
    )

    with open("result.csv", "w") as f:
        f.write("epoch, train_loss, val_loss, Bleu_1, Bleu_4, METEOR, ROUGE, CIDEr\n")

    best_cider = 0.
    phase = 'ft_xe'
    for epoch in range(10):
        train_loss, val_loss = train_xe(model, dataloaders, optimizers=optimizers, epoch=epoch, rank=rank, config=config, scheduler=scheduler)
        samplers['train'].set_epoch(epoch)

        if rank == 0:
            scores = evaluate_metrics(
                model,
                optimizers,
                dataloader=dataloaders['valid_dict'],
                epoch=epoch,
                split='valid',
                config=config,
                which=phase,
                scheduler=scheduler,
            )

            torch.save({"state_dict": model.module.state_dict()}, f"model.pth")
            if scores['CIDEr'] > best_cider:
                best_cider = scores['CIDEr']
                torch.save({"state_dict": model.module.state_dict()}, f"model_best.pth")

            with open("result.csv", "a") as f:
                f.write(
                    f"{epoch}, {train_loss:0.4f}, {val_loss:0.4f}, {scores['BLEU'][0]:0.4f}, {scores['BLEU'][-1]:0.4f}, {scores['METEOR']:0.4f}, {scores['ROUGE']:0.4f}, {scores['CIDEr']:0.4f}\n"
                )
        torch.distributed.barrier()


@hydra.main(config_path="configs/caption", config_name="custom_config")
def run_main(config: DictConfig) -> None:
    mp.spawn(main, nprocs=config.exp.ngpus_per_node, args=(config,))


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6688"
    run_main()