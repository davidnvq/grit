# ------------------------------------------------------------------------
# GRIT: Faster and Better Image captioning Transformer
# Licensed under the Creative Commons Attribution.
# ------------------------------------------------------------------------

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

def build_optimizers(model, config, mode='xe'):
    model = getattr(model, 'module', model)

    no_decay = ['bias', 'gamma', 'beta']

    model_parameters = [
        {
            'params': [
                p for n, p in model.named_parameters()
                if p.requires_grad and 'detector' not in n and any(nd in n for nd in no_decay)
            ],
            'weight_decay_rate': 0.0
        },
        {
            'params': [
                p for n, p in model.named_parameters()
                if p.requires_grad and 'detector' not in n and not any(nd in n for nd in no_decay)
            ],
            'weight_decay_rate': config.optimizer.weight_decay
        },
    ]

    backbone_parameters = [
        {
            'params': [
                p for n, p in model.named_parameters()
                if p.requires_grad and 'detector' in n and any(nd in n for nd in no_decay)
            ],
            'weight_decay_rate': 0.0
        },
        {
            'params': [
                p for n, p in model.named_parameters()
                if p.requires_grad and 'detector' in n and not any(nd in n for nd in no_decay)
            ],
            'weight_decay_rate': config.optimizer.weight_decay
        },
    ]

    optimizers = {
        'model':
            torch.optim.Adam(
                model_parameters,
                lr=getattr(config.optimizer, f'{mode}_lr', config.optimizer.sc_lr),
                betas=(config.optimizer.beta_1, config.optimizer.beta_2),
            ),
        'backbone':
            torch.optim.Adam(
                backbone_parameters,
                lr=getattr(config.optimizer, f'{mode}_backbone_lr', config.optimizer.sc_backbone_lr),
                betas=(config.optimizer.beta_1, config.optimizer.beta_2),
            ),
        'mode':
            mode
    }
    return optimizers


def gather_result(value):
    if isinstance(value, torch.Tensor):
        torch.distributed.all_reduce(value, async_op=False)  # compute the sum
        value.mul_(1.0 / torch.distributed.get_world_size())  # compute the avg
    return value


def save_checkpoint(
    model,
    optimizers,
    epoch,
    scores,
    best_ciders,
    config=None,
    filename='checkpoint_last.pth',
    scheduler=None,
):
    torch.save(
        {
            "state_dict": model.module.state_dict(),
            "optim_model": optimizers['model'].state_dict(),
            "optim_backbone": optimizers['backbone'].state_dict(),
            "scores": scores,
            "best_ciders": best_ciders,
            "epoch": epoch,
            "exp_name": "" if config is None else config.exp.name,
            "scheduler": [] if scheduler is None else scheduler.state_dict(),
        }, filename)


def log_epoch(config, writer, epoch, train_res, split, scores, which='ft_xe'):
    """For better logging and viewing the log file.
    Run the command in terminal: 
    >>> column -t -s, result.csv
    """
    head = 'exp, backbone, imsize, resize, raug, epoch, split, cider, B1, B4, R, M, B2, B3, t-loss, t-reward, b-reward, which, v-loss'

    if epoch == 0 and not os.path.exists('result.csv'):
        with open('result.csv', 'w') as f:
            f.write(head + '\n')

    with open('result.csv', 'a') as f:
        text = f'{config.exp.name.split("/")[-1]}, '
        backbone = 'B-'
        backbone += 'VG' if os.path.exists(config.model.detector.checkpoint) else 'IM'
        text += f'{backbone}, '
        text += f'{config.dataset.transform_cfg.size[0]}_{config.dataset.transform_cfg.size[1]}, '
        text += f'{config.dataset.transform_cfg.resize_name}, {config.dataset.transform_cfg.randaug}, '
        text += f'{epoch}, {split:<5}, '
        text += f'{scores["CIDEr"]*100:3.2f}, {scores["BLEU"][0]*100:3.2f}, '
        text += f'{scores["BLEU"][3]*100:3.2f}, {scores["ROUGE"]*100:3.2f}, '
        text += f'{scores["METEOR"]*100:3.2f}, {scores["BLEU"][1]*100:3.2f}, {scores["BLEU"][2]*100:3.2f}, '
        text += f'{train_res["loss"]:2.2f}, {train_res["reward"]:2.2f}, {train_res["reward_baseline"]:2.2f}, '
        text += f'{which}, {train_res["val_loss"]:1.2f}'
        f.write(text + '\n')
        print(text)

    writer.add_scalar(f'{split}_cider', scores['CIDEr'], epoch)
    writer.add_scalar(f'{split}_bleu1', scores['BLEU'][0], epoch)
    writer.add_scalar(f'{split}_bleu4', scores['BLEU'][3], epoch)
    writer.add_scalar(f'{split}_meteor', scores['METEOR'], epoch)
    writer.add_scalar(f'{split}_rouge', scores['ROUGE'], epoch)

    writer.add_scalar(f'train_loss', train_res['loss'], epoch)
    writer.add_scalar(f'train_reward', train_res['reward'], epoch)
    writer.add_scalar(f'train_reward_baseline', train_res['reward_baseline'], epoch)


def evaluate_metrics(
    model,
    optimizers,
    dataloader,
    text_field,
    epoch=0,
    split='test',
    config=None,
    train_res=None,
    writer=None,
    best_cider=None,
    which='ft_xe',
    scheduler=None,
    log_and_save=True,
):
    model.eval()
    gen, gts = {}, {}

    counter = 0
    times = []
    with tqdm(desc=f'Epoch {epoch} - evaluation on {split}', unit='it', total=len(dataloader)) as pbar:

        results = []
        for it, batch in enumerate(iter(dataloader)):
            counter += 1
            start_it = time.time()
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
            end_it = time.time()
            times.append(end_it - start_it)

            if 'samples' in batch and not isinstance(batch['samples'], dict):
                bs = batch['samples'].tensors.shape[0]
            else:
                bs = batch['samples']['reg_feat'].shape[0]
            if it % 100 == 0:
                print(
                    f"Number of iterations: {counter}, batch_size={bs}, Total time per 1 batch: {sum(times)/counter:0.5f}s"
                )

            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(batch['captions'], caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen[f'{it}_{i}'] = [gen_i]
                gts[f'{it}_{i}'] = gts_i
                res = {'image_id': batch['image_id'][i], 'caption': gen_i}
                results.append(res)
            pbar.update()

    avg_time = sum(times) / counter
    print(f"Epoch: {epoch} iters: {counter}\nTotal time per 1 batch: {avg_time:0.5f}s")
    gts = metrics.PTBTokenizer.tokenize(gts)
    gen = metrics.PTBTokenizer.tokenize(gen)
    scores, _ = metrics.compute_scores(gts, gen)
    print(f'Epoch {epoch}: {split} scores: ' + str(scores) + '\n')

    if log_and_save:
        with open('result.txt', 'a') as f:
            f.write(f'Epoch {epoch}: {split} scores: ' + str(scores) + '\n')
        log_epoch(config, writer, epoch, train_res, split=split, scores=scores, which=which)

        if scores['CIDEr'] >= best_cider:
            best_ciders = (scores['CIDEr'], 0) if split == 'valid' else (0, scores['CIDEr'])
            save_checkpoint(
                model,
                optimizers=optimizers,
                epoch=epoch,
                scores=scores,
                best_ciders=best_ciders,
                config=config,
                filename=f'checkpoint_best_{split}.pth',
                scheduler=scheduler,
            )
            best_cider = scores['CIDEr']
        return best_cider
    else:
        return scores


def inference_coco_test(
    model,
    dataloader,
    text_field,
    epoch=0,
    split='test',
    config=None,
):
    model.eval()
    gen, gts = {}, {}

    counter = 0
    times = []
    with tqdm(desc=f'Epoch {epoch} - evaluation on {split}', unit='it', total=len(dataloader)) as pbar:

        results = []
        for it, batch in enumerate(iter(dataloader)):
            counter += 1
            start_it = time.time()
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
            end_it = time.time()
            times.append(end_it - start_it)

            if 'samples' in batch:
                bs = batch['samples'].tensors.shape[0]
            elif 'vis_feat' in batch:
                bs = batch['vis_feat'].shape[0]
            if it % 100 == 0:
                print(
                    f"Number of iterations: {counter}, batch_size={bs}, Total time per 1 batch: {sum(times)/counter:0.5f}s"
                )

            caps_gen = text_field.decode(out, join_words=False)
            for i, gen_i in enumerate(caps_gen):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                res = {'image_id': batch['image_id'][i], 'caption': gen_i}
                results.append(res)
            pbar.update()

    with open(f'result_{split}.json', 'w') as f:
        json.dump(results, f)


def evaluate_loss(model, dataloader, loss_fn, text_field, epoch, writer):
    model.eval()

    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % epoch, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, batch in enumerate(dataloader):
                out = model(batch['samples'], batch['captions'])

                captions_gt = batch['captions'][:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))

                loss = gather_result(loss)
                running_loss += loss.item()

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    if dist.get_rank() == 0:
        writer.add_scalar('val_loss', val_loss, epoch)
    return val_loss


def train_xe(
    model,
    dataloaders,
    optimizers,
    text_field,
    epoch,
    rank=0,
    config=None,
    scheduler=None,
    writer=None,
):
    model.train()
    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
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
            loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
            loss.backward()

            optimizers['model'].step()
            optimizers['backbone'].step()

            loss = gather_result(loss)
            running_loss += loss.item()

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()

            if scheduler is not None:
                lr = scheduler.step()
                assert optimizers['model'].param_groups[0]['lr'] == lr, "LR scheduler doesn't work properly."

            if rank == 0:
                writer.add_scalar(
                    'backbone_lr',
                    optimizers['backbone'].param_groups[0]['lr'],
                    epoch * len(dataloaders['train']) + it,
                )
                writer.add_scalar(
                    'model_lr',
                    optimizers['model'].param_groups[0]['lr'],
                    epoch * len(dataloaders['train']) + it,
                )
                lr = optimizers['model'].param_groups[0]['lr']

    val_loss = evaluate_loss(model, dataloaders['valid'], loss_fn, text_field, epoch, writer)

    if rank == 0:
        save_checkpoint(
            model=model,
            optimizers=optimizers,
            epoch=epoch,
            scores=[],
            best_ciders=(0, 0),
            config=config,
            filename='checkpoint_last.pth',
            scheduler=scheduler,
        )
    torch.distributed.barrier()

    return {
        'loss': running_loss / len(dataloaders['train']),
        'reward': 0,
        'reward_baseline': 0,
        'val_loss': val_loss,
    }


def train_sc(model,
             dataloaders,
             optimizers,
             cider,
             text_field,
             tokenizer_pool,
             device,
             epoch,
             config,
             rank=0,
             writer=None):
    # Training with self-critical
    running_reward = .0
    running_reward_baseline = .0
    running_loss = .0
    seq_len = config.model.beam_len
    beam_size = config.model.beam_size
    model.train()

    with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(dataloaders['train_dict'])) as pbar:
        for it, batch in enumerate(dataloaders['train_dict']):
            if 'samples' in batch:
                if isinstance(batch['samples'], NestedTensor):
                    b_s = batch['samples'].tensors.shape[0]
                elif 'gri_feat' in batch['samples']:
                    b_s = batch['samples']['gri_feat'].shape[0]
                elif 'reg_feat' in batch['samples']:
                    b_s = batch['samples']['reg_feat'].shape[0]
            elif 'vis_feat' in batch:
                b_s = batch['vis_feat'].shape[0]
                
            optimizers['model'].zero_grad()
            optimizers['backbone'].zero_grad()
            outs, log_probs = model(
                batch['samples'],
                seq=None,
                use_beam_search=True,
                max_len=config.model.beam_len,
                eos_idx=config.model.eos_idx,
                beam_size=config.model.beam_size,
                out_size=beam_size,
                return_probs=False,
            )
            # Rewards
            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c] * beam_size for c in batch['captions'])))  # [c,]

            caps_gen, caps_gt = tokenizer_pool.map(metrics.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(b_s, beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()
            loss.backward()
            torch.distributed.barrier()

            optimizers['model'].step()
            optimizers['backbone'].step()

            loss = gather_result(loss)
            running_loss += loss.item()

            reward = gather_result(reward.mean())
            running_reward += reward.item()

            reward_baseline = gather_result(reward_baseline.mean())
            running_reward_baseline += reward_baseline.item()

            pbar.set_postfix(loss=running_loss / (it + 1),
                             reward=running_reward / (it + 1),
                             reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()
            if rank == 0:
                writer.add_scalar(
                    'backbone_lr',
                    optimizers['backbone'].param_groups[0]['lr'],
                    epoch * len(dataloaders['train_dict']) + it,
                )
                writer.add_scalar(
                    'model_lr',
                    optimizers['model'].param_groups[0]['lr'],
                    epoch * len(dataloaders['train_dict']) + it,
                )

    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    val_loss = evaluate_loss(model, dataloaders['valid'], loss_fn, text_field, epoch, writer)
    loss = running_loss / len(dataloaders['train_dict'])
    reward = running_reward / len(dataloaders['train_dict'])
    reward_baseline = running_reward_baseline / len(dataloaders['train_dict'])
    if rank == 0:
        save_checkpoint(
            model=model,
            optimizers=optimizers,
            epoch=epoch,
            scores=[],
            best_ciders=(0, 0),
            config=config,
            filename='checkpoint_last.pth',
            scheduler=None,
        )

    torch.distributed.barrier()

    return {'loss': loss, 'reward': reward, 'reward_baseline': reward_baseline, 'val_loss': val_loss}
