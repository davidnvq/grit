# ------------------------------------------------------------------------
# GRIT: Faster and Better Image captioning Transformer
# Licensed under the Creative Commons Attribution.
# ------------------------------------------------------------------------
import os
import pickle
import torch
from tqdm import tqdm
from engine.solver import SolverBase
from datasets.detection.metrics.coco_eval import CocoEvaluator
from datasets.detection.metrics.coco_utils import convert_to_coco_api
import torch.distributed as dist


def add_epoch_lr(self):
    if isinstance(self.optimizers, list):
        for i, optimizer in enumerate(self.optimizers):
            self.epoch_res[f'epoch_lr_{i}'] = optimizer.param_groups[0]['lr']
            self.keys.add(f'epoch_lr_{i}')
    else:
        self.epoch_res['epoch_lr'] = self.optimizers.param_groups[0]['lr']
        self.keys.add('epoch_lr')

    self.epoch_res['epoch'] = self.epoch
    self.keys.add('epoch')


class Trainer(SolverBase):

    def __init__(self,
                 model,
                 dataloader,
                 optimizers,
                 criterion,
                 device='cuda',
                 lr_scheduler=None,
                 max_norm=0.0,
                 eval_every_iters=-1):
        super(Trainer, self).__init__(model, dataloader, optimizers, lr_scheduler=lr_scheduler, device=device)
        self.criterion = criterion
        self.max_norm = max_norm
        self.validers = None
        self.eval_every_iters = eval_every_iters
        print(f"Evaluate every iterations = {eval_every_iters}")

    def set_validers(self, validers):
        self.validers = validers

    def update_loss_dict(self, loss_dict, res_out):
        for key in res_out:
            if key in loss_dict:
                loss_dict[key] += res_out[key]
            else:
                loss_dict[key] = res_out[key]

    def on_step(self, batch):
        self.model.train()
        self.criterion.train()
        samples, targets = batch
        # added
        samples = samples.to(self.device)
        # targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        targets = [{k: v.to(self.device)
                    for k, v in t.items()
                    if isinstance(v, torch.Tensor)} if t is not None else None
                   for t in targets]
        # forward pass
        outputs = self.model(samples)
        loss_dict = {
            'loss_ce': torch.sum(outputs['pred_logits']) * 0.0,
            'loss_bbox': torch.sum(outputs['pred_boxes']) * 0.0,
            'loss_giou': torch.sum(outputs['pred_boxes']) * 0.0,
        }

        if 'attr_logits' in outputs:
            loss_dict['attr_logits'] = torch.sum(outputs['attr_logits']) * 0.0

        # to compute object detection loss
        # outputs['pred_logits'] = outputs['pred_logits'][batch['has_bbox']]
        # outputs['pred_boxes'] = outputs['pred_boxes'][batch['has_bbox']]
        # outputs['attr_logits'] = outputs['attr_logits'][batch['has_bbox']]

        # loss_dict = self.criterion(outputs, targets)
        self.update_loss_dict(loss_dict, self.criterion(outputs, targets))

        weight_dict = self.criterion.weight_dict

        self.step_res['losses'] = 0.0
        self.keys.add('losses')
        for k in loss_dict.keys():
            if k in weight_dict:
                self.step_res['losses'] += loss_dict[k] * weight_dict[k]
                self.step_res[k] = loss_dict[k] * weight_dict[k]
                self.keys.add(k)

        if isinstance(self.optimizers, list):
            for i, optimizer in enumerate(self.optimizers):
                self.step_res[f'lr_{i}'] = self.optimizers[i].param_groups[0]['lr']
                self.keys.add(f'lr_{i}')
                optimizer.zero_grad()
        else:
            self.step_res[f'lr'] = self.optimizers.param_groups[0]['lr']
            self.keys.add(f'lr')
            self.optimizers.zero_grad()

        self.step_res['losses'].backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)

        # udpate params
        if isinstance(self.optimizers, list):
            for i, optimizer in enumerate(self.optimizers):
                optimizer.step()
        else:
            self.optimizers.step()

        # logging
        for key, value in self.step_res.items():
            if key not in self.epoch_res:
                self.epoch_res[key] = 0.0
            if isinstance(value, torch.Tensor):
                # pass  # Todo: Bugs with two line below
                torch.distributed.all_reduce(value, async_op=False)  # compute the sum
                value.mul_(1.0 / torch.distributed.get_world_size())  # compute the avg
            self.epoch_res[key] += value.detach() if isinstance(value, torch.Tensor) else value

    def run_epoch(self, epoch):
        self.epoch = epoch
        self.exec('before_epoch')

        self.progbar = tqdm(self.dataloader)
        for step, batch in enumerate(self.progbar):
            self.step = step
            self.exec('before_step')
            self.on_step(batch)
            self.exec('after_step')
            if self.validers is not None and self.eval_every_iters > 0:
                if step % self.eval_every_iters == 0 and step > 0:
                    for data_name, valider in self.validers.items():
                        print(f"Evaluate {data_name:<10} at epoch={epoch}, step={step:04d}...")
                        valider.run_epoch(epoch)
                    if dist.get_rank() == 0:
                        checkpoint = {
                            'epoch': self.epoch,
                            'model': self.model.module.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                        }
                        torch.save(checkpoint, f"checkpoint_epoch{epoch}_step{step:04d}.pth")

        for key in self.epoch_res:
            self.epoch_res[key] = float(self.epoch_res[key]) / float(len(self.dataloader))

        add_epoch_lr(self)
        self.exec('after_epoch')

        # reset epoch results
        for key in self.epoch_res:
            self.epoch_res[key] = 0.0


class Valider(SolverBase):

    def __init__(self,
                 model,
                 dataloader,
                 optimizers,
                 criterion,
                 postprocessors,
                 device='cuda',
                 lr_scheduler=None,
                 rank=0,
                 data_name='coco_val'):
        super(Valider, self).__init__(model, dataloader, optimizers, lr_scheduler=lr_scheduler, device=device)
        self.criterion = criterion
        self.data_name = data_name
        self.postprocessors = postprocessors
        if getattr(dataloader.dataset, "coco", None) is not None:
            self.coco = dataloader.dataset.coco
        else:
            if not os.path.exists(dataloader.dataset.coco_file):
                self.coco = convert_to_coco_api(dataloader.dataset)
                with open(dataloader.dataset.coco_file, 'wb') as f:
                    pickle.dump(self.coco, f)
            else:
                with open(dataloader.dataset.coco_file, 'rb') as f:
                    self.coco = pickle.load(f)

        self.iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
        self.coco_evaluator = CocoEvaluator(self.coco, self.iou_types)
        self.rank = rank

    def on_step(self, batch):
        with torch.no_grad():
            self.model.eval()
            self.criterion.eval()
            samples, targets = batch
            samples = samples.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            outputs = self.model(samples)
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict

            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

            results = self.postprocessors['bbox'](outputs, orig_target_sizes)

            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            if self.coco_evaluator is not None:
                self.coco_evaluator.update(res)

            # logging
            self.step_res['losses'] = 0.0
            self.keys.add('losses')
            for k in loss_dict.keys():
                if k in weight_dict:
                    self.step_res['losses'] += loss_dict[k] * weight_dict[k]
                    self.step_res[k] = loss_dict[k] * weight_dict[k]
                    self.keys.add(k)

            for key, value in self.step_res.items():
                if key not in self.epoch_res:
                    self.epoch_res[key] = 0.0
                if isinstance(value, torch.Tensor):
                    # pass  # Todo: Bugs with two line below
                    torch.distributed.all_reduce(value, async_op=False)  # compute the sum
                    value.mul_(1.0 / torch.distributed.get_world_size())  # compute the avg
                self.epoch_res[key] += value.detach() if isinstance(value, torch.Tensor) else value

    def run_epoch(self, epoch):
        n_threads = torch.get_num_threads()
        torch.set_num_threads(1)

        self.epoch = epoch
        self.exec('before_epoch')

        # iterate over the dataset
        self.progbar = tqdm(self.dataloader)

        for step, batch in enumerate(self.progbar):
            self.step = step
            self.exec('before_step')
            self.on_step(batch)
            self.exec('after_step')

        for key in self.epoch_res:
            self.epoch_res[key] = float(self.epoch_res[key]) / float(len(self.dataloader))

        if self.coco_evaluator is not None:
            self.coco_evaluator.synchronize_between_processes()
            self.coco_evaluator.accumulate()
            self.coco_evaluator.summarize()

            if self.coco_evaluator.coco_eval is not None:
                for iou_type in self.iou_types:
                    for idx, k in enumerate(['map', 'map50', 'map75', 'map_small', 'map_medium', 'map_large']):
                        self.epoch_res[iou_type + f'_{k}'] = self.coco_evaluator.coco_eval[iou_type].stats[idx]
                        self.keys.add(iou_type + f'_{k}')

                    if self.rank == 0:
                        message = f"\n" + self.coco_evaluator.coco_eval[iou_type].result_text
                        print(message)
                        self.hooks[self.hook_name2idx["TextLoggingHook"]].write(message)

        add_epoch_lr(self)

        self.exec('after_epoch')
        torch.set_num_threads(n_threads)

        # reset epoch results
        self.coco_evaluator = CocoEvaluator(self.coco, self.iou_types)
        for key in self.epoch_res:
            self.epoch_res[key] = 0.0