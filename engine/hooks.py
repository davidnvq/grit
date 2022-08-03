# ------------------------------------------------------------------------
# GRIT: Faster and Better Image captioning Transformer
# Licensed under the Creative Commons Attribution.
# ------------------------------------------------------------------------
import os
import torch
import logging
import os.path as osp
from torch.utils.tensorboard import SummaryWriter


class HookBase:
    """The abstract Hook for Trainer and Valider"""

    def __init__(self):
        self.solver = None

    def register(self, solver):
        self.solver = solver

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_step(self):
        pass

    def after_step(self):
        pass


class CheckpointHook(HookBase):

    def __init__(
            self,
            save_every_epochs=-1,
            save_every_iters=-1,
            save_topk=0,
            metric=None,  # (None, 'higher'), 
            which_epochs=[],
            save_dir="",
            **kwargs):
        super().__init__()
        self.save_every_iters = save_every_iters
        self.save_every_epochs = save_every_epochs
        self.which_epochs = which_epochs
        self.save_dir = save_dir
        self.save_topk = save_topk
        self.pathmetric_list = []
        self.metric = metric
        self.kwargs = kwargs
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Create {self.save_dir} for checkpointing")

    def get_checkpoint(self):
        # take care of DataParallel
        model = self.solver.model.module if hasattr(self.solver.model, "module") else self.solver.model
        lr_scheduler_dict = None if self.solver.scheduler is None else self.solver.scheduler.state_dict()

        checkpoint = {
            'epoch': self.solver.epoch,
            'model': model.state_dict(),
            'optimizer': self.solver.optimizer.state_dict(),
            'scheduler': lr_scheduler_dict,
            'epoch_res': self.solver.epoch_res
        }
        checkpoint.update(self.kwargs)
        return checkpoint

    def after_step(self):
        if self.save_every_iters != -1 and self.solver.step % self.save_every_iters == 0:
            checkpoint = self.get_checkpoint()
            save_file = osp.join(self.save_dir,
                                 'checkpoint_{:02d}_iter-{:05d}.pth'.format(self.solver.epoch, self.solver.step))
            torch.save(checkpoint, save_file)

    def after_epoch(self):
        epoch = self.solver.epoch
        save_file = osp.join(self.save_dir, 'checkpoint_{:02d}'.format(epoch))
        last_file = osp.join(self.save_dir, 'checkpoint_last.pth')

        if self.metric is not None:
            save_file += '_{}_{:.4f}'.format(self.metric[0], self.solver.epoch_res[self.metric[0]])
        save_file += '.pth'

        checkpoint = self.get_checkpoint()
        torch.save(checkpoint, save_file)
        torch.save(checkpoint, last_file)
        if self.save_topk > 0:
            print(f"Saving the topk to {save_file}")
            self.pathmetric_list.append((save_file, self.solver.epoch_res[self.metric[0]]))
            reverse = True if self.metric[1] == 'higher' else False
            self.pathmetric_list = sorted(self.pathmetric_list, key=lambda x: x[1], reverse=reverse)
            for item in self.pathmetric_list[self.save_topk:]:
                if item[0] != save_file:  # keep the last ckpt
                    os.remove(item[0])  # remove the lower value
            self.pathmetric_list = self.pathmetric_list[:self.save_topk]

        if self.save_every_epochs != -1 and self.solver.epoch % self.save_every_epochs == 0:
            logging.debug("Saving file to {}".format(save_file))
            torch.save(checkpoint, save_file)  #,  _use_new_zipfile_serialization=False)

        if epoch in self.which_epochs or epoch + 1 in self.which_epochs:
            torch.save(checkpoint, save_file)


class TextLoggingHook(HookBase):

    def __init__(self, name, save_dir='./'):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Create {save_dir} for text logging")
        self.name = name
        self.log_file = os.path.join(save_dir, "log.txt")

    def after_epoch(self):
        log_dict = {k: self.solver.epoch_res[k] for k in self.solver.keys if k in self.solver.epoch_res}
        self.write(str(log_dict))

    def write(self, message):
        with open(self.log_file, "a") as f:
            line = f"Epoch {self.solver.epoch:03d}: {self.name} " + message + "\n"
            f.write(line)


class TensorboardHook(HookBase):

    def __init__(self, name, save_dir="./", excluded_keys=[], log_every_step=100):
        # name: (str) name of data split ("train", "validate", etc)
        super().__init__()
        self.name = name
        os.makedirs(osp.join(save_dir, "tensorboard"), exist_ok=True)
        print(f"Create {osp.join(save_dir, 'tensorboard')} for tensorboard")
        self.log_file = os.path.join(save_dir, "log.txt")
        self.writer = SummaryWriter(osp.join(save_dir, "tensorboard"))
        self.excluded_keys = excluded_keys
        self.log_every_step = log_every_step

    def register(self, solver):
        self.solver = solver
        self.steps_per_epoch = len(solver.dataloader)

    def after_step(self):
        if self.solver.step % self.log_every_step == 0:
            keys = list(set(self.solver.keys) - set(self.excluded_keys))
            log_dict = {k: self.solver.step_res[k] for k in keys if k in self.solver.step_res}
            self.writer.add_scalars(main_tag=self.name + '/batch',
                                    tag_scalar_dict=log_dict,
                                    global_step=self.solver.step + self.solver.epoch * self.steps_per_epoch)

    def after_epoch(self):
        keys = list(set(self.solver.keys) - set(self.excluded_keys))
        log_dict = {k: self.solver.epoch_res[k] for k in keys if k in self.solver.epoch_res}
        print(log_dict)
        self.writer.add_scalars(main_tag=self.name + '/epoch', tag_scalar_dict=log_dict, global_step=self.solver.epoch)


class WarmUpLRSchedulerHook(HookBase):

    def __init__(self, warmup_factor=1. / 1000, warmup_iters=None):
        super().__init__()
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_scheduler = None

    def register(self, solver):
        self.solver = solver
        if self.warmup_iters is None:
            self.warmup_iters = len(solver.dataloader) - 1

        def f(x):
            if x >= self.warmup_iters:
                return 1
            alpha = float(x) / self.warmup_iters
            return self.warmup_factor * (1 - alpha) + alpha

        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(solver.optimizer, f)

    def after_step(self):
        cur_iters = self.solver.step + self.solver.epoch * len(self.solver.dataloader)
        if cur_iters < self.warmup_iters:
            self.warmup_scheduler.step()


class LRSchedulerHook(HookBase):

    def after_epoch(self):
        if self.solver.scheduler is not None:
            self.solver.scheduler.step()


class ProgressHook(HookBase):

    def __init__(self, name="train", excluded_keys=[]):
        super().__init__()
        self.name = name
        self.excluded_keys = excluded_keys

    def get_text(self, res):
        text = f'{self.name}-epoch {self.solver.epoch:>2}| '
        for k, v in res.items():
            if k in self.solver.keys and k not in self.excluded_keys:
                if isinstance(v, torch.Tensor):
                    v = v.item()
                fmt = '.2f' if type(v) == float else ''
                fmt = '.1e' if k == 'lr' else fmt
                text += f"{k} {v:{fmt}}| "
        return text

    def after_step(self):
        text = self.get_text(res=self.solver.step_res)
        logging.debug(text)
        self.solver.progbar.set_description(text)

    def after_epoch(self):
        text = self.get_text(res=self.solver.epoch_res)
        logging.info(text)
        self.solver.progbar.set_description(text)


def get_default_train_hooks(args, rank=0):
    train_hooks = [
        # WarmUpLRSchedulerHook(),
        ProgressHook(name='train'),
        LRSchedulerHook()
    ]
    if rank == 0:
        train_hooks += [
            CheckpointHook(save_period=1, save_dir=args.save_dir),
            TensorboardHook(name='train', save_dir=args.save_dir),
        ]
    return train_hooks


def get_default_valid_hooks(args, rank=0, name='valid'):
    valid_hooks = [ProgressHook(name=name)]
    if rank == 0:
        valid_hooks += [TensorboardHook(name=name, save_dir=args.save_dir)]
    return valid_hooks
