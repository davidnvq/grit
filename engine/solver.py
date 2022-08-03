# ------------------------------------------------------------------------
# GRIT: Faster and Better Image captioning Transformer
# Licensed under the Creative Commons Attribution.
# ------------------------------------------------------------------------
import weakref

from tqdm import tqdm
from .hooks import HookBase


class SolverBase:
    """The abstract for Trainer and Validator"""

    def __init__(self, model, dataloader, optimizers, device='cuda', lr_scheduler=None):
        self.model = model
        self.dataloader = dataloader
        self.optimizers = optimizers
        if isinstance(self.optimizers, list):
            self.optimizer = optimizers[0]

        self.scheduler = lr_scheduler
        self.hooks = []
        self.step_res = {}
        self.epoch_res = {}
        self.device = device
        self.step = 0
        self.epoch = 0
        self.progbar = None
        self.keys = {'epoch'}

    def register_hooks(self, hooks):
        """
        Register solver to every hook. That makes hooks can get the variables of Solver
        Args:
            hooks: list of hooks
        """
        # avoid circular references: bit.ly/2Fv4LRa
        for h in hooks:
            assert isinstance(h, HookBase)
            h.register(weakref.proxy(self))
        self.hooks.extend(hooks)
        self.hook_name2idx = {h.__class__.__name__: idx for idx, h in enumerate(self.hooks)}

    def exec(self, fn_name):
        for h in self.hooks:
            getattr(h, fn_name)()

    def on_step(self, batch):
        """Perform all the things within 1 iteration
        Args:
            batch: the batch of input data (for batch in dataloader)
        """
        self.model.train()

        # 1. zero gradients
        if isinstance(self.optimizers, list):
            for optimizer in self.optimizers:
                optimizer.zero_grad()
        else:
            self.optimizers.zero_grad()

        # 2. forward pass
        self.step_res = self.model(batch)

        # 3. log for epoch results
        for key, value in self.step_res.items():
            self.epoch_res[key].append(value)

        # 4. backward pass
        self.step_res['loss'].backward()

        # 5. update parameters
        if isinstance(self.optimizers, list):
            for optimizer in self.optimizers:
                optimizer.step()
        else:
            self.optimizers.step()

    def run_epoch(self, epoch):
        """Perform all the things within 1 epoch
        Args:
            epoch: (int) the epoch number
        """
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

        self.exec('after_epoch')

        # reset epoch results
        for key in self.epoch_res:
            self.epoch_res[key] = 0.0
