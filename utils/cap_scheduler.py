import math


class CosineLRScheduler(object):

    def __init__(self,
                 optimizer,
                 num_epochs,
                 num_its_per_epoch,
                 init_lr=5e-4,
                 min_lr=1e-4,
                 warmup_init_lr=1e-5,
                 warmup_factor=0.1,
                 warmup_epochs=1,
                 **kwargs):
        self.optimizer = optimizer

        self.init_lr = init_lr
        self.warmup_init_lr = warmup_init_lr
        self.min_lr = min_lr
        self.num_epochs = num_epochs
        self.num_its_per_epoch = num_its_per_epoch

        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.global_steps = 0

    def step(self):
        self.global_steps += 1
        current_epoch = self.global_steps // self.num_its_per_epoch
        if current_epoch < 1:
            lr = self.warmup_step()
            self.update(lr)
        else:
            lr = self.cosine_step()
            lr = max(self.min_lr, lr)
            self.update(lr)
        return lr

    def update(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def warmup_step(self):
        """
        current_iteration: 1 iter = 1 batch step
        (should be understood as the global_steps accumulated from the beginning.)
        return the factor.
        """
        current_epoch = float(self.global_steps) / self.num_its_per_epoch
        alpha = current_epoch / self.warmup_epochs
        return (self.init_lr - self.warmup_init_lr) * (self.warmup_factor * (1.0 - alpha) + alpha) + self.warmup_init_lr

    def cosine_step(self):
        # current_epoch = cur_iter // self.total_iters_per_epoch
        # return self.init_lr * (1 + math.cos(math.pi * current_epoch / self.num_epochs)) / 2
        total_iters = self.num_epochs * self.num_its_per_epoch
        return (self.init_lr - self.min_lr) * (1 +
                                               math.cos(math.pi * self.global_steps / total_iters)) / 2 + self.min_lr

    def state_dict(self):
        return {
            'init_lr': self.init_lr,
            'warmup_init_lr': self.warmup_init_lr,
            'min_lr': self.min_lr,
            'num_epochs': self.num_epochs,
            'num_its_per_epoch': self.num_its_per_epoch,
            'warmup_factor': self.warmup_factor,
            'warmup_epochs': self.warmup_epochs,
            'global_steps': self.global_steps,
        }

    def load_state_dict(self, state_dict):
        self.init_lr = state_dict['init_lr']
        self.min_lr = state_dict['min_lr']
        self.warmup_init_lr = state_dict['warmup_init_lr']
        self.num_epochs = state_dict['num_epochs']
        self.num_its_per_epoch = state_dict['num_its_per_epoch']
        self.warmup_factor = state_dict['warmup_factor']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.global_steps = state_dict['global_steps']


def test_lr_scheduler():
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    num_epochs = 15
    num_its_per_epoch = 1000
    warmup_init_lr = 1e-5
    init_lr = 5e-4
    min_lr = 1e-4
    optimizer = torch.optim.Adam(torch.nn.Linear(2, 3).parameters(), lr=init_lr)
    scheduler = CosineLRScheduler(optimizer,
                                  num_epochs,
                                  num_its_per_epoch=num_its_per_epoch,
                                  init_lr=init_lr,
                                  min_lr=min_lr,
                                  warmup_init_lr=warmup_init_lr)

    lr = []
    global_steps = 0
    for epoch in range(num_epochs):
        for i in range(num_its_per_epoch):
            lr.append(scheduler.step())
            global_steps += 1

    plt.plot(np.arange(global_steps), lr)
    plt.ylim(0, init_lr)
    plt.show()
    # test_lr_scheduler()