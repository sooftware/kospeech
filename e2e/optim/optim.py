import torch


class Optimizer(object):
    """
    The Optimizer class provides functionalities for learning rate scheduling and gradient norm clipping.

    Args:
        optimizer (torch.optim.Optimizer): optimizer object, the parameters to be optimized
            should be given when instantiating the object, e.g. torch.optim.Adam, torch.optim.SGD
        scheduler (e2e.optim.lr_scheduler): learning rate scheduler
        scheduler_period (int): timestep with learning rate scheduler
        max_grad_norm (int): If the norm of the gradient vector exceeds this, remormalize it to have the norm equal to
    """
    def __init__(self, optimizer, scheduler=None, scheduler_period=None, max_grad_norm=0):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_period = scheduler_period
        self.max_grad_norm = max_grad_norm
        self.count = 0

    def step(self, model, loss):
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        if self.scheduler is not None:
            self.update(loss)
            self.count += 1

            if self.scheduler_period == self.count:
                self.scheduler = None
                self.scheduler_period = 0
                self.count = 0

    def set_scheduler(self, scheduler, scheduler_period):
        self.scheduler = scheduler
        self.scheduler_period = scheduler_period
        self.count = 0

    def update(self, loss):
        if self.scheduler is None:
            pass
        elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(loss)

        else:
            self.scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g['lr']
