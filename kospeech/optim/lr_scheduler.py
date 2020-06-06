# Provides Ramp-Up & Exp-decay learning rate scheduling


class LearningRateScheduler(object):
    """
    Provides inteface of learning rate scheduler.

    Note:
        Do not use this class directly, use one of the sub classes.
    """
    def __init__(self, optimizer, init_lr):
        self.optimizer = optimizer
        self.init_lr = init_lr

    def step(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def set_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g['lr'] = lr

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g['lr']


class RampUpLR(LearningRateScheduler):
    """
    Ramp up learning rate for the `period` from `init_lr` to `high_plateau_lr`.

    Args:
        optimizer (torch.optim.Optimizer): optimizer object, the parameters to be optimized
            should be given when instantiating the object, e.g. torch.optim.Adam, torch.optim
        init_lr (float): initial learning rate
        high_plateau_lr (float): target learning rate
        period (int): timestep for which the scheduler is applied

    ATTRIBUTES:
        POWER (int): power of ramp up. three means exponential.
    """
    POWER = 3

    def __init__(self, optimizer, init_lr, high_plateau_lr, period):
        super(RampUpLR, self).__init__(optimizer, init_lr)
        self.timestep = 1
        self.period = period
        self.high_plateau_lr = high_plateau_lr

    def step(self):
        self.set_lr(self.optimizer, lr=self.high_plateau_lr * (self.timestep / self.period) ** self.POWER)
        self.timestep += 1


class ExponentialDecayLR(LearningRateScheduler):
    """
    Exponential decay learning rate for the `period` from `init_lr` to `low_plateau_lr`.

    Args:
        optimizer (torch.optim.Optimizer): optimizer object, the parameters to be optimized
            should be given when instantiating the object, e.g. torch.optim.Adam, torch.optim
        init_lr (float): initial learning rate
        low_plateau_lr (float): target learning rate
        period (int): timestep for which the scheduler is applied
    """
    def __init__(self, optimizer, init_lr, low_plateau_lr, period):
        super(ExponentialDecayLR, self).__init__(optimizer, init_lr)
        decay_rate = low_plateau_lr / init_lr
        self.decay_speed = decay_rate ** (1 / period)

    def step(self):
        lr = self.get_lr()
        self.set_lr(self.optimizer, lr * self.decay_speed)
