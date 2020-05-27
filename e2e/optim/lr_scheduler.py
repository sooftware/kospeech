"""
e2e.optim.lr_scheduler provides RampUp & ExponentialDecay method to adjust learning rate.
"""


class LearningRateScheduler(object):
    """ Interface of lr scheduler """
    def __init__(self, optimizer, init_lr):
        self.optimizer = optimizer
        self.init_lr = init_lr

    def step(self):
        raise NotImplementedError

    @staticmethod
    def set_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g['lr'] = lr

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g['lr']


class RampUpLR(LearningRateScheduler):
    """ Ramp up (Warm up) learning rate scheduler """
    def __init__(self, optimizer, init_lr, high_plateau_lr, period):
        super(RampUpLR, self).__init__(optimizer, init_lr)
        self.timestep = 1
        self.power = 3
        self.period = period
        self.high_plateau_lr = high_plateau_lr

    def step(self):
        self.set_lr(self.optimizer, lr=self.high_plateau_lr * (self.timestep / self.period) ** self.power)
        self.timestep += 1


class ExponentialDecayLR(LearningRateScheduler):
    """ Exponential decay learning rate scheduler """
    def __init__(self, optimizer, init_lr, low_plateau_lr, period):
        super(ExponentialDecayLR, self).__init__(optimizer, init_lr)
        decay_rate = low_plateau_lr / init_lr
        self.decay_speed = decay_rate ** (1 / period)

    def step(self):
        lr = self.get_lr()
        self.set_lr(self.optimizer, lr * self.decay_speed)
