# -*- coding: utf-8 -*-
# Soohwan Kim, Seyoung Bae, Cheolhwang Won.
# @ArXiv : KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np


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


class ThreeStateLRScheduler(LearningRateScheduler):
    def __init__(self, optimizer, init_lr, high_plateau_lr, low_plateau_lr, warmup_steps, total_steps):
        assert isinstance(warmup_steps, int), "warmup_steps should be inteager type"
        assert isinstance(total_steps, int), "total_steps should be inteager type"

        super(ThreeStateLRScheduler, self).__init__(optimizer, init_lr)
        self.low_plateau_lr = low_plateau_lr
        self.high_plateau_lr = high_plateau_lr
        self.warmup_steps = warmup_steps
        self.high_plateau_steps = int(warmup_steps + (total_steps * 0.4))
        self.total_steps = total_steps
        self.decay_steps = self.high_plateau_steps + self.warmup_steps

        self.exp_decay = -np.log10(0.01) / (self.total_steps - self.decay_steps)

        self.steps = 1

    def step(self):
        if self.steps < self.warmup_steps:
            self.set_lr(self.optimizer, lr=self.high_plateau_lr * (self.steps / self.warmup_steps) ** 3)
        elif self.steps > self.high_plateau_steps:
            update_lr = self.high_plateau_lr * np.power(10, -self.exp_decay * (self.steps - self.decay_steps))
            self.set_lr(self.optimizer, lr=update_lr)

        self.steps += 1


class WarmUpLRScheduler(LearningRateScheduler):
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
        super(WarmUpLRScheduler, self).__init__(optimizer, init_lr)
        self.steps = 1
        self.period = period
        self.high_plateau_lr = high_plateau_lr

    def step(self):
        self.set_lr(self.optimizer, lr=self.high_plateau_lr * (self.steps / self.period) ** self.POWER)
        self.steps += 1


class ExponentialDecayLRScheduler(LearningRateScheduler):
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
        super(ExponentialDecayLRScheduler, self).__init__(optimizer, init_lr)
        decay_rate = low_plateau_lr / init_lr
        self.decay_speed = decay_rate ** (1 / period)

    def step(self):
        lr = self.get_lr()
        self.set_lr(self.optimizer, lr * self.decay_speed)

