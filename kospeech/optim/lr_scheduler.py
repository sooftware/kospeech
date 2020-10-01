# -*- coding: utf-8 -*-
# Soohwan Kim, Seyoung Bae, Cheolhwang Won.
# @ArXiv : KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.
import math


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


class TriStageLRScheduler(LearningRateScheduler):
    def __init__(self, optimizer, init_lr, peak_lr, final_lr, init_lr_scale, final_lr_scale, warmup_steps, total_steps):
        assert isinstance(warmup_steps, int), "warmup_steps should be inteager type"
        assert isinstance(total_steps, int), "total_steps should be inteager type"

        super(TriStageLRScheduler, self).__init__(optimizer, init_lr)
        self.init_lr *= init_lr_scale
        self.final_lr = final_lr
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.hold_steps = int(total_steps >> 1) - warmup_steps
        self.decay_steps = int(total_steps >> 1)

        self.warmup_rate = (self.peak_lr - self.init_lr) / self.warmup_steps if self.warmup_steps != 0 else 0
        self.decay_factor = -math.log(final_lr_scale) / self.decay_steps

        self.lr = self.init_lr
        self.update_step = 0

    def _decide_stage(self):
        if self.update_step < self.warmup_steps:
            return 0, self.update_step

        offset = self.warmup_steps

        if self.update_step < offset + self.hold_steps:
            return 1, self.update_step - offset

        offset += self.hold_steps

        if self.update_step <= offset + self.decay_steps:
            # decay stage
            return 2, self.update_step - offset

        offset += self.decay_steps

        return 3, self.update_step - offset

    def step(self):
        stage, steps_in_stage = self._decide_stage()

        if stage == 0:
            self.lr = self.init_lr + self.warmup_rate * steps_in_stage
        elif stage == 1:
            self.lr = self.peak_lr
        elif stage == 2:
            self.lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
        elif stage == 3:
            self.lr = self.final_lr
        else:
            raise ValueError("Undefined stage")

        self.set_lr(self.optimizer, self.lr)
        self.update_step += 1

        return self.lr


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

