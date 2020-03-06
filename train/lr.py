"""
Module of learning rate

Reference
----------
    「SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition」Google Brain Team. 2019.
     https://github.com/DemisEom/SpecAugment/blob/master/SpecAugment/spec_augment_pytorch.py
"""

def ramp_up(optimizer, time_step, hparams):
    """ rampup learning rate """
    power = 3
    lr = hparams.high_plateau_lr * (time_step / 1000) ** power
    set_lr(optimizer, lr)

def exp_decay(optimizer, total_time_step, hparams):
    """ exponential decay learning rate """
    decay_rate = hparams.low_plateau_lr / hparams.high_plateau_lr
    decay_speed = decay_rate ** (1/total_time_step)
    lr = get_lr(optimizer)
    set_lr(optimizer, lr * decay_speed)

def set_lr(optimizer, lr):
    """ set learning rate """
    for g in optimizer.param_groups:
        g['lr'] = lr

def get_lr(optimizer):
    """ get learning rate """
    for g in optimizer.param_groups:
        return g['lr']