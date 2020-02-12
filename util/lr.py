"""
Copyright 2020- Kai.Lib
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

def set_lr(optimizer, lr):
    """ set learning rate """
    for g in optimizer.param_groups:
        g['lr'] = lr

def get_lr(optimizer):
    """ get learning rate """
    for g in optimizer.param_groups:
        return g['lr']

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