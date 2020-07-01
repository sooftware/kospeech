# Test code for e2e.optim.optim & e2e.optim.lr_scheduler

import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from kospeech.optim.optimizer import Optimizer
from kospeech.optim.lr_scheduler import RampUpLR, ExponentialDecayLR


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.projection = nn.Linear(10, 10)

    def forward(self):
        pass


INIT_LR = 1e-15
HIGH_PLATEAU_LR = 3e-04
LOW_PLATEAU_LR = 1e-05
RAMPUP_PERIOD = 1000
MAX_GRAD_NORM = 400
TOTAL_TIME_STEP = 7000
EXP_DECAY_START = 4500
EXP_DECAY_PERIOD = 1000

model = Model()

optimizer = optim.Adam(model.parameters(), lr=INIT_LR)
scheduler = RampUpLR(optimizer, INIT_LR, HIGH_PLATEAU_LR, RAMPUP_PERIOD)
optimizer = Optimizer(optimizer, scheduler, RAMPUP_PERIOD, MAX_GRAD_NORM)

lr_processes = list()

for timestep in range(TOTAL_TIME_STEP):
    optimizer.step(model, 0.0)
    lr_processes.append(optimizer.get_lr())

    if timestep == EXP_DECAY_START:
        scheduler = ExponentialDecayLR(optimizer.optim, optimizer.get_lr(), LOW_PLATEAU_LR, EXP_DECAY_PERIOD)
        optimizer.set_scheduler(scheduler, EXP_DECAY_PERIOD)

plt.title('Test Optimizer class')
plt.plot(lr_processes)
plt.xlabel('timestep', fontsize='large')
plt.ylabel('lr', fontsize='large')
plt.show()
