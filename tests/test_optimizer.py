# Test code for e2e.optim.optim & e2e.optim.lr_scheduler

import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from e2e.optim.optim import Optimizer
from e2e.optim.lr_scheduler import RampUpLR, ExponentialDecayLR


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.projection = nn.Linear(10, 10)

    def forward(self):
        return 0


model = Model()

optimizer = optim.Adam(model.parameters(), lr=1e-15)
scheduler = RampUpLR(optimizer, 1e-15, 3e-04, 1000)
optimizer = Optimizer(optimizer, scheduler, 1000, 400)

lr_processes = list()

for timestep in range(7000):
    optimizer.step(model, 0.0)
    lr_processes.append(optimizer.get_lr())

    if timestep == 4500:
        scheduler = ExponentialDecayLR(optimizer.optimizer, optimizer.get_lr(), 1e-05, 1000)
        optimizer.set_scheduler(scheduler, 100000)

plt.title('Test Optimizer class')
plt.plot(lr_processes)
plt.xlabel('timestep', fontsize='large')
plt.ylabel('lr', fontsize='large')
plt.show()
