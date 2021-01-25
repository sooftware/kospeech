# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from kospeech.optim.__init__ import Optimizer
from kospeech.optim.lr_scheduler.transformer_lr_scheduler import TransformerLRScheduler


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.projection = nn.Linear(10, 10)

    def forward(self):
        pass


PEAK_LR = 1e-04
FINAL_LR = 1e-07
WARMUP_STEPS = 1000
MAX_GRAD_NORM = 400
DECAY_STEPS = 10000
TOTAL_STEPS = WARMUP_STEPS + DECAY_STEPS + 10000

model = Model()

optimizer = optim.Adam(model.parameters(), lr=0.0)
scheduler = TransformerLRScheduler(
    optimizer=optimizer,
    peak_lr=PEAK_LR,
    final_lr=FINAL_LR,
    warmup_steps=WARMUP_STEPS,
    decay_steps=DECAY_STEPS,
    final_lr_scale=0.001,
)
optimizer = Optimizer(optimizer, scheduler, TOTAL_STEPS, MAX_GRAD_NORM)
lr_processes = list()

for timestep in range(TOTAL_STEPS):
    optimizer.step(model)
    lr_processes.append(optimizer.get_lr())

plt.title('Test Optimizer class')
plt.plot(lr_processes)
plt.xlabel('timestep', fontsize='large')
plt.ylabel('lr', fontsize='large')
plt.show()
