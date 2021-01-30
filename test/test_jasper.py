# Copyright (c) 2021, Soohwan Kim. All rights reserved.
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

import torch
import torch.nn as nn

from kospeech.models.jasper.model import Jasper

batch_size = 3
sequence_length = 14321
dimension = 80

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

model = Jasper(num_classes=10, version='5x3').to(device)

criterion = nn.CTCLoss(blank=3, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-04)

for i in range(10):
    inputs = torch.rand(batch_size, sequence_length, dimension).to(device)
    input_lengths = torch.IntTensor([12345, 12300, 12000])
    targets = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                                [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                                [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]]).to(device)
    target_lengths = torch.LongTensor([9, 8, 7])
    outputs, output_lengths = model(inputs, input_lengths)

    loss = criterion(outputs.transpose(0, 1), targets[:, 1:], output_lengths, target_lengths)
    loss.backward()
    optimizer.step()
    print(loss)
