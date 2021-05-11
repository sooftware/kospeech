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

import torch
import torch.nn as nn

from kospeech.models import ListenAttendSpell

B, T, D, H = 3, 12345, 80, 32

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

model = ListenAttendSpell(
    input_dim=D,
    num_classes=10,
    encoder_hidden_state_dim=H,
    decoder_hidden_state_dim=H << 1,
    bidirectional=True,
    max_length=10,
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-04)

for i in range(10):
    inputs = torch.rand(B, T, D).to(device)
    input_lengths = torch.IntTensor([12345, 12300, 12000])
    targets = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                                [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                                [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]]).to(device)
    outputs, output_lengths, encoder_log_probs = model(inputs, input_lengths, targets, teacher_forcing_ratio=1.0)

    loss = criterion(outputs.contiguous().view(-1, outputs.size(-1)), targets[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(loss)
