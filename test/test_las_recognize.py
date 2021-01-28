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

from kospeech.models import Speller, ListenAttendSpell
from kospeech.models.las.encoder import Listener

B, T, D, H = 3, 12345, 80, 32

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

inputs = torch.rand(B, T, D).to(device)
input_lengths = torch.IntTensor([T, T - 100, T - 1000])
targets = torch.LongTensor([[1, 1, 2], [3, 4, 2], [7, 2, 0]])

encoder = Listener(input_dim=D, hidden_state_dim=H, joint_ctc_attention=False)
decoder = Speller(num_classes=10, hidden_state_dim=H << 1, max_length=10)
model = ListenAttendSpell(encoder, decoder).to(device)

model.recognize(inputs, input_lengths)
print("teacher_forcing_ratio=0.0 PASS")

model(inputs, input_lengths, targets, teacher_forcing_ratio=1.0)
print("teacher_forcing_ratio=1.0 PASS")
