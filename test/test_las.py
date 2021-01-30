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

from kospeech.criterion import JointCTCCrossEntropyLoss
from kospeech.models import ListenAttendSpell
from kospeech.models.las.encoder import EncoderRNN
from kospeech.models.las.decoder import DecoderRNN

B, T, D, H = 3, 12345, 80, 32

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

criterion = JointCTCCrossEntropyLoss(num_classes=10, ignore_index=0, reduction='mean')

inputs = torch.rand(B, T, D).to(device)
input_lengths = torch.IntTensor([T, T - 100, T - 1000])
targets = torch.LongTensor([[1, 1, 2], [1, 4, 2], [1, 2, 0]]).to(device)
target_lengths = torch.LongTensor([2, 2, 1])

encoder = EncoderRNN(input_dim=D, hidden_state_dim=H, joint_ctc_attention=True, num_classes=10)
decoder = DecoderRNN(num_classes=10, hidden_state_dim=H << 1, max_length=10)
model = ListenAttendSpell(encoder, decoder).to(device)

predicted_log_probs, encoder_log_probs, output_lengths = model(inputs, input_lengths, targets, teacher_forcing_ratio=1.0)
outputs = torch.stack(predicted_log_probs, dim=1).to(device)
