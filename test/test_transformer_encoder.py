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

from kospeech.model_builder import build_transformer

batch_size = 4
seq_length = 200
target_length = 10
input_size = 80

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

transformer = build_transformer(
    num_classes=10,
    d_model=16,
    d_ff=32,
    num_heads=2,
    input_dim=input_size,
    num_encoder_layers=3,
    num_decoder_layers=2,
    extractor='vgg',
    dropout_p=0.1,
    device=device,
    pad_id=0,
    sos_id=1,
    eos_id=2,
    joint_ctc_attention=True,
    max_length=10,
)
transformer_encoder = transformer.module.encoder

criterion = nn.CTCLoss(blank=3, zero_infinity=True)
optimizer = torch.optim.Adam(transformer_encoder.parameters(), lr=1e-04)

for i in range(10):
    inputs = torch.FloatTensor(batch_size, seq_length, input_size).to(device)
    input_lengths = torch.LongTensor([seq_length, seq_length - 10, seq_length - 20, seq_length - 30])
    targets = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                                [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                                [1, 3, 3, 3, 3, 3, 4, 2, 0, 0],
                                [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]]).to(device)
    target_lengths = torch.LongTensor([9, 8, 7, 7])

    _, output_lengths, encoder_log_probs = transformer_encoder(inputs, input_lengths)
    loss = criterion(encoder_log_probs.transpose(0, 1), targets[:, 1:], output_lengths, target_lengths)
    loss.backward()
    optimizer.step()
    print(loss)
