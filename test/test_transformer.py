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
from kospeech.models import SpeechTransformer

batch_size = 4
seq_length = 200
target_length = 20
input_size = 80

transformer = SpeechTransformer(num_classes=10, d_model=16, d_ff=32, num_encoder_layers=3, num_decoder_layers=2)

inputs = torch.FloatTensor(batch_size, seq_length, input_size)
input_lengths = torch.LongTensor([seq_length, seq_length - 10, seq_length - 20, seq_length - 30])
targets = torch.randint(0, 10, size=(batch_size, target_length), dtype=torch.long)

output = transformer(inputs, input_lengths, targets)
print(output)
