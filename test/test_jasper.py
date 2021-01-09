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
from kospeech.models.jasper.model import Jasper

inputs = torch.rand(3, 14321, 80)  # BxTxD
input_lengths = torch.LongTensor([100, 90, 80])

model = Jasper(num_classes=10, version='10x5')
# print(model)
output, output_lengths = model(inputs, input_lengths)

print(output)
print(output_lengths)
