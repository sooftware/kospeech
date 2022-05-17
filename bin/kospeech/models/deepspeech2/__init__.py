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

from dataclasses import dataclass
from kospeech.models import ModelConfig


@dataclass
class DeepSpeech2Config(ModelConfig):
    architecture: str = "deepspeech2"
    use_bidirectional: bool = True
    rnn_type: str = "gru"
    hidden_dim: int = 1024
    activation: str = "hardtanh"
    num_encoder_layers: int = 3
