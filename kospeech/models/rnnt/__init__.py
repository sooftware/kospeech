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

from dataclasses import dataclass


@dataclass
class RNNTransducerConfig:
    architecture: str = "rnnt"
    num_encoder_layers: int = 4
    num_decoder_layers: int = 1
    encoder_hidden_state_dim: int = 320
    decoder_hidden_state_dim: int = 512
    output_dim: int = 512
    rnn_type: str = "lstm"
    bidirectional: bool = True
    encoder_dropout_p: float = 0.2
    decoder_dropout_p: float = 0.2
