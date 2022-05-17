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
from kospeech.models.las.encoder import EncoderRNN
from kospeech.models.las.decoder import DecoderRNN


@dataclass
class ListenAttendSpellConfig(ModelConfig):
    architecture: str = "las"
    use_bidirectional: bool = True
    dropout: float = 0.3
    num_heads: int = 4
    num_encoder_layers: int = 3
    num_decoder_layers: int = 2
    rnn_type: str = "lstm"
    hidden_dim: int = 512
    teacher_forcing_ratio: float = 1.0
    attn_mechanism: str = "multi-head"
    teacher_forcing_step: float = 0.01
    min_teacher_forcing_ratio: float = 0.9
    extractor: str = "vgg"
    activation: str = "hardtanh"
    mask_conv: bool = False
    joint_ctc_attention: bool = False


@dataclass
class JointCTCAttentionLASConfig(ListenAttendSpellConfig):
    hidden_dim: int = 768
    cross_entropy_weight: float = 0.7
    ctc_weight: float = 0.3
    mask_conv: bool = True
    joint_ctc_attention: bool = True