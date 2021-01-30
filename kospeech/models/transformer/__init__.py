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
class TransformerConfig(ModelConfig):
    architecture: str = "transformer"
    extractor: str = "vgg"
    use_bidirectional: bool = True
    dropout: float = 0.3
    d_model: int = 512
    d_ff: int = 2048
    num_heads: int = 8
    num_encoder_layers: int = 12
    num_decoder_layers: int = 6


@dataclass
class JointCTCAttentionTransformerConfig(TransformerConfig):
    cross_entropy_weight: float = 0.7
    ctc_weight: float = 0.3
    mask_conv: bool = True
    joint_ctc_attention: bool = True
