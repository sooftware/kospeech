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

from kospeech.models import ModelConfig
from kospeech.models.conformer.model import Conformer


@dataclass
class ConformerConfig(ModelConfig):
    architecture: str = "conformer"
    feed_forward_expansion_factor: int = 4
    conv_expansion_factor: int = 2
    input_dropout_p: float = 0.1
    feed_forward_dropout_p: float = 0.1
    attention_dropout_p: float = 0.1
    conv_dropout_p: float = 0.1
    decoder_dropout_p: float = 0.1
    conv_kernel_size: int = 31
    half_step_residual: bool = True
    num_decoder_layers: int = 1
    decoder_rnn_type: str = "lstm"
    decoder: str = "None"


@dataclass
class ConformerLargeConfig(ConformerConfig):
    encoder_dim: int = 512
    decoder_dim: int = 640
    num_encoder_layers: int = 17
    num_attention_heads: int = 8


@dataclass
class ConformerMediumConfig(ConformerConfig):
    encoder_dim: int = 256
    decoder_dim: int = 640
    num_encoder_layers: int = 16
    num_attention_heads: int = 4


@dataclass
class ConformerSmallConfig(ConformerConfig):
    encoder_dim: int = 144
    decoder_dim: int = 320
    num_encoder_layers: int = 16
    num_attention_heads: int = 4
