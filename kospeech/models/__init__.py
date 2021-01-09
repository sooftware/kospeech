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
from kospeech.models.deepspeech2.model import DeepSpeech2
from kospeech.models.las.encoder import Listener
from kospeech.models.las.decoder import Speller
from kospeech.models.las.topk_decoder import TopKDecoder
from kospeech.models.las.model import ListenAttendSpell
from kospeech.models.transformer.model import SpeechTransformer
from kospeech.models.transformer.sublayers import AddNorm
from kospeech.models.resnet1d.model import ResnetVADModel


@dataclass
class ModelConfig:
    architecture: str = "???"
    teacher_forcing_ratio: float = 1.0
    teacher_forcing_step: float = 0.01
    min_teacher_forcing_ratio: float = 0.9
    dropout: float = 0.3
    bidirectional: bool = False
    joint_ctc_attention: bool = False
    max_len: int = 400


@dataclass
class DeepSpeech2Config(ModelConfig):
    architecture: str = "deepspeech2"
    use_bidirectional: bool = True
    rnn_type: str = "gru"
    hidden_dim: int = 1024
    activation: str = "hardtanh"
    num_encoder_layers: int = 3


@dataclass
class ListenAttendSpellConfig(ModelConfig):
    architecture: str = "las"
    use_bidirectional: bool = True
    dropout: float = 0.3
    num_heads: int = 4
    label_smoothing: float = 0.1
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


@dataclass
class TransformerConfig(ModelConfig):
    architecture: str = "transformer"
    use_bidirectional: bool = True
    dropout: float = 0.3
    d_model: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 12
    num_decoder_layers: int = 6
    ffnet_style: str = "ff"


@dataclass
class JointCTCAttentionTransformerConfig(TransformerConfig):
    cross_entropy_weight: float = 0.7
    ctc_weight: float = 0.3
    mask_conv: bool = True
    joint_ctc_attention: bool = True
