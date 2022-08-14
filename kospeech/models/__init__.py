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


from kospeech.models.deepspeech2.model import DeepSpeech2
from kospeech.models.las.encoder import EncoderRNN
from kospeech.models.las.decoder import DecoderRNN
from kospeech.models.rnnt import RNNTransducerConfig
from kospeech.models.rnnt.model import RNNTransducer
from kospeech.models.las.model import ListenAttendSpell
from kospeech.models.transformer.model import SpeechTransformer
from kospeech.models.jasper.model import Jasper
from kospeech.models.conformer.model import Conformer
from kospeech.models.las import ListenAttendSpellConfig, JointCTCAttentionLASConfig
from kospeech.models.transformer import TransformerConfig, JointCTCAttentionTransformerConfig
from kospeech.models.deepspeech2 import DeepSpeech2Config
from kospeech.models.jasper import JasperConfig
from kospeech.models.conformer import ConformerSmallConfig, ConformerMediumConfig, ConformerLargeConfig
