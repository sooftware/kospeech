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
class AudioConfig:
    audio_extension: str = "pcm"
    sample_rate: int = 16000
    frame_length: int = 20
    frame_shift: int = 10
    normalize: bool = True
    del_silence: bool = True
    feature_extract_by: str = "kaldi"
    time_mask_num: int = 4
    freq_mask_num: int = 2
    spec_augment: bool = True
    input_reverse: bool = False


@dataclass
class FilterBankConfig(AudioConfig):
    transform_method: str = "fbank"
    n_mels: int = 80
    freq_mask_para: int = 18


@dataclass
class MelSpectrogramConfig(AudioConfig):
    transform_method: str = "mel"
    n_mels: int = 80
    freq_mask_para: int = 18


@dataclass
class MfccConfig(AudioConfig):
    transform_method: str = "mfcc"
    n_mels: int = 40
    freq_mask_para: int = 8


@dataclass
class SpectrogramConfig(AudioConfig):
    transform_method: str = "spectrogram"
    n_mels: int = 161  # Not used
    freq_mask_para: int = 24
