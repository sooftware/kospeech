# Soohwan Kim, Seyoung Bae, Cheolhwang Won.
# @ArXiv : KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.

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
