"""
Copyright 2020- Kai.Lib
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import librosa
import numpy as np
from definition import logger

def get_librosa_melspectrogram(filepath, n_mels=80, del_silence=True, mel_type='log_mel', format='pcm'):
    """
        Provides Mel-Spectrogram for Speech Recognition
        Args:
            - **del_silence**: flag indication whether to delete silence or not (default: True)
            - **mel_type**: flag indication whether to use mel or log(mel) (default: log(mel))
            - **n_mels**: number of mel filter
            - **format**:
        Inputs:
            - **filepath**: specific path of audio file
        Comment:
            - **sample rate**: A.I Hub dataset`s sample rate is 16,000
            - **frame length**: 21ms
            - **stride**: 5.2ms
            - **overlap**: 15.8ms (≒75%)
        Outputs:
            mel_spec: return log(mel-spectrogram) if mel_type is 'log_mel' or mel-spectrogram
        """
    if format == 'pcm':
        pcm = np.memmap(filepath, dtype='h', mode='r')
        sig = np.array([float(x) for x in pcm])
    elif format == 'wav':
        sig, _ = librosa.core.load(filepath=filepath, sr=16000)
    else:
        logger.info("get_librosa_melspectrogram() : Invalid format")

    if del_silence:
        non_silence_indices = librosa.effects.split(y=sig, top_db=30)
        sig = np.concatenate([sig[start:end] for start, end in non_silence_indices])
    feat = librosa.feature.melspectrogram(sig, n_mels=n_mels, n_fft=336, hop_length=84)

    if mel_type == 'log_mel':
        feat = librosa.amplitude_to_db(feat, ref=np.max)
    return torch.FloatTensor(feat).transpose(0, 1)

def get_librosa_mfcc(filepath, n_mfcc = 40, del_silence = True, format='pcm'):
    """
    Provides Mel Frequency Cepstral Coefficient (MFCC) for Speech Recognition
    Args:
        del_silence: flag indication whether to delete silence or not (default: True)
        n_mfcc: number of mel filter
    Inputs:
        filepath: specific path of audio file
    Comment:
        - **sample rate**: A.I Hub dataset`s sample rate is 16,000
        - **frame length**: 21ms
        - **stride**: 5.2ms
        - **overlap**: 15.8ms (≒75%)
    Outputs:
        mfcc: return MFCC values of signal
    """
    if format == 'pcm':
        pcm = np.memmap(filepath, dtype='h', mode='r')
        sig = np.array([float(x) for x in pcm])
    elif format == 'wav':
        sig, sr = librosa.core.load(filepath=filepath, sr=16000)
    else:
        logger.info("get_librosa_mfcc() : Invalid format")

    if del_silence:
        non_silence_indices = librosa.effects.split(sig, top_db=30)
        sig = np.concatenate([sig[start:end] for start, end in non_silence_indices])
    mfcc = librosa.feature.mfcc(y = sig, hop_length = 84, n_mfcc = n_mfcc, n_fft = 336)

    return torch.FloatTensor(mfcc).transpose(0, 1)