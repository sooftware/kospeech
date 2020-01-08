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

def get_librosa_melspectrogram(filepath, n_mels = 80, del_silence = True, mel_type = 'log_mel'):
    """
        Provides Mel-Spectrogram for Speech Recognition
        Args:
            del_silence: flag indication whether to delete silence or not (default: True)
            mel_type: flag indication whether to use mel or log(mel) (default: log(mel))
            n_mels: number of mel filter
        Inputs:
            filepath: specific path of audio file
        Local Varibles:
            SAMPLE_RATE: sampling rate of signal
            N_FFT: number of the Fast Fourier Transform window
            HOP_LENGTH: number of samples between successive frames
        Outputs:
            mel_spec: return log(mel-spectrogram) if mel_type is 'log_mel' or mel-spectrogram
        """
    pcm = np.memmap(filepath, dtype='h', mode='r')
    sig = np.array([float(x) for x in pcm])
    if del_silence:
        non_silence_indices = librosa.effects.split(y=sig, top_db=30)
        sig = np.concatenate([sig[start:end] for start, end in non_silence_indices])
    feat = librosa.feature.melspectrogram(sig, n_mels=n_mels, n_fft=336, hop_length=84)

    # get log Mel
    if mel_type == 'log_mel':
        feat = librosa.amplitude_to_db(feat, ref=np.max)
    return torch.FloatTensor(feat).transpose(0, 1)


def get_wav_melspectrogram(filepath, n_mels = 80, del_silence = True, mel_type = 'log_mel'):
    """
    Provides Mel-Spectrogram for Speech Recognition
    Args:
        del_silence: flag indication whether to delete silence or not (default: True)
        mel_type: flag indication whether to use mel or log(mel) (default: log(mel))
        n_mels: number of mel filter
    Inputs:
        filepath: specific path of audio file
    Local Varibles:
        SAMPLE_RATE: sampling rate of signal
        N_FFT: number of the Fast Fourier Transform window
        HOP_LENGTH: number of samples between successive frames
    Outputs:
        mel_spec: return log(mel-spectrogram) if mel_type is 'log_mel' or mel-spectrogram
    """
    sig, sr = librosa.core.load(filepath=filepath, sr=16000)
    # delete silence
    if del_silence:
        non_silence_indices = librosa.effects.split(y=sig, top_db = 30)
        sig = np.concatenate([sig[start:end] for start, end in non_silence_indices])
    feat = librosa.feature.melspectrogram(sig, n_mels = n_mels, n_fft = 336, hop_length = 84)

    # get log Mel
    if mel_type == 'log_mel':
        feat = librosa.amplitude_to_db(feat, ref = np.max)

    return torch.FloatTensor(feat).transpose(0, 1)

def get_librosa_mfcc(filepath, n_mfcc = 40, del_silence = True):
    """
    Provides Mel Frequency Cepstral Coefficient (MFCC) for Speech Recognition
    Args:
        del_silence: flag indication whether to delete silence or not (default: True)
        n_mfcc: number of mel filter
    Inputs:
        filepath: specific path of audio file
    Local Varibles:
        SAMPLE_RATE: sampling rate of signal
        N_FFT: number of the Fast Fourier Transform window
        HOP_LENGTH: number of samples between successive frames
    Outputs:
        mfcc: return MFCC values of signal
    """
    SAMPLE_RATE = 16000
    N_FFT = 336
    HOP_LENGTH = 84
    sig, sr = librosa.core.load(filepath, SAMPLE_RATE)
    # delete silence
    if del_silence:
        non_silence_indices = librosa.effects.split(sig, top_db=30)
        sig = np.concatenate([sig[start:end] for start, end in non_silence_indices])
    mfcc = librosa.feature.mfcc(y = sig, sr = sr, hop_length = HOP_LENGTH, n_mfcc = n_mfcc, n_fft = N_FFT)

    return torch.FloatTensor(mfcc).transpose(0, 1)