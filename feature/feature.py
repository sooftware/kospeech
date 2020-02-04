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

def get_librosa_melspectrogram(filepath, n_mels=80, del_silence=True,input_reverse=True, mel_type='log_mel', format='pcm'):
    """
        Provides Mel-Spectrogram for Speech Recognition
        Args:
            - **del_silence**: flag indication whether to delete silence or not (default: True)
            - **mel_type**: flag indication whether to use mel or log(mel) (default: log(mel))
            - **n_mels**: number of mel filter
        Inputs:
            - **filepath**: specific path of audio file
        Comment:
            - **sample rate**: A.I Hub dataset`s sample rate is 16,000
            - **frame length**: 30ms
            - **stride**: 7.5ms
            - **overlap**: 22.5ms (≒75%)
        Outputs:
            mel_spec: return log(mel-spectrogram) if mel_type is 'log_mel' or mel-spectrogram
        """
    if format == 'pcm':
        pcm = np.memmap(filepath, dtype='h', mode='r')
        sig = np.array([float(x) for x in pcm])
    elif format == 'wav':
        sig, _ = librosa.core.load(filepath, sr=16000)

    if del_silence:
        non_silence_indices = librosa.effects.split(y=sig, top_db=30)
        sig = np.concatenate([sig[start:end] for start, end in non_silence_indices])
    feat = librosa.feature.melspectrogram(sig, sr=16000, n_mels=n_mels, n_fft=480, hop_length=120, window='hamming')

    if mel_type == 'log_mel':
        feat = librosa.amplitude_to_db(feat, ref=np.max)
    if input_reverse:
        feat = feat[:,::-1]

    return torch.FloatTensor( np.ascontiguousarray( np.swapaxes(feat, 0, 1) ) )


def get_librosa_mfcc(filepath = None, n_mfcc = 33, del_silence = False, input_reverse = True, format='pcm'):
    """
    Provides Mel Frequency Cepstral Coefficient (MFCC) for Speech Recognition
    Args:
        filepath: specific path of audio file
        n_mfcc: number of mel filter
        del_silence: flag indication whether to delete silence or not (default: True)
        input_reverse: flag indication whether to reverse input or not (default: True)
        format: file format ex) pcm, wav (default: pcm)
    Comment:
        sample rate: A.I Hub dataset`s sample rate is 16,000
        frame length: 25ms
        stride: 10ms
        overlap: 15ms
        window: Hamming Window
        n_fft = sr * frame_length (16,000 * 30ms)
        hop_length = sr * stride (16,000 * 7.5ms)
    Outputs:
        mfcc: return MFCC values of signal
    """
    if format == 'pcm':
        try:
            pcm = np.memmap(filepath, dtype='h', mode='r')
        except: # exception handling
            logger.info("np.memmap error in %s" % filepath)
            return torch.zeros(1)
        sig = np.array([float(x) for x in pcm])
    elif format == 'wav':
        sig, _ = librosa.core.load(filepath, sr=16000)
    else: logger.info("%s is not Supported" % format)

    if del_silence:
        non_silence_indices = librosa.effects.split(sig, top_db=30)
        sig = np.concatenate([sig[start:end] for start, end in non_silence_indices])
    feat = librosa.feature.mfcc(y=sig,sr=16000, hop_length=160, n_mfcc=n_mfcc, n_fft=400, window='hamming')
    if input_reverse:
        feat = feat[:,::-1]

    return torch.FloatTensor( np.ascontiguousarray( np.swapaxes(feat, 0, 1) ) )