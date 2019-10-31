"""
feature for Speech Recognition
get_librosa_melspectrogram : get Mel-Spectrogram or log Mel feature using librosa library
get_librosa_mfcc : get MFCC (Mel-Frequency-Cepstral-Coefficient) feature using librosa library

FRAME_LENGTH : 21ms
STRIDE : 5.2ms ( 75% duplicated )

FRAME_LENGTH = N_FFT / SAMPLE_RATE => N_FFT = 336
STRIDE = HOP_LENGTH / SAMPLE_RATE => HOP_LENGTH = 84

+++++
remove silence Using librosa
+++++

 -*- Soo-Hwan -*-
"""

import torch
import librosa
import numpy as np

SAMPLE_RATE = 16000
N_FFT = 336
HOP_LENGTH = 84

def get_librosa_melspectrogram(filepath, n_mels = 80, rm_silence = True, type_ = 'mel'):
    sig, sr = librosa.core.load(filepath, SAMPLE_RATE)
    # delete silence
    if rm_silence:
        non_silence_indices = librosa.effects.split(sig, top_db = 30)
        sig = np.concatenate([sig[start:end] for start, end in non_silence_indices])
    mel = librosa.feature.melspectrogram(sig, n_mels = n_mels, n_fft = N_FFT, hop_length = HOP_LENGTH)

    # get log Mel
    if type_ == 'log_mel':
        log_mel = librosa.amplitude_to_db(mel, ref = np.max)
        return torch.FloatTensor(log_mel).transpose(0, 1)
    # get Mel-Spectrogram
    return torch.FloatTensor(mel).transpose(0, 1)

def get_librosa_mfcc(filepath, n_mfcc = 40, rm_silence = True):
    sig, sr = librosa.core.load(filepath, SAMPLE_RATE)
    # delete silence
    if rm_silence:
        non_silence_indices = librosa.effects.split(sig, top_db=30)
        sig = np.concatenate([sig[start:end] for start, end in non_silence_indices])
    mfccs = librosa.feature.mfcc(y = sig, sr = sr, hop_length = HOP_LENGTH, n_mfcc = n_mfcc, n_fft = N_FFT)

    return torch.FloatTensor(mfccs).transpose(0, 1)