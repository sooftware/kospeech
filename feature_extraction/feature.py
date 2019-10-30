"""
feature for Speech Recognition
get_librosa_melspectrogram : get Mel-Spectrogram feature using librosa library
get_librosa_mfcc : get MFCC (Mel-Frequency-Cepstral-Coefficient) feature using librosa library

FRAME_LENGTH : 21ms
STRIDE : 5.2ms ( 75% duplicated )

FRAME_LENGTH = N_FFT / SAMPLE_RATE => N_FFT = 336
STRIDE = HOP_LENGTH / SAMPLE_RATE => STRIDE = 168

=> 확장성보다는 빠른 학습을 위해 미리 계산해서 상수로 적용

by Kai.lib
"""

import torch
import librosa

SAMPLE_RATE = 16000
N_FFT = 336
HOP_LENGTH = 84

def get_librosa_melspectrogram(filepath, n_mels = 80):
    sig, sr = librosa.core.load(filepath, SAMPLE_RATE)
    mel_spectrogram = librosa.feature.melspectrogram(sig, n_mels = n_mels, n_fft = N_FFT, hop_length = HOP_LENGTH)
    mel_spectrogram = torch.FloatTensor(mel_spectrogram).transpose(0, 1)
    return mel_spectrogram


def get_librosa_mfcc(filepath, n_mfcc = 40):
    sig, sr = librosa.core.load(filepath, SAMPLE_RATE)
    mfccs = librosa.feature.mfcc(y = sig, sr = sr, hop_length = HOP_LENGTH, n_mfcc = n_mfcc, n_fft = N_FFT)
    mfccs = torch.FloatTensor(mfccs).transpose(0, 1)
    return mfccs

