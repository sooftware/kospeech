import torch
import librosa
import numpy as np
import random
from package.definition import logger


def get_librosa_melspectrogram(filepath, n_mels=128, del_silence=False, input_reverse=True, mel_type='log_mel'):
    r"""
    Compute a mel-scaled soectrigram (or Log-Mel).

    Args:
        filepath (str): specific path of audio file
        n_mels (int): number of mel filter
        del_silence (bool): flag indication whether to delete silence or not (default: True)
        mel_type (str): if 'log_mel' return log-mel (default: 'log_mel')
        input_reverse (bool): flag indication whether to reverse input or not (default: True)

    Feature Parameters:
        - **sample rate**: A.I Hub dataset`s sample rate is 16,000
        - **frame length**: 25ms
        - **stride**: 10ms
        - **overlap**: 15ms
        - **window**: Hamming Window

    .. math::
        \begin{array}{ll}
        NFFT = sr * frame length \\
        Hop Length = sr * stride \\
        \end{array}

    Returns: mel_spectrogram
        - **mel_spectrogram** (torch.Tensor): return Mel-Spectrogram (or Log-Mel) feature

    Examples::
        Generate mel spectrogram from a time series

    >>> get_librosa_melspectrogram("KaiSpeech_021458.pcm", n_mels=128, input_reverse=True)
    Tensor([[  2.891e-07,   2.548e-03, ...,   8.116e-09,   5.633e-09],
            [  1.986e-07,   1.162e-02, ...,   9.332e-08,   6.716e-09],
            ...,
            [  3.668e-09,   2.029e-08, ...,   3.208e-09,   2.864e-09],
            [  2.561e-10,   2.096e-09, ...,   7.543e-10,   6.101e-10]])
    """
    if filepath.split('.')[-1] == 'pcm':
        try:
            pcm = np.memmap(filepath, dtype='h', mode='r')

        except:
            logger.info("%s Error Occur !!" % filepath)
            return None

        sig = np.array([float(x) for x in pcm])

    elif filepath.split('.')[-1] == 'wav':
        sig, _ = librosa.core.load(filepath, sr=16000)

    else:
        raise ValueError("Invalid format !!")

    if del_silence:
        non_silence_ids = librosa.effects.split(y=sig, top_db=30)
        sig = np.concatenate([sig[start:end] for start, end in non_silence_ids])

    mel_spectrogram = librosa.feature.melspectrogram(sig, sr=16000, n_mels=n_mels, n_fft=400, hop_length=160, window='hamming')

    if mel_type == 'log_mel':
        mel_spectrogram = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)

    if input_reverse:
        mel_spectrogram = mel_spectrogram[:,::-1]

    return torch.FloatTensor( np.ascontiguousarray( np.swapaxes(mel_spectrogram, 0, 1) ) )


def get_librosa_mfcc(filepath, n_mfcc=40, del_silence=False, input_reverse=True):
    r""":
    Mel-frequency cepstral coefficients (MFCCs)

    Args:
        filepath (str): specific path of audio file
        n_mfcc (int): number of mel filter
        del_silence (bool): flag indication whether to delete silence or not (default: True)
        input_reverse (bool): flag indication whether to reverse input or not (default: True)

    Feature Parameters:
        - **sample rate**: A.I Hub dataset`s sample rate is 16,000
        - **frame length**: 25ms
        - **stride**: 10ms
        - **overlap**: 15ms
        - **window**: Hamming Window

    .. math::
        \begin{array}{ll}
        NFFT = sr * frame length \\
        HopLength = sr * stride \\
        \end{array}

    Returns: mfcc
        - **mfcc** (torch.Tensor): return mel frequency cepstral coefficient feature

    Examples::
        Generate mfccs from a time series

        >>> get_librosa_mfcc("KaiSpeech_021458.pcm", n_mfcc=40, input_reverse=True)
        Tensor([[ -5.229e+02,  -4.944e+02, ...,  -5.229e+02,  -5.229e+02],
                [  7.105e-15,   3.787e+01, ...,  -7.105e-15,  -7.105e-15],
                ...,
                [  1.066e-14,  -7.500e+00, ...,   1.421e-14,   1.421e-14],
                [  3.109e-14,  -5.058e+00, ...,   2.931e-14,   2.931e-14]])
    """
    if filepath.split('.')[-1] == 'pcm':
        try:
            pcm = np.memmap(filepath, dtype='h', mode='r')

        except:
            logger.info("%s Error Occur !!" % filepath)
            return None

        sig = np.array([float(x) for x in pcm])

    elif filepath.split('.')[-1] == 'wav':
        sig, _ = librosa.core.load(filepath, sr=16000)

    else:
        raise ValueError("Invalid format !!")

    if del_silence:
        non_silence_ids = librosa.effects.split(sig, top_db=30)
        sig = np.concatenate([sig[start:end] for start, end in non_silence_ids])

    mfcc = librosa.feature.mfcc(sig, sr=16000, hop_length=160, n_mfcc=n_mfcc, n_fft=400, window='hamming')

    if input_reverse:
        mfcc = mfcc[:,::-1]

    return torch.FloatTensor( np.ascontiguousarray( np.swapaxes(mfcc, 0, 1) ) )



def spec_augment(feat, T=70, F=20, time_mask_num=2, freq_mask_num=2):
    """
    Provides Augmentation for audio

    Args:
        feat (torch.Tensor): input data feature
        T (int): Hyper Parameter for Time Masking to limit time masking length
        F (int): Hyper Parameter for Freq Masking to limit freq masking length
        time_mask_num (int): how many time-masked area to make
        freq_mask_num (int): how many freq-masked area to make

    Returns: feat
        - **feat**: Augmented feature

    Reference:
        「SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition」Google Brain Team. 2019.
         https://github.com/DemisEom/SpecAugment/blob/master/SpecAugment/spec_augment_pytorch.py

    Examples::
        Generate spec augmentation from a feature

        >>> spec_augment(feat, T = 70, F = 20, time_mask_num = 2, freq_mask_num = 2)
        Tensor([[ -5.229e+02,  0, ...,  -5.229e+02,  -5.229e+02],
                [  7.105e-15,  0, ...,  -7.105e-15,  -7.105e-15],
                ...,
                [          0,  0, ...,           0,           0],
                [  3.109e-14,  0, ...,   2.931e-14,   2.931e-14]])
    """
    length = feat.size(0)
    n_mels = feat.size(1)

    # time mask
    for _ in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=T)
        t = int(t)
        t0 = random.randint(0, length - t)
        feat[t0 : t0 + t, :] = 0

    # freq mask
    for _ in range(freq_mask_num):
        f = np.random.uniform(low=0.0, high=F)
        f = int(f)
        f0 = random.randint(0, n_mels - f)
        feat[:, f0 : f0 + f] = 0

    return feat