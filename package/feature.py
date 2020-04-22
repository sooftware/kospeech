import torch
import librosa
import numpy as np
import random
from package.definition import logger


def get_librosa_melspectrogram(filepath, n_mels=80,
                               del_silence=False, input_reverse=True, normalize=False,
                               sr=16000, window_size=20, stride=10):
    r"""
    Compute a mel-scaled soectrigram (or Log-Mel).

    Args: filepath, n_mels, del_silence, input_reverse, normalize
        filepath (str): specific path of audio file
        n_mels (int): number of mel filter
        del_silence (bool): flag indication whether to delete silence or not (default: True)
        input_reverse (bool): flag indication whether to reverse input or not (default: True)
        normalize (bool): flag indication whether to normalize spectrum or not (default:True)

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

    Returns: spectrogram
        - **spectrogram** (torch.Tensor): return Mel-Spectrogram (or Log-Mel) feature

    Examples::
        Generate mel spectrogram from a time series

    >>> get_librosa_melspectrogram("KaiSpeech_021458.pcm", n_mels=80, input_reverse=True, normalize=True)
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

        signal = np.array([float(x) for x in pcm])

    elif filepath.split('.')[-1] == 'wav':
        signal, _ = librosa.core.load(filepath, sr=sr)

    else:
        raise ValueError("Invalid format !!")

    N_FFT = int(sr * 0.001 * window_size)
    STRIDE = int(sr * 0.001 * stride)

    if del_silence:
        non_silence_ids = librosa.effects.split(y=signal, top_db=30)
        signal = np.concatenate([signal[start:end] for start, end in non_silence_ids])

    spectrogram = librosa.feature.melspectrogram(signal, sr=sr, n_mels=n_mels, n_fft=N_FFT, hop_length=STRIDE)
    spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

    if normalize:
        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram -= mean
        spectrogram /= std

    if input_reverse:
        spectrogram = spectrogram[:, ::-1]

    spectrogram = torch.FloatTensor(np.ascontiguousarray(np.swapaxes(spectrogram, 0, 1)))

    return spectrogram


def get_librosa_mfcc(filepath, n_mfcc=40,
                     del_silence=False, input_reverse=True, normalize=True,
                     sr=16000, window_size=20, stride=10):
    r""":
    Mel-frequency cepstral coefficients (MFCCs)

    Args: filepath, n_mfcc, del_silence, input_reverse, normalize
        filepath (str): specific path of audio file
        n_mfcc (int): number of mel filter
        del_silence (bool): flag indication whether to delete silence or not (default: True)
        input_reverse (bool): flag indication whether to reverse input or not (default: True)
        normalize (bool): flag indication whether to normalize spectrum or not (default:True)

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

    Returns: spectrogram
        - **spectrogram** (torch.Tensor): return mel frequency cepstral coefficient feature

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

        signal = np.array([float(x) for x in pcm])

    elif filepath.split('.')[-1] == 'wav':
        signal, _ = librosa.core.load(filepath, sr=sr)

    else:
        raise ValueError("Invalid format !!")

    N_FFT = int(sr * 0.001 * window_size)
    STRIDE = int(sr * 0.001 * stride)

    if del_silence:
        non_silence_ids = librosa.effects.split(signal, top_db=30)
        signal = np.concatenate([signal[start:end] for start, end in non_silence_ids])

    spectrogram = librosa.feature.mfcc(signal, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=STRIDE)

    if normalize:
        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram -= mean
        spectrogram /= std

    if input_reverse:
        spectrogram = spectrogram[:, ::-1]

    spectrogram = torch.FloatTensor(np.ascontiguousarray(np.swapaxes(spectrogram, 0, 1)))

    return spectrogram


def spec_augment(spectrogram, time_mask_para=70, freq_mask_para=20, time_mask_num=2, freq_mask_num=2):
    """
    Provides Augmentation for audio

    Args:
        spectrogram (torch.Tensor): spectrum
        time_mask_para (int): Hyper Parameter for Time Masking to limit time masking length
        freq_mask_para (int): Hyper Parameter for Freq Masking to limit freq masking length
        time_mask_num (int): how many time-masked area to make
        freq_mask_num (int): how many freq-masked area to make

    Returns: feat
        - **feat**: Augmented feature

    Reference:
        「SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition」Google Brain Team. 2019.
         https://github.com/DemisEom/SpecAugment/blob/master/SpecAugment/spec_augment_pytorch.py

    Examples::
        Generate spec augmentation from a feature

        >>> spec_augment(spectrogram, time_mask_para=70, freq_mask_para=20, time_mask_num=2, freq_mask_num=2)
        Tensor([[ -5.229e+02,  0, ...,  -5.229e+02,  -5.229e+02],
                [  7.105e-15,  0, ...,  -7.105e-15,  -7.105e-15],
                ...,
                [          0,  0, ...,           0,           0],
                [  3.109e-14,  0, ...,   2.931e-14,   2.931e-14]])
    """
    length = spectrogram.size(0)
    n_mels = spectrogram.size(1)

    # time mask
    for _ in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=time_mask_para)
        t = int(t)
        t0 = random.randint(0, length - t)
        spectrogram[t0: t0 + t, :] = 0

    # freq mask
    for _ in range(freq_mask_num):
        f = np.random.uniform(low=0.0, high=freq_mask_para)
        f = int(f)
        f0 = random.randint(0, n_mels - f)
        spectrogram[:, f0: f0 + f] = 0

    return spectrogram
