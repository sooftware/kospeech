import torch
import librosa
import numpy as np
import random
#from utils.define import logger

def get_librosa_melspectrogram(filepath, n_mels=128, del_silence=False, input_reverse=True, mel_type='log_mel', format='pcm'):
    """
    Compute a mel-scaled soectrigram (or Log-Mel).

    Parameters:
        - **filepath** (str): specific path of audio file
        - **n_mels** (int): number of mel filter
        - **del_silence** (bool): flag indication whether to delete silence or not (default: True)
        - **mel_type** (str): if 'log_mel' return log-mel (default: 'log_mel')
        - **input_reverse** (bool): flag indication whether to reverse input or not (default: True)
        - **format** (str): file format ex) pcm, wav (default: pcm)

    Feature Parameters:
        - **sample rate**: A.I Hub dataset`s sample rate is 16,000
        - **frame length**: 25ms
        - **stride**: 10ms
        - **overlap**: 15ms
        - **window**: Hamming Window
        - **n_fft**: sr * frame_length (16,000 * 30ms)
        - **hop_length**: sr * stride (16,000 * 7.5ms)

    Returns:
        - **feat** (torch.Tensor): return Mel-Spectrogram (or Log-Mel)
    """
    if format == 'pcm':
        try:
            pcm = np.memmap(filepath, dtype='h', mode='r')
        except:  # exception handling
            logger.info("%s Error Occur !!" % filepath)
            return None
        signal = np.array([float(x) for x in pcm])
    elif format == 'wav':
        signal, _ = librosa.core.load(filepath, sr=16000)
    else:
        raise ValueError("Invalid format !!")

    if del_silence:
        non_silence_indices = librosa.effects.split(y=signal, top_db=30)
        signal = np.concatenate([signal[start:end] for start, end in non_silence_indices])

    feat = librosa.feature.melspectrogram(signal, sr=16000, n_mels=n_mels, n_fft=400, hop_length=160, window='hamming')

    if mel_type == 'log_mel':
        feat = librosa.amplitude_to_db(feat, ref=np.max)
    if input_reverse:
        feat = feat[:,::-1]

    return torch.FloatTensor( np.ascontiguousarray( np.swapaxes(feat, 0, 1) ) )


def get_librosa_mfcc(filepath = None, n_mfcc = 33, del_silence = False, input_reverse = True, format='pcm'):
    """:
    Mel-frequency cepstral coefficients (MFCCs)

    Parameters:
        - **filepath** (str): specific path of audio file
        - **n_mfcc** (int): number of mel filter
        - **del_silence** (bool): flag indication whether to delete silence or not (default: True)
        - **input_reverse** (bool): flag indication whether to reverse input or not (default: True)
        - **format** (str): file format ex) pcm, wav (default: pcm)

    Feature Parameters:
        - **sample rate**: A.I Hub dataset`s sample rate is 16,000
        - **frame length**: 25ms
        - **stride**: 10ms
        - **overlap**: 15ms
        - **window**: Hamming Window
        - **n_fft**: sr * frame_length (16,000 * 30ms)
        - **hop_length**: sr * stride (16,000 * 7.5ms)

    Returns:
        - **feat** (torch.Tensor): MFCC values of signal
    """
    if format == 'pcm':
        try:
            pcm = np.memmap(filepath, dtype='h', mode='r')
        except: # exception handling
            logger.info("%s Error Occur !!" % filepath)
            return None
        signal = np.array([float(x) for x in pcm])
    elif format == 'wav':
        signal, _ = librosa.core.load(filepath, sr=16000)
    else:
        raise ValueError("Invalid format !!")

    if del_silence:
        non_silence_indices = librosa.effects.split(signal, top_db=30)
        signal = np.concatenate([signal[start:end] for start, end in non_silence_indices])

    feat = librosa.feature.mfcc(signal, 16000, hop_length = 160, n_mfcc = 33, n_fft = 400, window = 'hamming')
    if input_reverse:
        feat = feat[:,::-1]

    return torch.FloatTensor( np.ascontiguousarray( np.swapaxes(feat, 0, 1) ) )

def spec_augment(feat, T = 70, F = 20, time_mask_num = 2, freq_mask_num = 2):
    """
    Provides Augmentation for audio

    Parameters:
        - **feat** (torch.Tensor): input data feature
        - **T** (int): Hyper Parameter for Time Masking to limit time masking length
        - **F** (int): Hyper Parameter for Freq Masking to limit freq masking length
        - **time_mask_num** (int): how many time-masked area to make
        - **freq_mask_num** (int): how many freq-masked area to make

    Returns:
        - **feat**: Augmented feature

    Reference:
        「SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition」Google Brain Team. 2019.
         https://github.com/DemisEom/SpecAugment/blob/master/SpecAugment/spec_augment_pytorch.py
    """
    feat_size = feat.size(1)
    seq_len = feat.size(0)

    # time mask
    for _ in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=T)
        t = int(t)
        t0 = random.randint(0, seq_len - t)
        feat[t0 : t0 + t, :] = 0

    # freq mask
    for _ in range(freq_mask_num):
        f = np.random.uniform(low=0.0, high=F)
        f = int(f)
        f0 = random.randint(0, feat_size - f)
        feat[:, f0 : f0 + f] = 0

    return feat