import torch
import librosa
import platform
import numpy as np
from torch import Tensor, FloatTensor

# torchaudio is only supported on Linux (Linux, Mac)
if platform.system().lower() == 'linux':
    try:
        import torchaudio
    except ImportError:
        raise ImportError("SpectrogramPaser requires torchaudio package.")


class Spectrogram(object):
    """
    Create a spectrogram from a audio signal.

    Args: sample_rate, window_size, frame_shift, feature_extract_by
        sample_rate (int): Sample rate of audio signal. (Default: 16000)
        frame_length (int): frame length for spectrogram (ms) (Default : 20)
        frame_shift (int): Length of hop between STFT windows. (ms) (Default: 10)
    """
    def __init__(self, sample_rate: int = 16000,
                 frame_length: int = 20, frame_shift: int = 10,
                 feature_extract_by: str = 'torch') -> None:
        self.sample_rate = sample_rate
        self.feature_extract_by = feature_extract_by.lower()

        if self.feature_extract_by == 'kaldi':
            # torchaudio is only supported on Linux (Linux, Mac)
            assert platform.system().lower() == 'linux' or platform.system().lower() == 'darwin'
            self.frame_length = frame_length
            self.frame_shift = frame_shift

        else:
            self.n_fft = int(round(sample_rate * 0.001 * frame_length))
            self.hop_length = int(round(sample_rate * 0.001 * frame_shift))

    def __call__(self, signal):
        if self.feature_extract_by == 'kaldi':
            spectrogram = torchaudio.compliance.kaldi.spectrogram(
                Tensor(signal).unsqueeze(0),
                frame_length=self.frame_length, frame_shift=self.frame_shift,
                sample_frequency=self.sample_rate
            ).transpose(0, 1)

        else:
            spectrogram = torch.stft(
                Tensor(signal), self.n_fft, hop_length=self.hop_length,
                win_length=self.n_fft, window=torch.hamming_window(self.n_fft),
                center=False, normalized=False, onesided=True
            )
            spectrogram = (spectrogram[:, :, 0].pow(2) + spectrogram[:, :, 1].pow(2)).pow(0.5)
            spectrogram = np.log1p(spectrogram.numpy())

        return spectrogram


class MelSpectrogram(object):
    """
    Create MelSpectrogram for a raw audio signal. This is a composition of Spectrogram and MelScale.

    Args: sample_rate, n_mels, frame_length, frame_shift, feature_extract_by
        sample_rate (int): Sample rate of audio signal. (Default: 16000)
        n_mels (int):  Number of mfc coefficients to retain. (Default: 80)
        frame_length (int): frame length for spectrogram (ms) (Default : 20)
        frame_shift (int): Length of hop between STFT windows. (ms) (Default: 10)
        feature_extract_by (str): which library to use for feature extraction(default: librosa)
    """
    def __init__(self, sample_rate=16000, n_mels=80, frame_length=20, frame_shift=10, feature_extract_by='librosa'):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = int(round(sample_rate * 0.001 * frame_length))
        self.hop_length = int(round(sample_rate * 0.001 * frame_shift))
        self.feature_extract_by = feature_extract_by.lower()

        if self.feature_extract_by == 'torchaudio':
            # torchaudio is only supported on Linux (Linux, Mac)
            assert platform.system().lower() == 'linux' or platform.system().lower() == 'darwin'
            self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
            self.transforms = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, win_length=frame_length,
                hop_length=self.hop_length, n_fft=self.n_fft,
                n_mels=n_mels
            )

    def __call__(self, signal):
        if self.feature_extract_by == 'torchaudio':
            melspectrogram = self.transforms(Tensor(signal))
            melspectrogram = self.amplitude_to_db(melspectrogram)
            melspectrogram = melspectrogram.numpy()

        elif self.feature_extract_by == 'librosa':
            melspectrogram = librosa.feature.melspectrogram(
                signal, sr=self.sample_rate, n_mels=self.n_mels,
                n_fft=self.n_fft, hop_length=self.hop_length
            )
            melspectrogram = librosa.amplitude_to_db(melspectrogram, ref=np.max)

        else:
            raise ValueError("Unsupported library : {0}".format(self.feature_extract_by))

        return melspectrogram


class MFCC(object):
    """
    Create the Mel-frequency cepstrum coefficients (MFCCs) from an audio signal.

    Args: sample_rate, n_mfcc, frame_length, frame_shift, feature_extract_by
        sample_rate (int): Sample rate of audio signal. (Default: 16000)
        n_mfcc (int):  Number of mfc coefficients to retain. (Default: 40)
        frame_length (int): frame length for spectrogram (ms) (Default : 20)
        frame_shift (int): Length of hop between STFT windows. (ms) (Default: 10)
        feature_extract_by (str): which library to use for feature extraction(default: librosa)
    """
    def __init__(self, sample_rate=16000, n_mfcc=40, frame_length=20, frame_shift=10, feature_extract_by='librosa'):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = int(round(sample_rate * 0.001 * frame_length))
        self.hop_length = int(round(sample_rate * 0.001 * frame_shift))
        self.feature_extract_by = feature_extract_by.lower()

        if self.feature_extract_by == 'torchaudio':
            # torchaudio is only supported on Linux (Linux, Mac)
            assert platform.system().lower() == 'linux' or platform.system().lower() == 'darwin'
            self.transforms = torchaudio.transforms.MFCC(
                sample_rate=sample_rate, n_mfcc=n_mfcc,
                log_mels=True, win_length=frame_length,
                hop_length=self.hop_length, n_fft=self.n_fft
            )

    def __call__(self, signal):
        if self.feature_extract_by == 'torchaudio':
            mfcc = self.transforms(FloatTensor(signal))
            mfcc = mfcc.numpy()

        elif self.feature_extract_by == 'librosa':
            mfcc = librosa.feature.mfcc(
                y=signal, sr=self.sample_rate, n_mfcc=self.n_mfcc,
                n_fft=self.n_fft, hop_length=self.hop_length
            )

        else:
            raise ValueError("Unsupported library : {0}".format(self.feature_extract_by))

        return mfcc


class FilterBank(object):
    """
    Create a fbank from a raw audio signal. This matches the input/output of Kaldi’s compute-fbank-feats

    Args: sample_rate, n_mels, frame_length, frame_shift, feature_extract_by
        sample_rate (int): Sample rate of audio signal. (Default: 16000)
        n_mels (int):  Number of mfc coefficients to retain. (Default: 80)
        frame_length (int): frame length for spectrogram (ms) (Default : 20)
        frame_shift (int): Length of hop between STFT windows. (ms) (Default: 10)
    """
    def __init__(self, sample_rate=16000, n_mels=80, frame_length=20, frame_shift=10):
        # torchaudio is only supported on Linux (Linux, Mac)
        assert platform.system().lower() == 'linux' or platform.system().lower() == 'darwin'

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift

    def __call__(self, signal):
        return torchaudio.compliance.kaldi.fbank(
            Tensor(signal).unsqueeze(0), num_mel_bins=self.n_mels,
            frame_length=self.frame_length, frame_shift=self.frame_shift,
            window_type='povey'
        ).transpose(0, 1).numpy()
