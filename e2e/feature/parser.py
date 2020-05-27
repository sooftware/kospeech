import librosa
import torch
import platform
import random
import numpy as np
from abc import abstractmethod
from e2e.feature.core import split
from e2e.modules.global_ import logger

if platform.system() == 'Linux':
    import torchaudio


class AudioParser(object):
    """
    Provides load_audio(), inject_noise(), instancewise_standardization(), spec_audment() function.

    Note:
        Do not use this class directly, use one of the sub classes.

    Method:
        - **parse_audio()**: abstract method. you have to override this method.
        - **parse_script()**: abstract method. you have to override this method.
        - **load_audio()**: load audio file to signal. (PCM)
        - **instancewise_standardization**: provides instance-wise standardization normalization.
    """
    @staticmethod
    def load_audio(audio_path, del_silence):
        """
        Load audio file to signal. (PCM)
        if del_silence is True, Eliminate all signals below 30dB
        """
        try:
            signal = np.memmap(audio_path, dtype='h', mode='r').astype('float32')

            if del_silence:
                non_silence_indices = split(signal, top_db=30)
                signal = np.concatenate([signal[start:end] for start, end in non_silence_indices])

            signal = signal / 32767  # normalize audio
            return signal
        except ValueError:
            logger.debug('ValueError in {0}'.format(audio_path))
            return None
        except RuntimeError:
            logger.debug('RuntimeError in {0}'.format(audio_path))
            return None

    @staticmethod
    def instancewise_standardization(spectrogram):
        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram -= mean
        spectrogram /= std
        return spectrogram

    @staticmethod
    def inject_noise(signal, noise_factor):
        noise = np.random.randn(len(signal))
        signal += noise * noise_factor
        return signal

    @abstractmethod
    def parse_audio(self, audio_path, augment_method):
        raise NotImplementedError

    @abstractmethod
    def parse_script(self, script_path):
        raise NotImplementedError

    @staticmethod
    def spec_augment(spectrogram, time_mask_para, freq_mask_para, time_mask_num, freq_mask_num):
        """ Provides SpecAugmentation for audio """
        length = spectrogram.size(0)
        n_mels = spectrogram.size(1)

        # time mask
        for _ in range(time_mask_num):
            t = np.random.uniform(low=0.0, high=time_mask_para)
            t = int(t)
            if length - t > 0:
                t0 = random.randint(0, length - t)
                spectrogram[t0: t0 + t, :] = 0

        # freq mask
        for _ in range(freq_mask_num):
            f = np.random.uniform(low=0.0, high=freq_mask_para)
            f = int(f)
            f0 = random.randint(0, n_mels - f)
            spectrogram[:, f0: f0 + f] = 0

        return spectrogram


class SpectrogramParser(AudioParser):
    """
    Parses audio file into spectrogram with various options.

    Args:
        feature_extract_by (str): which library to use for feature extraction: [librosa, torchaudio] (default: librosa)
        sample_rate (int): sample rate
        n_mels (int): number of mel filter
        window_size (int): window size (ms)
        stride (int): forwarding size (ms)
        del_silence (bool): flag indication whether to delete silence or not (default: True)
        input_reverse (bool): flag indication whether to reverse input or not (default: True)
        normalize (bool): flag indication whether to normalize spectrum or not (default:True)
        time_mask_para (int): Hyper Parameter for Time Masking to limit time masking length
        freq_mask_para (int): Hyper Parameter for Freq Masking to limit freq masking length
        time_mask_num (int): how many time-masked area to make
        freq_mask_num (int): how many freq-masked area to make
        sos_id (int): start of sentence token`s identification
        eos_id (int): end of sentence token`s identification
        target_dict (dict): dictionary of filename and labels
    """
    VANILLA = 0           # Not apply augmentation
    SPEC_AUGMENT = 1      # SpecAugment
    NOISE_INJECTION = 2   # Noise Injection

    def __init__(self, feature_extract_by='librosa', sample_rate=16000, n_mels=80, window_size=20, stride=10,
                 del_silence=False, input_reverse=True, normalize=False,
                 time_mask_para=70, freq_mask_para=12, time_mask_num=2, freq_mask_num=2,
                 sos_id=1, eos_id=2, target_dict=None):
        super(SpectrogramParser, self).__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.window_size = window_size
        self.del_silence = del_silence
        self.input_reverse = input_reverse
        self.normalize = normalize
        self.n_fft = int(sample_rate * 0.001 * window_size)
        self.hop_length = int(sample_rate * 0.001 * stride)
        self.time_mask_para = time_mask_para
        self.freq_mask_para = freq_mask_para
        self.time_mask_num = time_mask_num
        self.freq_mask_num = freq_mask_num
        self.feature_extract_by = feature_extract_by
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.target_dict = target_dict

        if feature_extract_by == 'torchaudio':
            self.transforms = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,  win_length=window_size,
                                                                   hop_length=self.hop_length,  n_fft=self.n_fft,
                                                                   n_mels=n_mels)
            self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def parse_audio(self, audio_path, augment_method):
        """
        Parses audio (Get Mel-Spectrogram features). @Override

        Args:
             audio_path (str): path of audio file
             augment_method (int): flag indication which augmentation method to use.

        Returns: spectrogram
            - **spectrogram** (torch.FloatTensor): Mel-Spectrogram feature from audio file.
        """
        signal = self.load_audio(audio_path, self.del_silence)

        if signal is None:  # Exception handling
            return None
        elif augment_method == self.NOISE_INJECTION:  # Noise injection
            signal = self.inject_noise(signal, noise_factor=0.01)

        if self.feature_extract_by == 'torchaudio':
            spectrogram = self.transforms(torch.FloatTensor(signal))
            spectrogram = self.amplitude_to_db(spectrogram)
            spectrogram = spectrogram.numpy()

        else:
            spectrogram = librosa.feature.melspectrogram(signal, self.sample_rate, n_mels=self.n_mels,
                                                         n_fft=self.n_fft, hop_length=self.hop_length)
            spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

        if self.normalize:
            spectrogram = self.instancewise_standardization(spectrogram)

        if self.input_reverse:   # Refer to "Sequence to Sequence Learning with Neural Network" paper
            spectrogram = spectrogram[:, ::-1]

        spectrogram = torch.FloatTensor(np.ascontiguousarray(np.swapaxes(spectrogram, 0, 1)))

        if augment_method == self.SPEC_AUGMENT:
            spectrogram = self.spec_augment(spectrogram, self.time_mask_para, self.freq_mask_para,
                                            self.time_mask_num, self.freq_mask_num)
        return spectrogram

    @abstractmethod
    def parse_script(self, script_path):
        raise NotImplementedError
