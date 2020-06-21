import torch
import librosa
import platform
import numpy as np
from kospeech.data.preprocess.audio import load_audio
from kospeech.data.preprocess.augment import NoiseInjector, SpecAugment

# torchaudio is only supported on Linux
if platform.system() == 'Linux':
    try:
        import torchaudio
    except ImportError:
        raise ImportError("SpectrogramPaser requires torchaudio package.")


class AudioParser(object):
    """
    Provides inteface of audio parser.

    Note:
        Do not use this class directly, use one of the sub classes.

    Method:
        - **parse_audio()**: abstract method. you have to override this method.
        - **parse_transcript()**: abstract method. you have to override this method.
    """
    def __init__(self, dataset_path, noiseset_size, sample_rate=16000, noise_level=0.7, noise_augment=False):
        if noise_augment:
            self.noise_injector = NoiseInjector(dataset_path, noiseset_size, sample_rate, noise_level)

    def parse_audio(self, *args, **kwargs):
        raise NotImplementedError

    def parse_transcript(self, *args, **kwargs):
        raise NotImplementedError


class SpectrogramParser(AudioParser):
    """
    Parses audio file into (spectrogram / mel spectrogram / mfcc) with various options.

    Args:
        feature (str): which feature to use (default: mel)
        feature_extract_by (str): which library to use for feature extraction(default: librosa)
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
                 del_silence=False, input_reverse=True, normalize=False, feature='mel',
                 time_mask_para=70, freq_mask_para=12, time_mask_num=2, freq_mask_num=2,
                 sos_id=1, eos_id=2, target_dict=None,
                 noise_augment=False, dataset_path=None, noiseset_size=0, noise_level=0.7):
        super(SpectrogramParser, self).__init__(dataset_path, noiseset_size, sample_rate, noise_level, noise_augment)
        self.feature = feature.lower()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.window_size = window_size
        self.del_silence = del_silence
        self.input_reverse = input_reverse
        self.normalize = normalize
        self.n_fft = int(sample_rate * 0.001 * window_size)
        self.hop_length = int(sample_rate * 0.001 * stride)
        self.feature_extract_by = feature_extract_by.lower()
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.target_dict = target_dict
        self.spec_augment = SpecAugment(time_mask_para, freq_mask_para, time_mask_num, freq_mask_num)

        if self.feature_extract_by == 'torchaudio':
            if self.feature == 'mel':
                self.transforms = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,  win_length=window_size,
                                                                       hop_length=self.hop_length,  n_fft=self.n_fft,
                                                                       n_mels=n_mels)
                self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
            elif self.feature == 'mfcc':
                self.transforms = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_mels, log_mels=True,
                                                             win_length=window_size, hop_length=self.hop_length,
                                                             n_fft=self.n_fft)

    def parse_audio(self, audio_path, augment_method):
        """
        Parses audio.

        Args:
             audio_path (str): path of audio file
             augment_method (int): flag indication which augmentation method to use.

        Returns: spectrogram
            - **spectrogram** (torch.FloatTensor): Mel-Spectrogram feature from audio file.
        """
        signal = load_audio(audio_path, self.del_silence)

        if signal is None:  # Exception handling
            return None
        elif augment_method == SpectrogramParser.NOISE_INJECTION:  # Noise injection
            signal = self.noise_injector(signal)
            if signal is None:
                return None

        if self.feature == 'mel':
            feature = self.get_melspectrogram_feature(signal)

        elif self.feature == 'spect':
            feature = self.get_spectrogram_feature(signal)

        elif self.feature == 'mfcc':
            feature = self.get_mfcc_feature(signal)

        else:
            raise ValueError("Unsupported feature: {0}".format(self.feature))

        if augment_method == SpectrogramParser.SPEC_AUGMENT:
            feature = self.spec_augment(feature)

        return feature

    def get_melspectrogram_feature(self, signal):
        if self.feature_extract_by == 'torchaudio':
            melspectrogram = self.transforms(torch.FloatTensor(signal))
            melspectrogram = self.amplitude_to_db(melspectrogram)
            melspectrogram = melspectrogram.numpy()

        else:
            melspectrogram = librosa.feature.melspectrogram(signal, self.sample_rate, n_mels=self.n_mels,
                                                            n_fft=self.n_fft, hop_length=self.hop_length)
            melspectrogram = librosa.amplitude_to_db(melspectrogram, ref=np.max)

        if self.normalize:
            melspectrogram -= melspectrogram.mean()

        # Refer to "Sequence to Sequence Learning with Neural Network" paper
        if self.input_reverse:
            melspectrogram = melspectrogram[:, ::-1]
            melspectrogram = torch.FloatTensor(np.ascontiguousarray(np.swapaxes(melspectrogram, 0, 1)))

        return melspectrogram

    def get_spectrogram_feature(self, signal):
        spectrogram = torch.stft(
            torch.FloatTensor(signal),
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=torch.hamming_window(self.n_fft),
            center=False,
            normalized=False,
            onesided=True
        )
        spectrogram = (spectrogram[:, :, 0].pow(2) + spectrogram[:, :, 1].pow(2)).pow(0.5)
        spectrogram = np.log1p(spectrogram.numpy())

        # Refer to "Sequence to Sequence Learning with Neural Network" paper
        if self.input_reverse:
            spectrogram = spectrogram[:, ::-1]
            spectrogram = torch.FloatTensor(np.ascontiguousarray(np.swapaxes(spectrogram, 0, 1)))

        else:
            spectrogram = torch.FloatTensor(spectrogram).transpose(0, 1)

        if self.normalize:
            spectrogram -= spectrogram.mean()

        return spectrogram

    def get_mfcc_feature(self, signal):
        if self.feature_extract_by == 'torchaudio':
            mfcc = self.transforms(torch.FloatTensor(signal))
            mfcc = mfcc.numpy()

        else:
            mfcc = librosa.feature.mfcc(signal, sr=self.sample_rate, n_mfcc=self.n_mels,
                                        n_fft=self.n_fft, hop_length=self.hop_length)

        if self.normalize:
            mfcc -= mfcc.mean()

        # Refer to "Sequence to Sequence Learning with Neural Network" paper
        if self.input_reverse:
            mfcc = mfcc[:, ::-1]
            mfcc = torch.FloatTensor(np.ascontiguousarray(np.swapaxes(mfcc, 0, 1)))

        return mfcc

    def parse_transcript(self, *args, **kwargs):
        raise NotImplementedError
