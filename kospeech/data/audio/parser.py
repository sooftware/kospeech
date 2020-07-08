import torch
import numpy as np
from kospeech.data.audio.core import load_audio
from kospeech.data.audio.augment import NoiseInjector, SpecAugment
from kospeech.data.audio.feature import MelSpectrogram, MFCC, Spectrogram


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
        sample_rate (int): Sample rate of audio signal. (Default: 16000)
        n_mels (int):  Number of mfc coefficients to retain. (Default: 40)
        window_size (int): window size (ms) (Default : 20)
        stride (int): Length of hop between STFT windows. (ms) (Default: 10)
        feature_extract_by (str): which library to use for feature extraction(default: librosa)
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
        self.del_silence = del_silence
        self.input_reverse = input_reverse
        self.normalize = normalize
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.target_dict = target_dict
        self.spec_augment = SpecAugment(time_mask_para, freq_mask_para, time_mask_num, freq_mask_num)

        if feature.lower() == 'mel':
            self.transforms = MelSpectrogram(sample_rate, n_mels, window_size, stride, feature_extract_by)
        elif feature.lower() == 'mfcc':
            self.transforms = MFCC(sample_rate, n_mels, window_size, stride, feature_extract_by)
        elif feature.lower() == 'spect':
            self.transforms = Spectrogram(sample_rate, window_size, stride)
        else:
            raise ValueError("Unsupported feature : {0}".format(feature))

    def parse_audio(self, audio_path, augment_method):
        """
        Parses audio.

        Args:
             audio_path (str): path of audio file
             augment_method (int): flag indication which augmentation method to use.

        Returns: feature
            - **feature** (torch.FloatTensor): feature from audio file.
        """
        signal = load_audio(audio_path, self.del_silence)

        if augment_method == SpectrogramParser.NOISE_INJECTION:
            signal = self.noise_injector(signal)

        feature_vector = self.transforms(signal)

        if self.normalize:
            feature_vector -= feature_vector.mean()

        if self.input_reverse:  # Refer to "Sequence to Sequence Learning with Neural Network" paper
            feature_vector = feature_vector[:, ::-1]
            feature_vector = torch.FloatTensor(np.ascontiguousarray(np.swapaxes(feature_vector, 0, 1)))
        else:
            feature_vector = torch.FloatTensor(feature_vector).transpose(0, 1)

        if augment_method == SpectrogramParser.SPEC_AUGMENT:
            feature_vector = self.spec_augment(feature_vector)

        return feature_vector

    def parse_transcript(self, *args, **kwargs):
        raise NotImplementedError
