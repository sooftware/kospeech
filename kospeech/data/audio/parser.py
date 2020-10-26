# -*- coding: utf-8 -*-
# Soohwan Kim, Seyoung Bae, Cheolhwang Won.
# @ArXiv : KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from kospeech.data.audio.core import load_audio
from torch import Tensor, FloatTensor
from kospeech.data.audio.augment import SpecAugment
from kospeech.data.audio.feature import (
    MelSpectrogram,
    MFCC,
    Spectrogram,
    FilterBank
)


class AudioParser(object):
    """
    Provides inteface of audio parser.

    Note:
        Do not use this class directly, use one of the sub classes.

    Method:
        - **parse_audio()**: abstract method. you have to override this method.
        - **parse_transcript()**: abstract method. you have to override this method.
    """
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def parse_audio(self, *args, **kwargs):
        raise NotImplementedError

    def parse_transcript(self, *args, **kwargs):
        raise NotImplementedError


class SpectrogramParser(AudioParser):
    """
    Parses audio file into (spectrogram / mel spectrogram / mfcc) with various options.

    Args:
        transform_method (str): which feature to use (default: mel)
        sample_rate (int): Sample rate of audio signal. (Default: 16000)
        n_mels (int):  Number of mfc coefficients to retain. (Default: 40)
        frame_length (int): frame length for spectrogram (ms) (Default : 20)
        frame_shift (int): Length of hop between STFT windows. (ms) (Default: 10)
        feature_extract_by (str): which library to use for feature extraction (default: librosa)
        del_silence (bool): flag indication whether to delete silence or not (default: True)
        input_reverse (bool): flag indication whether to reverse input or not (default: True)
        normalize (bool): flag indication whether to normalize spectrum or not (default:True)
        freq_mask_para (int): Hyper Parameter for Freq Masking to limit freq masking length
        time_mask_num (int): how many time-masked area to make
        freq_mask_num (int): how many freq-masked area to make
        sos_id (int): start of sentence token`s identification
        eos_id (int): end of sentence token`s identification
        dataset_path (str): noise dataset path
    """
    VANILLA = 0           # Not apply augmentation
    SPEC_AUGMENT = 1      # SpecAugment

    def __init__(
            self,
            feature_extract_by: str = 'librosa',      # which library to use for feature extraction
            sample_rate: int = 16000,                 # sample rate of audio signal.
            n_mels: int = 80,                         # Number of mfc coefficients to retain.
            frame_length: int = 20,                   # frame length for spectrogram
            frame_shift: int = 10,                    # Length of hop between STFT windows.
            del_silence: bool = False,                # flag indication whether to delete silence or not
            input_reverse: bool = True,               # flag indication whether to reverse input or not
            normalize: bool = False,                  # flag indication whether to normalize spectrum or not
            transform_method: str = 'mel',            # which feature to use [mel, fbank, spect, mfcc]
            freq_mask_para: int = 12,                 # hyper Parameter for Freq Masking to limit freq masking length
            time_mask_num: int = 2,                   # how many time-masked area to make
            freq_mask_num: int = 2,                   # how many freq-masked area to make
            sos_id: int = 1,                          # start of sentence token`s identification
            eos_id: int = 2,                          # end of sentence token`s identification
            dataset_path: str = None                  # noise dataset path
    ) -> None:
        super(SpectrogramParser, self).__init__(dataset_path)
        self.del_silence = del_silence
        self.input_reverse = input_reverse
        self.normalize = normalize
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.spec_augment = SpecAugment(freq_mask_para, time_mask_num, freq_mask_num)

        if transform_method.lower() == 'mel':
            self.transforms = MelSpectrogram(sample_rate, n_mels, frame_length, frame_shift, feature_extract_by)

        elif transform_method.lower() == 'mfcc':
            self.transforms = MFCC(sample_rate, n_mels, frame_length, frame_shift, feature_extract_by)

        elif transform_method.lower() == 'spect':
            self.transforms = Spectrogram(sample_rate, frame_length, frame_shift, feature_extract_by)

        elif transform_method.lower() == 'fbank':
            self.transforms = FilterBank(sample_rate, n_mels, frame_length, frame_shift)

        else:
            raise ValueError("Unsupported feature : {0}".format(transform_method))

    def parse_audio(self, audio_path: str, augment_method: int) -> Tensor:
        """
        Parses audio.

        Args:
             audio_path (str): path of audio file
             augment_method (int): flag indication which augmentation method to use.

        Returns: feature_vector
            - **feature_vector** (torch.FloatTensor): feature from audio file.
        """
        signal = load_audio(audio_path, self.del_silence)
        feature_vector = self.transforms(signal)

        if self.normalize:
            feature_vector -= feature_vector.mean()
            feature_vector /= np.std(feature_vector)

        # Refer to "Sequence to Sequence Learning with Neural Network" paper
        if self.input_reverse:
            feature_vector = feature_vector[:, ::-1]
            feature_vector = FloatTensor(np.ascontiguousarray(np.swapaxes(feature_vector, 0, 1)))
        else:
            feature_vector = FloatTensor(feature_vector).transpose(0, 1)

        if augment_method == SpectrogramParser.SPEC_AUGMENT:
            feature_vector = self.spec_augment(feature_vector)

        return feature_vector

    def parse_transcript(self, *args, **kwargs):
        raise NotImplementedError
