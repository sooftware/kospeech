# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from kospeech.utils import logger
from kospeech.data.audio.core import load_audio
from torch import Tensor, FloatTensor
from kospeech.data.audio.augment import SpecAugment
from kospeech.data.audio.feature import (
    MelSpectrogram,
    MFCC,
    Spectrogram,
    FilterBank,
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
            dataset_path: str = None,                 # noise dataset path
            audio_extension: str = 'pcm',             # audio extension
    ) -> None:
        super(SpectrogramParser, self).__init__(dataset_path)
        self.del_silence = del_silence
        self.input_reverse = input_reverse
        self.normalize = normalize
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.spec_augment = SpecAugment(freq_mask_para, time_mask_num, freq_mask_num)
        self.audio_extension = audio_extension

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
        signal = load_audio(audio_path, self.del_silence, extension=self.audio_extension)

        if signal is None:
            logger.info("Audio is None : {0}".format(audio_path))
            return None

        feature = self.transforms(signal)

        if self.normalize:
            feature -= feature.mean()
            feature /= np.std(feature)

        # Refer to "Sequence to Sequence Learning with Neural Network" paper
        if self.input_reverse:
            feature = feature[:, ::-1]
            feature = FloatTensor(np.ascontiguousarray(np.swapaxes(feature, 0, 1)))
        else:
            feature = FloatTensor(feature).transpose(0, 1)

        if augment_method == SpectrogramParser.SPEC_AUGMENT:
            feature = self.spec_augment(feature)

        return feature

    def parse_transcript(self, *args, **kwargs):
        raise NotImplementedError
