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

import os
import random
import numpy as np
from torch import Tensor
from kospeech.utils import logger
from kospeech.data.audio.core import split


class SpecAugment(object):
    """
    Provides Spec Augment. A simple data augmentation method for speech recognition.
    This concept proposed in https://arxiv.org/abs/1904.08779

    Args:
        freq_mask_para (int): maximum frequency masking length
        time_mask_num (int): how many times to apply time masking
        freq_mask_num (int): how many times to apply frequency masking

    Inputs: feature_vector
        - **feature_vector** (torch.FloatTensor): feature vector from audio file.

    Returns: feature_vector:
        - **feature_vector**: masked feature vector.
    """
    def __init__(self, freq_mask_para: int = 18, time_mask_num: int = 10, freq_mask_num: int = 2) -> None:
        self.freq_mask_para = freq_mask_para
        self.time_mask_num = time_mask_num
        self.freq_mask_num = freq_mask_num

    def __call__(self, feature: Tensor) -> Tensor:
        """ Provides SpecAugmentation for audio """
        time_axis_length = feature.size(0)
        freq_axis_length = feature.size(1)
        time_mask_para = time_axis_length / 20      # Refer to "Specaugment on large scale dataset" paper

        # time mask
        for _ in range(self.time_mask_num):
            t = int(np.random.uniform(low=0.0, high=time_mask_para))
            t0 = random.randint(0, time_axis_length - t)
            feature[t0: t0 + t, :] = 0

        # freq mask
        for _ in range(self.freq_mask_num):
            f = int(np.random.uniform(low=0.0, high=self.freq_mask_para))
            f0 = random.randint(0, freq_axis_length - f)
            feature[:, f0: f0 + f] = 0

        return feature


class NoiseInjector(object):
    """
    Provides noise injection for noise augmentation.
    The noise augmentation process is as follows:

    Step 1: Randomly sample audios by `noise_size` from dataset
    Step 2: Extract noise from `audio_paths`
    Step 3: Add noise to sound

    Args:
        dataset_path (str): path of dataset
        noiseset_size (int): size of noise dataset
        sample_rate (int): sampling rate
        noise_level (float): level of noise

    Inputs: signal
        - **signal**: signal from pcm file

    Returns: signal
        - **signal**: noise added signal
    """
    def __init__(self, dataset_path, noiseset_size, sample_rate=16000, noise_level=0.7):
        if not os.path.exists(dataset_path):
            logger.info("Directory doesn`t exist: {0}".format(dataset_path))
            raise IOError

        logger.info("Create Noise injector...")

        self.noiseset_size = noiseset_size
        self.sample_rate = sample_rate
        self.noise_level = noise_level
        self.audio_paths = self.create_audio_paths(dataset_path)
        self.dataset = self.create_noiseset(dataset_path)

        logger.info("Create Noise injector complete !!")

    def __call__(self, signal):
        noise = np.random.choice(self.dataset)
        noise_level = np.random.uniform(0, self.noise_level)

        signal_length = len(signal)
        noise_length = len(noise)

        if signal_length >= noise_length:
            noise_start = int(np.random.rand() * (signal_length - noise_length))
            noise_end = int(noise_start + noise_length)
            signal[noise_start: noise_end] += noise * noise_level

        else:
            signal += noise[:signal_length] * noise_level

        return signal

    def create_audio_paths(self, dataset_path):
        audio_paths = list()
        data_list = os.listdir(dataset_path)
        data_list_size = len(data_list)

        while True:
            index = int(random.random() * data_list_size)

            if data_list[index].endswith('.pcm'):
                audio_paths.append(data_list[index])

            if len(audio_paths) == self.noiseset_size:
                break

        return audio_paths

    def create_noiseset(self, dataset_path):
        dataset = list()

        for audio_path in self.audio_paths:
            path = os.path.join(dataset_path, audio_path)
            noise = self.extract_noise(path)

            if noise is not None:
                dataset.append(noise)

        return dataset

    def extract_noise(self, audio_path):
        try:
            signal = np.memmap(audio_path, dtype='h', mode='r').astype('float32')
            non_silence_indices = split(signal, top_db=30)

            for (start, end) in non_silence_indices:
                signal[start:end] = 0

            noise = signal[signal != 0]
            return noise / 32767

        except RuntimeError:
            logger.info("RuntimeError in {0}".format(audio_path))
            return None

        except ValueError:
            logger.info("RuntimeError in {0}".format(audio_path))
            return None
