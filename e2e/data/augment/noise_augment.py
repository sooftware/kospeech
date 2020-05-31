import os
import random
import numpy as np
from e2e.utils import logger
from e2e.data.preprocess.core import split


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

    Inputs: sound
        - **sound**: sound from pcm file

    Returns: sound
        - **sound**: noise added sound
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

    def __call__(self, sound):
        noise = np.random.choice(self.dataset)
        noise_level = np.random.uniform(0, self.noise_level)

        sound_length = len(sound)
        noise_length = len(noise)

        if sound_length >= noise_length:
            noise_start = int(np.random.rand() * (sound_length - noise_length))
            noise_end = int(noise_start + noise_length)
            sound[noise_start: noise_end] += noise * noise_level

        else:
            sound += noise[:sound_length] * noise_level

        return sound

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
            noise /= 32767
            return noise

        except RuntimeError:
            logger.info("RuntimeError in {0}".format(audio_path))
            return None

        except ValueError:
            logger.info("RuntimeError in {0}".format(audio_path))
            return None
