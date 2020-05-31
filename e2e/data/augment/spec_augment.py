import random
import numpy as np


class SpecAugment(object):
    """
    Provides Spec Augment. A simple data augmentation method for speech recognition.
    This concept proposed in https://arxiv.org/abs/1904.08779

    Args:
        time_mask_para (int): maximum time masking length
        freq_mask_para (int): maximum frequency masking length
        time_mask_num (int): how many times to apply time masking
        freq_mask_num (int): how many times to apply frequency masking

    Inputs: spectrogram
        - **spectrogram** (torch.FloatTensor): spectrogram feature from audio file.

    Returns: spectrogram:
        - **spectrogram**: masked spectrogram feature.
    """
    def __init__(self, time_mask_para, freq_mask_para, time_mask_num, freq_mask_num):
        self.time_mask_para = time_mask_para
        self.freq_mask_para = freq_mask_para
        self.time_mask_num = time_mask_num
        self.freq_mask_num = freq_mask_num

    def __call__(self, spectrogram):
        """ Provides SpecAugmentation for audio """
        time_axis_length = spectrogram.size(0)
        freq_axis_length = spectrogram.size(1)

        # time mask
        for _ in range(self.time_mask_num):
            t = np.random.uniform(low=0.0, high=self.time_mask_para)
            t = int(t)
            if time_axis_length - t > 0:
                t0 = random.randint(0, time_axis_length - t)
                spectrogram[t0: t0 + t, :] = 0

        # freq mask
        for _ in range(self.freq_mask_num):
            f = np.random.uniform(low=0.0, high=self.freq_mask_para)
            f = int(f)
            f0 = random.randint(0, freq_axis_length - f)
            spectrogram[:, f0: f0 + f] = 0

        return spectrogram
