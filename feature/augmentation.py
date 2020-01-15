import numpy as np
import random

def spec_augment(feat, T=40, F=30, time_mask_num=2, freq_mask_num=2):
    """
    Provides Augmentation for audio
    Inputs:
        - **feat**: input data feature
    Args:
        T: Hyper Parameter for Time Masking to limit time masking length
        F: Hyper Parameter for Freq Masking to limit freq masking length
        time_mask_num: how many time-masked area to make
        freq_mask_num: how many freq-masked area to make
    Outputs:
        - **augmented**: Applied Spec-Augmentation to feat

    Reference :
        「SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition」Google Brain Team. 2019.12.03
    """
    n_mfcc = feat.size(1)
    feat_len = feat.size(0)
    augmented = feat.clone()

    # time mask
    for _ in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=T)
        t = int(t)
        t0 = random.randint(0, feat_len - t)
        augmented[t0:t0+t, :] = 0

    # freq mask
    for _ in range(freq_mask_num):
        f = np.random.uniform(low=0.0, high=F)
        f = int(f)
        f0 = random.randint(0, n_mfcc - f)
        augmented[:, f0:f0 + f] = 0

    return augmented
