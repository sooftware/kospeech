"""
Copyright 2020- Kai.Lib
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from torch.utils.data import Dataset

from feature.augmentation import spec_augment
from feature.feature import get_librosa_mfcc
from label.label_func import get_label
import random
from tqdm import trange
from definition import logger

class BaseDataset(Dataset):
    """
    Dataset for audio & label matching
    Args: audio_paths, label_paths, bos_id, eos_id, target_dict
        audio_paths: set of audio path
                Format : [base_dir/KaiSpeech/KaiSpeech_123260.pcm, ... , base_dir/KaiSpeech/KaiSpeech_621245.pcm]
        label_paths: set of label paths
                Format : [base_dir/KaiSpeech/KaiSpeech_label_123260.txt, ... , base_dir/KaiSpeech/KaiSpeech_label_621245.txt]
        bos_id: <s>`s id
        eos_id: </s>`s id
        target_dict: dictionary of filename and labels
                Format : {KaiSpeech_label_FileNum : '5 0 49 4 0 8 190 0 78 115', ... }
    Outputs:
        - **feat**: feature vector for audio
        - **label**: label for audio
    """
    def __init__(self, audio_paths, label_paths, bos_id = 2037, eos_id = 2038, target_dict = None, reverse = True, use_augment = True, augment_ratio = 0.3):
        self.audio_paths = list(audio_paths)
        self.label_paths = list(label_paths)
        self.bos_id, self.eos_id = bos_id, eos_id
        self.target_dict = target_dict
        self.reverse = reverse
        self.augment_ratio = augment_ratio
        self.features = list()
        self.get_feature()
        if use_augment: self.append_augmentation()

    def get_feature(self):
        logger.info("get Features for reducing disking I/O during training...")
        for idx in trange(len(self.audio_paths)):
            feat = get_librosa_mfcc(self.audio_paths[idx], n_mfcc = 33, del_silence = False, input_reverse = self.reverse, format='pcm')
            if feat.size(0) == 1:
                logger.info("Delete label_paths : %s" % self.label_paths[idx])
                del self.label_paths[idx]
            else:
                self.features.append(feat)

    def append_augmentation(self):
        #      0                            augment_end                             end_idx (len(self.audio_paths)
        #      │-----hparams.augment_ratio------│-----------------else-----------------│
        augment_end_idx = int(0 + ((len(self.audio_paths) - 0) * self.augment_ratio))
        logger.info("Applying Augmentation...")
        for idx in trange(augment_end_idx):
            label = get_label(self.label_paths[idx], self.bos_id, self.eos_id, self.target_dict)
            feat = get_librosa_mfcc(self.audio_paths[idx], n_mfcc=33, del_silence=False, input_reverse=self.reverse, format='pcm')
            augmented = spec_augment(feat, T=40, F=30, time_mask_num=2, freq_mask_num=2)
            self.features.append(augmented)
            self.label_paths.append(label)

        # after add data which applied Spec-Augmentation, shuffle
        featureNlabel = list(zip(self.features, self.label_paths))
        random.shuffle(featureNlabel)
        self.features, self.label_paths = zip(*featureNlabel)

    def __len__(self):
        return len(self.features)

    def count(self):
        return len(self.features)

    def getitem(self, idx):
        label = get_label(self.label_paths[idx], self.bos_id, self.eos_id, self.target_dict)
        feat = self.features[idx]
        return feat, label