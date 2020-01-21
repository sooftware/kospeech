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
    def __init__(self, audio_paths, label_paths, bos_id = 2037, eos_id = 2038,
                 target_dict = None, input_reverse = True, use_augmentation = True, augment_ratio = 0.3):
        self.audio_paths = list(audio_paths)
        self.label_paths = list(label_paths)
        self.bos_id, self.eos_id = bos_id, eos_id
        self.target_dict = target_dict
        self.input_reverse = input_reverse
        self.augment_ratio = augment_ratio
<<<<<<< HEAD
        self.is_augment = [False] * len(self.audio_paths)
        if use_augmentation: self.apply_augment()
=======
>>>>>>> 601fe03bcc7840c4b4ade3bcc8d2d1fd4c1debf3

    # 오그멘테이션 적용
    # augmentation list로 하나 만들어서 [True, True ...., False, False] 식으로 만듭시다
    def apply_augment(self):
        #      0                            augment_end                             end_idx (len(self.audio_paths)
        #      │-----hparams.augment_ratio------│-----------------else-----------------│
        augment_end_idx = int(0 + ((len(self.audio_paths) - 0) * self.augment_ratio))
        logger.info("Applying Augmentation...")

        for idx in range(augment_end_idx):
            self.is_augment.append(True)
            self.audio_paths.append(self.audio_paths[idx])
            self.label_paths.append(self.label_paths[idx])

        # after add data which applied Spec-Augmentation, shuffle
        tmp = list(zip(self.audio_paths, self.label_paths, self.is_augment))
        random.shuffle(tmp)
        self.audio_paths, self.label_paths, self.is_augment = zip(*tmp)

    def __len__(self):
        return len(self.audio_paths)

    def count(self):
        return len(self.audio_paths)

    def get_item(self, idx):
        label = get_label(self.label_paths[idx], self.bos_id, self.eos_id, self.target_dict)
        feat = get_librosa_mfcc(self.audio_paths[idx], n_mfcc = 33, del_silence = False, input_reverse = self.input_reverse, format='pcm')
        if feat.size(0) == 1:
            logger.info("Delete label_paths : %s" % self.label_paths[idx])
            del self.label_paths[idx]
            del self.audio_paths[idx]
            label = ''
<<<<<<< HEAD

        if self.is_augment[idx]:
            feat = spec_augment(feat, T=40, F=30, time_mask_num=2, freq_mask_num=2)

=======
>>>>>>> 601fe03bcc7840c4b4ade3bcc8d2d1fd4c1debf3
        return feat, label