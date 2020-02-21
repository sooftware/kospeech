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
import random
import math
from torch.utils.data import Dataset
from utils.augment import spec_augment
from utils.feature import get_librosa_mfcc
from utils.label import get_label
from utils.define import logger, SOS_token, EOS_token
from utils.save import save_pickle


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
    def __init__(self, audio_paths, label_paths, sos_id = 2037, eos_id = 2038,
                 target_dict = None, input_reverse = True, use_augment = True,
                 augment_ratio = 0.3):
        self.audio_paths = list(audio_paths)
        self.label_paths = list(label_paths)
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.target_dict = target_dict
        self.input_reverse = input_reverse
        self.augment_ratio = augment_ratio
        self.is_augment = [False] * len(self.audio_paths)
        if use_augment:
            self.apply_augment()

    def __len__(self):
        return len(self.audio_paths)

    def count(self):
        return len(self.audio_paths)

    def get_item(self, idx):
        label = get_label(
            filepath = self.label_paths[idx],
            sos_id = self.sos_id,
            eos_id = self.eos_id,
            target_dict = self.target_dict
        )
        feat = get_librosa_mfcc(
            filepath = self.audio_paths[idx],
            n_mfcc = 33,
            del_silence = False,
            input_reverse = self.input_reverse,
            format = 'pcm'
        )
        # exception handling
        if feat is None:
            return None, None
        if self.is_augment[idx]:
            feat = spec_augment(
                feat = feat,
                T=40,
                F=15,
                time_mask_num=2,
                freq_mask_num=2
            )
        return feat, label

    def apply_augment(self):
        """
        Apply Spec-Augmentation
        Comment:
            - **audio_paths**: [KaiSpeech_135248.pcm, KaiSpeech_453892.pcm, ......, KaiSpeech_357891.pcm]
            - **label_paths**: [KaiSpeech_135248.txt, KaiSpeech_453892.txt, ......, KaiSpeech_357891.txt]
            - **is_augment**: [True, False, ......, False]
            Apply SpecAugmentation if is_augment[idx] == True otherwise, it doesn`t

        0                            augment_end                             end_idx (len(self.audio_paths)
        │-----hparams.augment_ratio------│-----------------else-----------------│
        """
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

    def shuffle(self):
        """Shuffle Dataset"""
        tmp = list(zip(self.audio_paths, self.label_paths, self.is_augment))
        random.shuffle(tmp)
        self.audio_paths, self.label_paths, self.is_augment = zip(*tmp)


def split_dataset(hparams, audio_paths, label_paths, valid_ratio=0.05, target_dict = None):
    """
    Dataset split into training and validation Dataset.
    Args:
        valid_ratio: ratio for validation data
    Inputs: hparams, audio_paths, label_paths, target_dict
        - **hparams**: set of hyper parameters
        - **audio_paths**: set of audio path
                Format : [base_dir/KaiSpeech/KaiSpeech_123260.pcm, ... , base_dir/KaiSpeech/KaiSpeech_621245.pcm]
        - **label_paths**: set of label path
                Format : [base_dir/KaiSpeech/KaiSpeech_label_123260.txt, ... , base_dir/KaiSpeech/KaiSpeech_label_621245.txt]
        - **target_dict**: dictionary of filename and labels
                {KaiSpeech_label_FileNum : '5 0 49 4 0 8 190 0 78 115', ... }
    Local Variables:
        - **train_num**: num of training data
        - **batch_num**: total num of batch
        - **valid_batch_num**: num of batch for validation
        - **train_num_per_worker**: num of train data per CPU core
        - **data_paths**: temp variables for audio_paths and label_paths to be shuffled in the same order
        - **train_begin_idx**: begin index of worker`s training dataset
        - **train_end_idx**: end index of worker`s training dataset
    Outputs: train_batch_num, train_dataset_list, valid_dataset
        - **train_batch_num**: num of batch for training
        - **train_dataset_list**: list of training data
        - **valid_dataset**: list of validation data
    """
    logger.info("split dataset start !!")
    train_dataset_list = []
    train_num = math.ceil(len(audio_paths) * (1 - valid_ratio))
    total_time_step = math.ceil(len(audio_paths) / hparams.batch_size)
    valid_time_step = math.ceil(total_time_step * valid_ratio)
    train_time_step = total_time_step - valid_time_step
    if hparams.use_augment:
        train_time_step = int( train_time_step * (1 + hparams.augment_ratio))
    train_num_per_worker = math.ceil(train_num / hparams.worker_num)

    # audio_paths & label_paths shuffled in the same order
    # for seperating train & validation
    data_paths = list(zip(audio_paths, label_paths))
    random.shuffle(data_paths)
    audio_paths, label_paths = zip(*data_paths)

    # seperating the train dataset by the number of workers
    for idx in range(hparams.worker_num):
        train_begin_index = train_num_per_worker * idx
        train_end_index = min(train_num_per_worker * (idx + 1), train_num)
        train_dataset_list.append(BaseDataset(audio_paths=audio_paths[train_begin_index:train_end_index],
                                              label_paths=label_paths[train_begin_index:train_end_index],
                                              sos_id=SOS_token, eos_id=EOS_token, target_dict=target_dict,
                                              input_reverse=hparams.input_reverse, use_augment=hparams.use_augment,
                                              augment_ratio=hparams.augment_ratio))
    valid_dataset = BaseDataset(audio_paths=audio_paths[train_num:],
                                label_paths=label_paths[train_num:],
                                sos_id=SOS_token, eos_id=EOS_token,
                                target_dict=target_dict, input_reverse=hparams.input_reverse, use_augment=False)

    save_pickle(train_dataset_list, "./data/pickle/train_dataset.bin", "dump all train_dataset_list using pickle complete !!")
    save_pickle(valid_dataset, "./data/pickle/valid_dataset.bin", "dump all valid_dataset using pickle complete !!")
    logger.info("split dataset complete !!")
    return train_time_step, train_dataset_list, valid_dataset