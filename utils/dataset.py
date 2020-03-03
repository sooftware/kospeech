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
from utils.feature import spec_augment, get_librosa_melspectrogram
from utils.label import get_label
from utils.define import logger, SOS_TOKEN, EOS_TOKEN


class BaseDataset(Dataset):
    """
    Dataset for audio & label matching

    Args: audio_paths, label_paths, bos_id, eos_id, target_dict
        - **audio_paths** (list): set of audio path
                Format : [base_dir/KaiSpeech/KaiSpeech_123260.pcm, ... , base_dir/KaiSpeech/KaiSpeech_621245.pcm]
        - **label_paths** (list): set of label paths
                Format : [base_dir/KaiSpeech/KaiSpeech_label_123260.txt, ... , base_dir/KaiSpeech/KaiSpeech_label_621245.txt]
        - **bos_id** (int): <s>`s id
        - **eos_id** (int): </s>`s id
        - **target_dict** (dict): dictionary of filename and labels
                Format : {KaiSpeech_label_FileNum : '5 0 49 4 0 8 190 0 78 115', ... }
    Outputs:
        - **feat**: feature vector for audio
        - **label**: label for audio
    """
    def __init__(self, audio_paths, label_paths, sos_id, eos_id,
                 target_dict = None, input_reverse = True, use_augment = True,
                 batch_size = None, augment_ratio = 1.0, pack_by_length = True):
        self.audio_paths = list(audio_paths)
        self.label_paths = list(label_paths)
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.batch_size = batch_size
        self.target_dict = target_dict
        self.input_reverse = input_reverse
        self.augment_ratio = augment_ratio
        self.augment_flags = [False] * len(self.audio_paths)
        self.pack_by_length = pack_by_length
        if use_augment:
            self.augmentation()
        if pack_by_length:
            self.sort_by_length()
            self.audio_paths, self.label_paths, self.augment_flags = self.batch_shuffle(remain_drop=False)
        else:
            bundle = list(zip(self.audio_paths, self.label_paths, self.augment_flags))
            random.shuffle(bundle)
            self.audio_paths, self.label_paths, self.augment_flags = zip(*bundle)

    def __len__(self):
        return len(self.audio_paths)

    def count(self):
        return len(self.audio_paths)

    def get_item(self, idx):
        label = get_label(self.label_paths[idx], sos_id = self.sos_id, eos_id = self.eos_id, target_dict = self.target_dict)
        feat = get_librosa_melspectrogram(self.audio_paths[idx], n_mels = 128, mel_type='log_mel', input_reverse = self.input_reverse)
        # exception handling
        if feat is None:
            return None, None
        if self.augment_flags[idx]:
            feat = spec_augment(feat, T = 70, F = 20, time_mask_num = 2, freq_mask_num = 2 )
        return feat, label

    def augmentation(self):
        """ Apply Spec-Augmentation """
        augment_end_idx = int(0 + ((len(self.audio_paths) - 0) * self.augment_ratio))
        logger.info("Applying Augmentation...")

        for idx in range(augment_end_idx):
            self.augment_flags.append(True)
            self.audio_paths.append(self.audio_paths[idx])
            self.label_paths.append(self.label_paths[idx])

    def shuffle(self):
        """ Shuffle Dataset """
        if self.pack_by_length:
            self.audio_paths, self.label_paths, self.augment_flags = self.batch_shuffle(remain_drop=False)
        else:
            bundle = list(zip(self.audio_paths, self.label_paths, self.augment_flags))
            random.shuffle(bundle)
            self.audio_paths, self.label_paths, self.augment_flags = zip(*bundle)

    def sort_by_length(self):
        """ descending sort by sequence length """
        target_lengths = list()
        for idx, label_path in enumerate(self.label_paths):
            key = label_path.split('/')[-1].split('.')[0]
            target_lengths.append(len(self.target_dict[key].split()))

        bundle = list(zip(target_lengths, self.audio_paths, self.label_paths, self.augment_flags))
        junk, self.audio_paths, self.label_paths, self.augment_flags = zip(*sorted(bundle, reverse=True))

    def batch_shuffle(self, remain_drop = False):
        """ batch shuffle """
        total_audio_batch, total_label_batch, total_augment_flag = [], [], []
        audio_paths, label_paths, augment_flags = [], [], []
        index = 0

        while True:
            if index == len(self.audio_paths):
                if len(audio_paths) != 0:
                    total_audio_batch.append(audio_paths)
                    total_label_batch.append(label_paths)
                    total_augment_flag.append(augment_flags)
                break
            if len(audio_paths) == self.batch_size:
                total_audio_batch.append(audio_paths)
                total_label_batch.append(label_paths)
                total_augment_flag.append(augment_flags)
                audio_paths, label_paths, augment_flags = [], [], []
            audio_paths.append(self.audio_paths[index])
            label_paths.append(self.label_paths[index])
            augment_flags.append(self.augment_flags[index])
            index += 1

        remain_audio, remain_label, remain_augment_flag = total_audio_batch[-1], total_label_batch[-1], total_augment_flag[-1]
        total_audio_batch, total_label_batch, total_augment_flag = total_audio_batch[:-1], total_label_batch[:-1], total_augment_flag[:-1]

        bundle = list(zip(total_audio_batch, total_label_batch, total_augment_flag))
        random.shuffle(bundle)
        total_audio_batch, total_label_batch, total_augment_flag = zip(*bundle)

        audio_paths, label_paths, augment_flags = [], [], []

        for (audio_batch, label_batch, augment_flag) in zip(total_audio_batch, total_label_batch, total_augment_flag):
            audio_paths.extend(audio_batch)
            label_paths.extend(label_batch)
            augment_flags.extend(augment_flag)

        audio_paths = list(audio_paths)
        label_paths = list(label_paths)
        augment_flags = list(augment_flags)

        if not remain_drop:
            audio_paths.extend(remain_audio)
            label_paths.extend(remain_label)
            augment_flags.extend(remain_augment_flag)

        return audio_paths, label_paths, augment_flags


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
    train_dataset_list = list()
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
        train_dataset_list.append(BaseDataset(
                                    audio_paths=audio_paths[train_begin_index:train_end_index],
                                    label_paths=label_paths[train_begin_index:train_end_index],
                                    sos_id=SOS_TOKEN, eos_id=EOS_TOKEN,
                                    target_dict=target_dict,
                                    input_reverse=hparams.input_reverse,
                                    use_augment=hparams.use_augment,
                                    batch_size=hparams.batch_size,
                                    augment_ratio=hparams.augment_ratio,
                                    pack_by_length=hparams.pack_by_length
                                )
        )

    valid_dataset = BaseDataset(
                        audio_paths=audio_paths[train_num:],
                        label_paths=label_paths[train_num:],
                        sos_id=SOS_TOKEN, eos_id=EOS_TOKEN,
                        batch_size=hparams.batch_size,
                        target_dict=target_dict,
                        input_reverse=hparams.input_reverse,
                        use_augment=False,
                        pack_by_length=hparams.pack_by_length
    )

    logger.info("split dataset complete !!")
    return train_time_step, train_dataset_list, valid_dataset