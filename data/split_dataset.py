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

import math, random
from definition import SOS_token, EOS_token
from data.baseDataset import BaseDataset
from train.save_and_load import save_pickle


def split_dataset(hparams, audio_paths, label_paths, valid_ratio=0.05, target_dict = dict()):
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
    train_dataset_list = []
    train_num = math.ceil(len(audio_paths) * (1 - valid_ratio))
    batch_num = math.ceil(len(audio_paths) / hparams.batch_size)
    valid_batch_num = math.ceil(batch_num * valid_ratio)
    train_batch_num = batch_num - valid_batch_num
    if hparams.use_augment:
        train_batch_num = int( train_batch_num * (1 + hparams.augment_ratio))
    train_num_per_worker = math.ceil(train_num / hparams.worker_num)

    # audio_paths & label_paths shuffled in the same order
    # for seperating train & validation
    data_paths = list(zip(audio_paths, label_paths))
    random.shuffle(data_paths)
    audio_paths, label_paths = zip(*data_paths)

    # seperating the train dataset by the number of workers
    for idx in range(hparams.worker_num):
        train_begin_idx = train_num_per_worker * idx
        train_end_idx = min(train_num_per_worker * (idx + 1), train_num)
        train_dataset_list.append(BaseDataset(audio_paths=audio_paths[train_begin_idx:train_end_idx],
                                              label_paths=label_paths[train_begin_idx:train_end_idx],
                                              bos_id=SOS_token, eos_id=EOS_token, target_dict=target_dict,
                                              input_reverse=hparams.input_reverse, use_augment=hparams.use_augment))

    valid_dataset = BaseDataset(audio_paths=audio_paths[train_num:],
                                label_paths=label_paths[train_num:],
                                bos_id=SOS_token, eos_id=EOS_token,
                                target_dict=target_dict, input_reverse=hparams.input_reverse, use_augment=False)

    save_pickle(train_dataset_list, "./pickle/train_dataset.txt", "dump all train_dataset_list using pickle complete !!")
    save_pickle(valid_dataset, "./pickle/valid_dataset.txt", "dump all valid_dataset using pickle complete !!")

    return train_batch_num, train_dataset_list, valid_dataset