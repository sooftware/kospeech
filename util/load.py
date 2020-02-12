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

import torch
import pickle
import math
import pandas as pd
from tqdm import trange
from util.define import logger, TRAIN_DATASET_PICKLE_PATH, VALID_DATASET_PICKLE_PATH, ENCODING

def load_targets(label_paths):
    """
    Provides dictionary of filename and labels
    Inputs: label_paths
        - **label_paths**: set of label paths
                Format : [base_dir/KaiSpeech/KaiSpeech_label_123260.txt, ... , base_dir/KaiSpeech/KaiSpeech_label_621245.txt]
    Outputs: target_dict
        - **target_dict**: dictionary of filename and labels
                Format : {KaiSpeech_label_FileNum : '5 0 49 4 0 8 190 0 78 115', ... }
    """
    target_dict = dict()
    for idx in trange(len(label_paths)):
        label_txt = label_paths[idx]
        with open(file=label_txt, mode="r") as f:
            label = f.readline()
            file_num = label_txt.split('/')[-1].split('.')[0].split('_')[-1]
            target_dict['KaiSpeech_label_%s' % file_num] = label

    return target_dict

def load_data_list(data_list_path, dataset_path):
    """
    Provides set of audio path & label path
    Inputs: data_list_path
        - **data_list_path**: csv file with training or test data list
    Outputs: audio_paths, label_paths
        - **audio_paths**: set of audio path
                Format : [base_dir/KaiSpeech/KaiSpeech_123260.pcm, ... , base_dir/KaiSpeech/KaiSpeech_621245.pcm]
        - **label_paths**: set of label path
                Format : [base_dir/KaiSpeech/KaiSpeech_label_123260.txt, ... , base_dir/KaiSpeech/KaiSpeech_label_621245.txt]
    """
    data_list = pd.read_csv(data_list_path, "r", delimiter = ",", encoding=ENCODING)
    audio_paths = list(dataset_path + data_list["audio"])
    label_paths = list(dataset_path + data_list["label"])

    return audio_paths, label_paths

def load_model(filepath):
    logger.info("Load model..")
    model = torch.load(filepath)
    model.eval()
    logger.info("Load model Succesfuuly completely !!")
    return model

def load_pickle(filepath, message=""):
    with open(filepath, "rb") as f:
        load_result = pickle.load(f)
        logger.info(message)
        return load_result

def load_dataset(hparams, audio_paths, valid_ratio=0.05):
    batch_num = math.ceil(len(audio_paths) / hparams.batch_size)
    valid_batch_num = math.ceil(batch_num * valid_ratio)
    train_batch_num = batch_num - valid_batch_num
    if hparams.use_augmentation: train_batch_num *= int(1 + hparams.augment_ratio)

    with open(TRAIN_DATASET_PICKLE_PATH, 'rb') as f:
        train_dataset = pickle.load(f)

    with open(VALID_DATASET_PICKLE_PATH, 'rb') as f:
        valid_dataset = pickle.load(f)

    return train_batch_num, train_dataset, valid_dataset