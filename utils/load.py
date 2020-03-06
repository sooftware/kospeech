"""
Copyright 2020- Kai.Lib

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pickle
import math
import pandas as pd
import csv
from tqdm import trange
from utils.define import logger, TRAIN_DATASET_PICKLE_PATH, VALID_DATASET_PICKLE_PATH
from utils.save import save_pickle


def load_targets(label_paths):
    """
    Provides dictionary of filename and labels

    Parameters
    -----------
        - **label_paths** (list): set of label paths

    Returns
    ---------
        - **target_dict** (dict): dictionary of filename and labels
    """
    target_dict = dict()
    for idx in trange(len(label_paths)):
        label_txt = label_paths[idx]
        with open(file=label_txt, mode="r") as f:
            label = f.readline()
            file_num = label_txt.split('/')[-1].split('.')[0].split('_')[-1]
            target_dict['KaiSpeech_label_%s' % file_num] = label
    save_pickle(target_dict, "./data/pickle/target_dict.bin", message="target_dict save complete !!")
    return target_dict

def load_data_list(data_list_path, dataset_path):
    """
    Provides set of audio path & label path

    Parameters
    -----------
        - **data_list_path** (list): csv file with training or test data list

    Returns
    --------
        - **audio_paths** (list): set of audio path
        - **label_paths** (list): set of label path
    """
    data_list = pd.read_csv(data_list_path, "r", delimiter = ",", encoding="cp949")
    audio_paths = list(dataset_path + data_list["audio"])
    label_paths = list(dataset_path + data_list["label"])

    return audio_paths, label_paths

def load_label(label_path, encoding='utf-8'):
    """
    Provides char2id, id2char

    Parameters
    -----------
        - **label_path** (list): csv file with character labels

    Returns
    --------
        - **char2id** (dict): char2id[ch] = id
        - **id2char** (dict): id2char[id] = ch
    """
    char2id = dict()
    id2char = dict()
    with open(label_path, 'r', encoding=encoding) as f:
        labels = csv.reader(f, delimiter=',')
        next(labels)

        for row in labels:
            char2id[row[1]] = row[0]
            id2char[int(row[0])] = row[1]

    return char2id, id2char

def load_pickle(filepath, message=""):
    """
    load pickle file

    Parameters
    -----------
        - **filepath** (str): Path to pickle file to load

    Returns
    --------
        -**load_result** : load result of pickle
    """
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