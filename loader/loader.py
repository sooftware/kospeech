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
import math
import pandas as pd
import torch
import random
import threading
from torch.utils.data import Dataset
from feature_extraction.feature import get_librosa_melspectrogram
from definition import *

def load_data_list(data_list_path):
    """
    Provides set of audio path & label path
    Inputs: data_list_path
        data_list_path: csv file with training or test data list
    Outputs: audio_paths, label_paths
        - **audio_paths**: set of audio path
                Format : [base_dir/KaiSpeech/KaiSpeech_123260.pcm, ... , base_dir/KaiSpeech/KaiSpeech_621245.pcm]
        - **label_paths**: set of label path
                Format : [base_dir/KaiSpeech/KaiSpeech_label_123260.txt, ... , base_dir/KaiSpeech/KaiSpeech_label_621245.txt]
    """
    data_list = pd.read_csv(data_list_path, "r", delimiter = ",", encoding="UTF-8")
    audio_paths = list(DATASET_PATH + data_list["audio"])
    label_paths = list(DATASET_PATH + data_list["label"])
    return audio_paths, label_paths


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
    for label_txt in label_paths:
        f = open(file=label_txt, mode="r")
        label = f.readline()
        f.close()
        file_num = label_txt.split('/')[-1].split('.')[0].split('_')[-1]
        target_dict['KaiSpeech_label_'+file_num] = label
    return target_dict


def get_label(label_path, bos_id=2037, eos_id=2038, target_dict=None):
    """
    Provides specific file`s label to list format.
    Inputs: filepath, bos_id, eos_id, target_dict
        - **filepath**: specific path of label file
        - **bos_id**: <s>`s id
        - **eos_id**: </s>`s id
        - **target_dict**: dictionary of filename and labels
                Format : {KaiSpeech_label_FileNum : '5 0 49 4 0 8 190 0 78 115', ... }
    Outputs: label
        - **label**: list of bos + sequence of label + eos
                Format : [<s>, 5, 0, 49, 4, 0, 8, 190, 0, 78, 115, </s>]
    """
    if target_dict == None: logger.info("target_dict is None")
    key = label_path.split('/')[-1].split('.')[0]
    script = target_dict[key]
    tokens = script.split(' ')

    label = list()
    label.append(bos_id)
    for token in tokens:
        label.append(int(token))
    label.append(eos_id)
    return label

class BaseDataset(Dataset):
    """
    Inputs: audio_paths, label_paths, bos_id, eos_id, target_dict
        - **audio_paths**: set of audio path
                Format : [base_dir/KaiSpeech/KaiSpeech_123260.pcm, ... , base_dir/KaiSpeech/KaiSpeech_621245.pcm]
        - **label_paths**: set of label paths
                Format : [base_dir/KaiSpeech/KaiSpeech_label_123260.txt, ... , base_dir/KaiSpeech/KaiSpeech_label_621245.txt]
        - **bos_id**: <s>`s id
        - **eos_id**: </s>`s id
        - **target_dict**: dictionary of filename and labels
                Format : {KaiSpeech_label_FileNum : '5 0 49 4 0 8 190 0 78 115', ... }
    """
    def __init__(self, audio_paths, label_paths, bos_id = 2037, eos_id = 2038, target_dict = None):
        self.audio_paths = audio_paths
        self.label_paths = label_paths
        self.bos_id, self.eos_id = bos_id, eos_id
        self.target_dict = target_dict

    def __len__(self):
        return len(self.audio_paths)

    def count(self):
        return len(self.audio_paths)

    def getitem(self, idx):
        # 리스트 형식으로 label을 저장
        script = get_label(self.label_paths[idx], self.bos_id, self.eos_id, self.target_dict)
        # 음성데이터에 대한 feature를 feat에 저장 -> tensor 형식
        feat = get_librosa_melspectrogram(self.audio_paths[idx], n_mels = 80, del_silence = True, type_='log_mel')
        return feat, script

def _collate_fn(batch):
    PAD = 0
    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    seq_lengths = [len(s[0]) for s in batch]
    target_lengths = [len(s[1]) for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]

    max_seq_size = max_seq_sample.size(0)
    max_target_size = len(max_target_sample)

    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)

    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(PAD)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)
        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))
    return seqs, targets, seq_lengths, target_lengths

class BaseDataLoader(threading.Thread):
    def __init__(self, dataset, queue, batch_size, thread_id):
        threading.Thread.__init__(self)
        self.collate_fn = _collate_fn
        self.dataset = dataset
        self.queue = queue
        self.index = 0
        self.batch_size = batch_size
        self.dataset_count = dataset.count()
        self.thread_id = thread_id

    def count(self):
        return math.ceil(self.dataset_count / self.batch_size)

    # 큐에 들어가는 batch를 만드는 함수
    def create_empty_batch(self):
        seqs = torch.zeros(0, 0, 0)  #  3차원의 0 벡터
        targets = torch.zeros(0, 0).to(torch.long)
        seq_lengths = list()
        target_lengths = list()
        return seqs, targets, seq_lengths, target_lengths

    def run(self):
        logger.debug('loader %d start' % (self.thread_id))
        while True:
            items = list()

            for i in range(self.batch_size): 
                if self.index >= self.dataset_count:
                    break
                items.append(self.dataset.getitem(self.index))
                self.index += 1

            if len(items) == 0:
                batch = self.create_empty_batch()
                # 큐에 batch 삽입
                self.queue.put(batch)
                break

            random.shuffle(items)

            batch = self.collate_fn(items)
            # 큐에 batch 삽입
            self.queue.put(batch)
        logger.debug('loader %d stop' % (self.thread_id))

# BaseLoader()를 여러 개 호출하는 클래스
class MultiLoader():
    def __init__(self, dataset_list, queue, batch_size, worker_size):
        self.dataset_list = dataset_list
        self.queue = queue
        self.batch_size = batch_size
        self.worker_size = worker_size
        self.loader = list()

        for i in range(self.worker_size):
            self.loader.append(BaseDataLoader(self.dataset_list[i], self.queue, self.batch_size, i))

    def start(self):
        # BaseDataLoader run()을 실행!!
        for i in range(self.worker_size):
            self.loader[i].start()

    def join(self):
        for i in range(self.worker_size):
            self.loader[i].join()