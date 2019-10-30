"""
Copyright 2019-present NAVER Corp.
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

#-*- coding: utf-8 -*-

import sys
sys.path.append('..')
import math
import torch
import random
import threading
import logging
from torch.utils.data import Dataset, DataLoader
from feature_extraction.feature import get_librosa_mfcc, get_librosa_melspectrogram

logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)

def load_targets(path, target_dict):
    with open(path, 'r') as f:
        for no, line in enumerate(f):
            key, target = line.strip().split(',')
            target_dict[key] = target

# 정답을 리스트 형식으로 담아주는 함수
def get_script(filepath, bos_id, eos_id, target_dict):
    # key : 41_0508_171_0_08412_03.script 중 41_0508_171_0_08412_03 -> label
    key = filepath.split('/')[-1].split('.')[0]
    # 41_0508_171_0_08412_03 에 해당하는 label 텍스트일 듯

    script = target_dict[key.split('\\')[1]]
    # 텍스트를 ' ' 기준으로 나눈다 -> 10 268 10207 와 같이 레이블 되어 있으니까!!
    tokens = script.split(' ')

    # result를 담을 리스트 초기화
    result = list()

    # result에 bos_id로 시작을 표시하는 듯
    # Begin Of Script 일 듯
    result.append(bos_id)

    # 나눈 token들을 result에 추가하는 듯
    for i in range(len(tokens)):
        if len(tokens[i]) > 0:
            result.append(int(tokens[i]))
    # 마지막 End Of Script 표시를 해주는 듯
    result.append(eos_id)
    return result

class BaseDataset(Dataset):
    # wav_paths : wav_path가 모여있는 리스트
    # script_paths : script_path가 모여있는 리스트 script == label
    # bos_id : Begin Of Script -> script의 시작을 표시하는 Number
    # eos_id : End Of Script -> script의 끝을 표시하는 Number
    def __init__(self, wav_paths, script_paths, bos_id = 1307, eos_id = 1308, target_dict = dict()):
        self.wav_paths = wav_paths
        self.script_paths = script_paths
        self.bos_id, self.eos_id = bos_id, eos_id
        self.target_dict = target_dict

    def __len__(self):
        return len(self.wav_paths)

    def count(self):
        return len(self.wav_paths)

    def getitem(self, idx):
        # 음성데이터에 대한 feature를 feat에 저장 -> tensor 형식
        feat = get_librosa_mfcc(self.wav_paths[idx], n_mfcc = 40)
        # 리스트 형식으로 label을 저장
        script = get_script(self.script_paths[idx], self.bos_id, self.eos_id, self.target_dict)
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
                #logger.info('BaseDataLoader 들어옴')
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

