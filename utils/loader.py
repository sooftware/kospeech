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
import threading
import math
import torch
import random
from utils.define import logger, PAD_TOKEN


class MultiLoader():
    """
    Multi Data Loader using Threads.

    Args:
        dataset: object of BaseDataset
        queue: queue for threading
        batch_size: size of batch
        worker_num: the number of cpu cores used
    """
    def __init__(self, dataset_list, queue, batch_size, worker_num):
        self.dataset_list = dataset_list
        self.queue = queue
        self.batch_size = batch_size
        self.worker_num = worker_num
        self.loader = list()

        for idx in range(self.worker_num):
            self.loader.append(BaseDataLoader(self.dataset_list[idx], self.queue, self.batch_size, idx))

    def start(self):
        for idx in range(self.worker_num):
            self.loader[idx].start()

    def join(self):
        for idx in range(self.worker_num):
            self.loader[idx].join()

class BaseDataLoader(threading.Thread):
    """
    Base Data Loader

    Args:
        dataset (utils.dataset.BaseDataset): object of BaseDataset
        queue (queue): queue for threading
        batch_size (int): size of batch
        thread_id (int): identification of thread
    """
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

    def create_empty_batch(self):
        seqs = torch.zeros(0, 0, 0)
        targets = torch.zeros(0, 0).to(torch.long)
        seq_lengths = list()
        target_lengths = list()
        return seqs, targets, seq_lengths, target_lengths

    def run(self):
        logger.debug('loader %d start' % (self.thread_id))
        while True:
            items = list()
            for _ in range(self.batch_size):
                if self.index >= self.dataset_count:
                    break
                feat, label = self.dataset.get_item(self.index)
                if feat is not None:
                    items.append((feat, label))
                self.index += 1

            if len(items) == 0:
                batch = self.create_empty_batch()
                self.queue.put(batch)
                break

            random.shuffle(items)

            batch = self.collate_fn(items)
            self.queue.put(batch)
        logger.debug('loader %d stop' % (self.thread_id))

def _collate_fn(batch):
    """ functions that pad to the maximum sequence length """
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
    targets.fill_(PAD_TOKEN)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)
        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))
    return seqs, targets, seq_lengths, target_lengths