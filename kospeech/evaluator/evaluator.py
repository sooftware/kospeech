# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import queue
import torch
from kospeech.utils import logger
from kospeech.data import AudioDataLoader
from kospeech.decode.search import (
    GreedySearch,
    BeamSearch
)


class Evaluator(object):
    """
    Class to evaluate models with given datasets.

    Args:
        dataset (kospeech.data.data_loader.SpectrogramDataset): dataset for spectrogram & script matching
        batch_size (int): size of batch. recommended batch size is 1.
        device (torch.device): device - 'cuda' or 'cpu'
        num_workers (int): the number of cpu cores used
        print_every (int): to determine whether to store training progress every N timesteps (default: 10)
    """

    def __init__(self, dataset, vocab, batch_size=1, device=None,
                 num_workers=1, print_every=100, decode='greedy', beam_size=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.print_every = print_every

        if decode == 'greedy':
            self.search = GreedySearch(vocab)

        elif decode == 'beam':
            assert beam_size > 1, "beam_size should be greater than 1. You can choose `greedy` search"
            self.search = BeamSearch(vocab, beam_size, batch_size)

        else:
            raise ValueError("Unsupported decode : {0}".format(decode))

    def evaluate(self, model):
        """ Evaluate a model on given dataset and return performance. """
        logger.info('evaluate() start')

        eval_queue = queue.Queue(self.num_workers << 1)
        eval_loader = AudioDataLoader(self.dataset, eval_queue, self.batch_size, thread_id=0, pad_id=0)
        eval_loader.start()

        cer = self.search.search(model, eval_queue, self.device, self.print_every)
        self.search.save_result('data/train_result/%s.csv' % type(self.decoder).__name__)

        logger.info('Evaluate CER: %s' % cer)
        logger.info('evaluate() completed')
        eval_loader.join()
